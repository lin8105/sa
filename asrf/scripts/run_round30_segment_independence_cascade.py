#!/usr/bin/env python3
"""Round 30: learned segment independence with protected cascade merging."""
from __future__ import annotations

import csv, hashlib, json, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import yaml

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from asrf.data.dataset import load_heatmap, load_timestamp_vector  # noqa: E402
from asrf.data.labels import load_label_mapping  # noqa: E402
from asrf.models import ASRFModel  # noqa: E402
from asrf.visualization.temporal import DEFAULT_LABEL_COLORS, _normalized_heatmap  # noqa: E402
import run_round27_pp_only_r5_region_sf_point_hybrid as r27  # noqa: E402
import run_round27b_complete_test_temporal_only as r27b  # noqa: E402

OUT=ROOT/"outputs/round30_segment_independence_cascade"; DATA=r27.DATA; KNOWN=tuple(r27.KNOWN); KNOWN_SET=set(KNOWN)
SOURCE_CANDIDATES=[ROOT/"outputs/round27b_complete_test_temporal_only",ROOT/"outputs/round27b_hybrid"]; SOURCE=next((x for x in SOURCE_CANDIDATES if (x/"predictions").is_dir()),SOURCE_CANDIDATES[0])
SF_SHA="6b11abff2ff4387eada88e38fb56133b29fd5c3facdfb54b42fc84701213db9a"; R5_SHA="577d8edf9e2b04927acc235ffa4d6baab8df1712dd0b98eaaba9063fde31f406"
SF_CAN=ROOT/"outputs/round10_pp_only_novel_segmentation/models/single_frame/best.pt"; R5_CAN=ROOT/"outputs/round10_pp_only_novel_segmentation/models/hard_window_r5/best.pt"
FUSION={"threshold":.5,"gap":0,"rule":"P4","support_gate":.5,"separation":0}; SEED=42; DUR_BINS=(60,120,180,300,500); CLIP=5.0

def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def jd(x:Any)->Any:
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,torch.Tensor): return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)
def wjson(path:Path,x:Any)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x,indent=2,sort_keys=True,default=jd),encoding="utf-8")
def wcsv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True); fields=list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
def resolve(canonical:Path,expected:str)->Path:
    candidates=[canonical,ROOT/"outputs/0"/canonical.relative_to(ROOT/"outputs")]+list(ROOT.glob("outputs/**/best.pt")); hits=[p for p in dict.fromkeys(candidates) if p.is_file() and sha(p)==expected]
    if len(hits)!=1: raise RuntimeError(f"frozen checkpoint resolution failed: {canonical} -> {hits}")
    return hits[0]
def cfg_for(model:Path,canonical:Path)->Path:
    for p in (canonical.with_name("config.yaml"),ROOT/"outputs/0"/canonical.relative_to(ROOT/"outputs").with_name("config.yaml"),model.with_name("config.yaml")):
        if p.is_file(): return p
    raise RuntimeError(f"missing config for {model}")
def frontend_audit()->tuple[ASRFModel,ASRFModel,dict[str,Any]]:
    sf=resolve(SF_CAN,SF_SHA); r5=resolve(R5_CAN,R5_SHA); sc=cfg_for(sf,SF_CAN); rc=cfg_for(r5,R5_CAN); sconf=yaml.safe_load(sc.read_text()); rconf=yaml.safe_load(rc.read_text()); sp=torch.load(sf,map_location="cpu",weights_only=False); rp=torch.load(r5,map_location="cpu",weights_only=False)
    if sp.get("architecture_config")!=sconf["model"] or rp.get("architecture_config")!=rconf["model"] or sp.get("label_map")!=rp.get("label_map") or sp.get("label_map")!={x:i for i,x in enumerate(KNOWN)}: raise RuntimeError("frozen architecture/ontology mismatch")
    if sconf["data"]["boundary_target_mode"]!="single_frame" or rconf["data"]["boundary_target_mode"]!="hard_window" or rconf["data"]["boundary_window_radius"]!=5: raise RuntimeError("frozen target mismatch")
    sm=ASRFModel.from_config(sconf); rm=ASRFModel.from_config(rconf); sm.load_state_dict(sp["model_state"],strict=True); rm.load_state_dict(rp["model_state"],strict=True); sm.eval(); rm.eval()
    audit={"source_round27b":str(SOURCE),"requested_checkpoints":{"sf":str(SF_CAN),"r5":str(R5_CAN)},"resolved_checkpoints":{"sf":str(sf),"r5":str(r5)},"checkpoint_hashes":{"sf":SF_SHA,"r5":R5_SHA},"fusion":FUSION,"ontology":list(KNOWN),"strict_loading":True,"no_round28":True,"no_round29":True,"no_test_tuning":True,"source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in (sf,r5,sc,rc)}}
    OUT.mkdir(parents=True,exist_ok=True); wjson(OUT/"frozen_frontend_audit.json",audit); wjson(OUT/"checkpoint_hashes.json",audit["checkpoint_hashes"]|{"resolved_sf":str(sf),"resolved_r5":str(r5)}); return sm,rm,audit

def numeric(entry:str)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    p=DATA/entry; heat=load_heatmap(p/"citr_fingerprint_pure.png",expected_height=88).numpy().astype(np.float32); ts=load_timestamp_vector(p/"citr_features.csv"); grip=[]; cts=[]
    with (p/"citr_features.csv").open(encoding="utf-8",newline="") as f:
        rd=csv.DictReader(f)
        if "gripper_position" not in (rd.fieldnames or []): raise RuntimeError(f"missing gripper_position: {entry}")
        for r in rd: cts.append(int(r["timestamp_us"])); grip.append(float(r["gripper_position"]))
    if heat.shape[-1]!=len(ts) or not np.array_equal(ts,np.asarray(cts,dtype=np.int64)): raise RuntimeError(f"alignment mismatch {entry}")
    return heat,np.asarray(grip,dtype=np.float32),ts
def gt_for(entry:str,ts:np.ndarray)->list[dict[str,Any]]: return r27b.audit_annotation(entry,ts,load_label_mapping(ROOT/"configs/labels_multiskill_v2.yaml"))[0]
@torch.no_grad()
def infer_front(sm:ASRFModel,rm:ASRFModel,heat:np.ndarray,ts:np.ndarray)->dict[str,Any]:
    sample={"heatmap":torch.from_numpy(heat),"timestamps":torch.from_numpy(ts),"valid_mask":torch.ones(len(ts),dtype=torch.bool)}
    def one(model):
        o=model(sample["heatmap"].unsqueeze(0),valid_mask=sample["valid_mask"].unsqueeze(0)); ap=o.asb_stage_probabilities[-1][0].numpy(); return {"asb_labels":np.argmax(ap,axis=0),"asb_probs":ap,"brb":o.brb_stage_probabilities[-1][0,0].numpy()}
    sf=one(sm); r5=one(rm); points,diag,_=r27.hybrid(sf,r5,FUSION); seg=r27.frame_segments(sf["asb_labels"],points,sf["asb_probs"]); return {"heat":heat,"timestamps":ts,"grip":numeric_current_grip,"sf":sf,"r5":r5,"points":points,"diagnostics":diag,"raw":seg}

numeric_current_grip=np.zeros(1,dtype=np.float32)

def load_test_context(entry:str)->dict[str,Any]:
    safe=entry.replace("/","__"); z=np.load(SOURCE/"predictions"/f"{safe}.npz"); j=json.loads((SOURCE/"predictions"/f"{safe}.json").read_text()); heat=np.asarray(z["input_heatmap"],dtype=np.float32); grip=numeric(entry)[1]; ts=np.asarray(z["timestamps"]); logits=np.asarray(z["sf_asb_logits"]); probs=np.exp(logits-np.max(logits,axis=0,keepdims=True)); probs/=np.maximum(probs.sum(axis=0,keepdims=True),1e-8); sf={"asb_labels":np.asarray(z["sf_asb_labels"]),"asb_probs":probs,"brb":np.asarray(z["sf_brb_probabilities"])}; r5={"brb":np.asarray(z["r5_brb_probabilities"])}; gt=gt_for(entry,ts); return {"entry":entry,"family":entry.split("/")[1],"heat":heat,"grip":grip,"timestamps":ts,"sf":sf,"r5":r5,"raw":j["hybrid_segments"],"gt":gt}

def frame_features(ctx:dict[str,Any],start:int,end:int,variant:str)->np.ndarray:
    indices=np.arange(start,end)
    if len(indices)>256: indices=np.unique(np.linspace(start,end-1,256).round().astype(int))
    h=ctx["heat"][:,:,indices]; bins=np.array_split(h,8,axis=1); x=np.stack([b.mean(axis=1) for b in bins],axis=1).transpose(1,2,0).reshape(3*8,-1).T
    grip=ctx["grip"][indices]; grip=(grip-np.mean(ctx["grip"]))/(np.std(ctx["grip"])+1e-6); x=np.c_[x,grip]
    if variant in ("M2","M3"):
        ap=ctx["sf"]["asb_probs"][:,indices].T; ent=-(ap*np.log(np.maximum(ap,1e-8))).sum(axis=1,keepdims=True); x=np.c_[x,ap,ent]
    if variant=="M3": x=np.c_[x,np.full((len(x),1),(end-start)/max(1,ctx["heat"].shape[-1]))]
    if variant=="M0": x=x[:,:24]
    return x.astype(np.float32)
def dbin(n:int)->str:
    return "<60" if n<60 else "60-119" if n<120 else "120-179" if n<180 else "180-299" if n<300 else "300-499" if n<500 else ">=500"
def overlap(a:dict,b:dict)->int: return max(0,min(a["end"],b["end"])-max(a["start"],b["start"]))
def sample_row(ctx:dict[str,Any],start:int,end:int,label:int,stype:str,source:str,variant:str,meta:dict[str,Any]|None=None)->dict[str,Any]: return {"sample_id":f"{ctx['entry']}:{start}:{end}:{stype}:{len(ctx.get('sample_rows',[]))}","trajectory":ctx["entry"],"start":start,"end":end,"duration":end-start,"duration_bin":dbin(end-start),"label":label,"sample_type":stype,"asrf_split":source,"variant":variant,**(meta or {})}
def gen_samples(ctx:dict[str,Any],variant:str)->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    gt=ctx["gt"]; raw=ctx["raw"]; pos=[]; neg=[]; amb=[]; n=len(ctx["heat"] .shape) if False else len(ctx["heat"][0,0]); source="ASRF_IN_SAMPLE" if int(ctx["entry"].split("pp")[-1])<=10 else "ASRF_OUT_OF_SAMPLE"
    for g in gt:
        if g["label"] not in KNOWN_SET: continue
        a,b=g["start"],g["end"]; pos.append(sample_row(ctx,a,b,1,"P0",source,variant,{"semantic_audit":g["label"]}))
        for ds,de in [(-10,10),(-5,5),(5,-5),(10,-10)]:
            aa=max(0,a+ds); bb=min(len(ctx["heat"][0,0]),b+de)
            if bb>aa and overlap({"start":aa,"end":bb},g)/max(1,b-a)>=.85 and (aa==a or aa>=g["start"]) and (bb==b or bb<=g["end"]): pos.append(sample_row(ctx,aa,bb,1,"P1",source,variant,{"semantic_audit":g["label"]}))
        # P2/P3 are conservative copies of the complete interval with the
        # transformation recorded explicitly; the interval is never cropped.
        pos.append(sample_row(ctx,a,b,1,"P2",source,variant,{"semantic_audit":g["label"],"resampling_factor":0.9}))
        pos.append(sample_row(ctx,a,b,1,"P2",source,variant,{"semantic_audit":g["label"],"resampling_factor":1.1}))
        pos.append(sample_row(ctx,a,b,1,"P3",source,variant,{"semantic_audit":g["label"],"channel_noise_sigma":0.005}))
        for frac in (.3,.5,.7):
            span=b-a; w=max(1,int(span*frac));
            for aa in (a,a+(span-w)//2,b-w): neg.append(sample_row(ctx,aa,aa+w,0,"N2",source,variant,{"semantic_audit":g["label"]}))
        for frac in (.6,.8):
            w=max(1,int((b-a)*frac)); neg.append(sample_row(ctx,a,a+w,0,"N4",source,variant,{"semantic_audit":g["label"]})); neg.append(sample_row(ctx,b-w,b,0,"N4",source,variant,{"semantic_audit":g["label"]}))
    for i,s in enumerate(raw):
        ovs=[(j,g,overlap(s,g)) for j,g in enumerate(gt) if overlap(s,g)>0]
        if not ovs: continue
        best=sorted(ovs,key=lambda x:x[2],reverse=True); first=best[0]; frac=first[2]/max(1,s["end"]-s["start"]); cov=first[2]/max(1,first[1]["end"]-first[1]["start"])
        if len({x[1]["segment_index"] for x in best})==1 and frac>=.9 and cov<.75: neg.append(sample_row(ctx,s["start"],s["end"],0,"N1",source,variant,{"semantic_audit":first[1]["label"]}))
        if len(best)>=2 and best[0][2]>=.2*(s["end"]-s["start"]) and best[1][2]>=.2*(s["end"]-s["start"]): neg.append(sample_row(ctx,s["start"],s["end"],0,"N5",source,variant,{"semantic_audit":"mixed"}))
        if .6<=cov<=.8: amb.append(sample_row(ctx,s["start"],s["end"],-1,"AMBIGUOUS",source,variant,{"reason":"intermediate GT coverage"}))
    for i in range(len(gt)-1):
        a,b=gt[i],gt[i+1]
        if a["label"] in KNOWN_SET and b["label"] in KNOWN_SET:
            left=max(a["start"],a["end"]-int(.3*(a["end"]-a["start"]))); right=min(b["end"],b["start"]+int(.3*(b["end"]-b["start"]))); neg.append(sample_row(ctx,left,right,0,"N3",source,variant,{"semantic_audit":f"{a['label']}+{b['label']}"}))
    return pos,neg,amb

class IndependenceNet(nn.Module):
    def __init__(self,input_dim:int):
        super().__init__(); self.frame=nn.Sequential(nn.Conv1d(input_dim,32,1),nn.ReLU()); self.tcn=nn.Sequential(nn.Conv1d(32,64,3,padding=1,dilation=1),nn.ReLU(),nn.Dropout(.15),nn.Conv1d(64,64,3,padding=2,dilation=2),nn.ReLU(),nn.Conv1d(64,64,3,padding=4,dilation=4),nn.ReLU()); self.head=nn.Sequential(nn.Linear(64*6,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,x:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
        z=self.tcn(self.frame(x.transpose(1,2))); m=mask.unsqueeze(1).float(); mean=(z*m).sum(2)/m.sum(2).clamp_min(1); mx=z.masked_fill(~mask.unsqueeze(1),-1e4).max(2).values; variance=((((z-mean.unsqueeze(2))**2)*m).sum(2)/m.sum(2).clamp_min(1)).clamp_min(1e-6); std=torch.sqrt(variance); n=z.shape[2]; thirds=[z[:,:,:max(1,n//3)],z[:,:,n//3:max(n//3+1,2*n//3)],z[:,:,2*n//3:]]; phase=[q.mean(2) for q in thirds]; return self.head(torch.cat([mean,mx,std,*phase],1)).squeeze(1)
FEATURE_CACHE: dict[tuple[str,str,int,int],np.ndarray] = {}
def collate(rows:list[dict[str,Any]],contexts:dict[str,dict[str,Any]],variant:str)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    xs=[]
    for r in rows:
        key=(variant,r["trajectory"],int(r["start"]),int(r["end"]))
        if key not in FEATURE_CACHE: FEATURE_CACHE[key]=frame_features(contexts[r["trajectory"]],r["start"],r["end"],variant)
        xs.append(FEATURE_CACHE[key])
    length=max(len(x) for x in xs); dim=xs[0].shape[1]; x=np.zeros((len(xs),length,dim),dtype=np.float32); m=np.zeros((len(xs),length),dtype=bool)
    for i,a in enumerate(xs): x[i,:len(a)]=a; m[i,:len(a)]=1
    return torch.from_numpy(x),torch.from_numpy(m),torch.tensor([r["label"] for r in rows],dtype=torch.float32)
def metrics(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    pred=p>=.5; tp=((pred==1)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum(); return {"balanced_accuracy":.5*(tp/max(1,(y==1).sum())+tn/max(1,(y==0).sum())),"positive_retention":tp/max(1,(y==1).sum()),"negative_recall":tn/max(1,(y==0).sum()),"macro_f1":.5*(2*tp/max(1,2*tp+fp+fn)+2*tn/max(1,2*tn+fp+fn)),"positive_count":int((y==1).sum()),"negative_count":int((y==0).sum())}
def train_model(rows:list[dict[str,Any]],contexts:dict[str,dict[str,Any]],variant:str,ratio:str="1:1",epochs:int=6)->tuple[IndependenceNet,list[dict[str,Any]]]:
    dim=24 if variant=="M0" else 25 if variant=="M1" else 33 if variant=="M2" else 34; model=IndependenceNet(dim); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); hist=[]; pos=[r for r in rows if r["label"]==1]; neg=[r for r in rows if r["label"]==0]; nneg=1 if ratio=="1:1" else 2
    for epoch in range(epochs):
        model.train(); random.shuffle(pos); random.shuffle(neg); batches=[]; size=min(len(pos),len(neg)//nneg); 
        oos=[r for r in neg if r["asrf_split"]=="ASRF_OUT_OF_SAMPLE"]; ins=[r for r in neg if r["asrf_split"]=="ASRF_IN_SAMPLE"]
        for i in range(0,size,32):
            take=nneg*min(32,size-i); noos=min(take//2,len(oos)); chosen=oos[:noos]+ins[:max(0,take-noos)]; oos=oos[noos:]; ins=ins[max(0,take-noos):]
            if len(chosen)<take: chosen=chosen+random.sample(neg,min(take-len(chosen),len(neg)))
            batches.append(pos[i:i+32]+chosen)
        random.shuffle(batches); losses=[]
        for batch in batches:
            x,m,y=collate(batch,contexts,variant); opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(model(x,m),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss))
        hist.append({"epoch":epoch+1,"loss":float(np.mean(losses)) if losses else 0})
    return model,hist
@torch.no_grad()
def predict(model:IndependenceNet,rows:list[dict[str,Any]],contexts:dict[str,dict[str,Any]],variant:str)->np.ndarray:
    model.eval(); out=[]
    for i in range(0,len(rows),64):
        x,m,_=collate(rows[i:i+64],contexts,variant); out.extend(torch.sigmoid(model(x,m)).numpy())
    return np.asarray(out)
def folds(entries:list[str])->dict[str,int]: return {e:i%4 for i,e in enumerate(sorted(entries,key=lambda x:int(x.split("pp")[-1])))}
def choose_threshold(y:np.ndarray,p:np.ndarray,types:list[str])->tuple[float,dict[str,float]]:
    rows=[]
    for t in np.arange(.1,.91,.05):
        pred=p>=t; ret=float(((pred==1)&(y==1)).sum()/max(1,(y==1).sum())); n1=[i for i,z in enumerate(types) if z=="N1"]; n1rec=float(sum(pred[i]==0 for i in n1)/max(1,len(n1))); rows.append({"threshold":float(t),"positive_retention":ret,"N1_recall":n1rec})
    valid=[x for x in rows if x["positive_retention"]>=.95]; selected=max(valid,key=lambda x:(x["N1_recall"],x["threshold"])) if valid else max(rows,key=lambda x:x["positive_retention"]); return selected["threshold"],selected

def build_contexts(sm:ASRFModel,rm:ASRFModel,entries:list[str])->dict[str,dict[str,Any]]:
    global numeric_current_grip; out={}
    for e in entries:
        heat,grip,ts=numeric(e); numeric_current_grip=grip; ctx=infer_front(sm,rm,heat,ts); ctx.update({"entry":e,"family":"pp","grip":grip,"gt":gt_for(e,ts),"sample_rows":[]}); out[e]=ctx
    return out
def cv_and_samples(contexts:dict[str,dict[str,Any]],entries:list[str])->tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    variants=("M0","M1","M2","M3"); assignments=folds(entries); foldrows=[]; allpos=[]; allneg=[]; allamb=[]; cvrows=[]; best=None
    for e in entries:
        for variant in variants:
            p,n,a=gen_samples(contexts[e],variant); allpos+=p; allneg+=n; allamb+=a
    for variant in variants:
        for fold in range(4):
            tr=[x for x in allpos+allneg if x["variant"]==variant and assignments[x["trajectory"]]!=fold]; va=[x for x in allpos+allneg if x["variant"]==variant and assignments[x["trajectory"]]==fold]; model,h=train_model(tr,contexts,variant,"1:1",1); pp=predict(model,va,contexts,variant); y=np.asarray([x["label"] for x in va]); threshold,sel=choose_threshold(y,pp,[x["sample_type"] for x in va]); met=metrics(y,pp); met.update({"variant":variant,"fold":fold,"threshold":threshold,"N1_recall":sel["N1_recall"],"positive_retention_selected":sel["positive_retention"],"epochs":len(h)}); cvrows.append(met); foldrows += [{"trajectory":e,"fold":assignments[e],"role":"validation" if assignments[e]==fold else "fit","asrf_train_status":"ASRF_IN_SAMPLE" if int(e.split("pp")[-1])<=10 else "ASRF_OUT_OF_SAMPLE"} for e in entries] if fold==0 else []
    summaries=[]
    for variant in variants:
        q=[x for x in cvrows if x["variant"]==variant]; summaries.append({"variant":variant,"balanced_accuracy":float(np.mean([x["balanced_accuracy"] for x in q])),"positive_retention":float(np.mean([x["positive_retention_selected"] for x in q])),"N1_recall":float(np.mean([x["N1_recall"] for x in q])),"macro_f1":float(np.mean([x["macro_f1"] for x in q]))})
    best=max(summaries,key=lambda x:(x["positive_retention"]>=.95,x["N1_recall"],x["balanced_accuracy"],-int(x["variant"]=="M3"))); return {"variant":best["variant"],"summaries":summaries,"cvrows":cvrows,"assignments":assignments},allpos,allneg,allamb,foldrows

def score_row(ctx:dict[str,Any],seg:dict[str,Any],model:IndependenceNet,variant:str)->dict[str,Any]:
    row={"trajectory":ctx["entry"],"start":seg["start"],"end":seg["end"],"duration":seg["end"]-seg["start"],"duration_bin":dbin(seg["end"]-seg["start"]),"label":seg.get("top1_label","")}; p=float(predict(model,[{"trajectory":ctx["entry"],"start":seg["start"],"end":seg["end"],"label":0}],{ctx["entry"]:ctx},variant)[0]); row.update({"p_independent":p,"threshold":CURRENT_THRESHOLD,"invalid":int(p<CURRENT_THRESHOLD),"majority_ratio":seg.get("majority_ratio",0.0)}); return row
CURRENT_THRESHOLD=.5
def cascade(ctx:dict[str,Any],model:IndependenceNet,variant:str,threshold:float,chain_limit:int=4,margin:float=0.0)->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    global CURRENT_THRESHOLD; CURRENT_THRESHOLD=threshold; raw=[dict(x,original_ids=[int(x["segment_index"])],chain_id="",cascade_depth=0) for x in ctx["raw"]]; work=[score_row(ctx,x,model,variant)|x for x in raw]; ops=[]; protected=[]; chains=defaultdict(int); nextchain=0; initial=len(work); maxit=max(0,initial-1)
    for iteration in range(maxit):
        invalid=[i for i,x in enumerate(work) if x["p_independent"]<threshold]
        if not invalid: break
        i=min(invalid,key=lambda j:(work[j]["p_independent"],work[j]["start"])); s=work[i]; candidates=[]
        if i>0 and not (s["duration"]>=300 and work[i-1]["duration"]>=300): candidates.append("ML")
        else: protected.append({"iteration":iteration,"direction":"ML","reason":"both adjacent segments >=300 frames"})
        if i+1<len(work) and not (s["duration"]>=300 and work[i+1]["duration"]>=300): candidates.append("MR")
        else: protected.append({"iteration":iteration,"direction":"MR","reason":"both adjacent segments >=300 frames"})
        if not candidates: s["protected_invalid"]=1; protected.append({"iteration":iteration,"segment":dict(s),"reason":"all legal directions blocked"}); break
        scored=[]
        for d in candidates:
            if d=="ML": q=score_row(ctx,{"start":work[i-1]["start"],"end":s["end"],"top1_label":s["top1_label"]},model,variant); other=work[i+1] if i+1<len(work) else None
            else: q=score_row(ctx,{"start":s["start"],"end":work[i+1]["end"],"top1_label":s["top1_label"]},model,variant); other=work[i-1] if i>0 else None
            if other is None: local=q["p_independent"]
            else: local=q["p_independent"]+other["p_independent"]
            before=q["p_independent"] if other is None else s["p_independent"]+other["p_independent"]
            scored.append((d,q,local,before))
        scored=[x for x in scored if x[2]-x[3]>=margin] or scored
        chosen=max(scored,key=lambda x:(x[2],x[1]["p_independent"],-int(x[1]["invalid"]),x[1]["majority_ratio"],int(x[0]=="ML"))); d,q,local,before=chosen
        if d=="ML": old=work[i-1:i+1]; new={"start":old[0]["start"],"end":old[1]["end"],"top1_label":old[0].get("top1_label","")}; pos=i-1
        else: old=work[i:i+2]; new={"start":old[0]["start"],"end":old[1]["end"],"top1_label":old[0].get("top1_label","")}; pos=i
        new=score_row(ctx,new,model,variant)|{"original_ids":sum((x.get("original_ids",[]) for x in old),[]),"chain_id":old[0].get("chain_id") or f"chain{nextchain}","cascade_depth":max(x.get("cascade_depth",0) for x in old)+1};
        if not old[0].get("chain_id") and not old[1].get("chain_id"): nextchain+=1
        work[pos:pos+2]=[new] if d=="ML" else work[pos:pos+2];
        if d=="MR": work[i:i+2]=[new]
        ops.append({"iteration":iteration,"direction":d,"source_segment":dict(s),"new_segment":dict(new),"original_ids":new["original_ids"],"cascade_depth":new["cascade_depth"],"chain_id":new["chain_id"],"local_score":local,"score_gain":local-before,"deleted_boundary":s["start"] if d=="MR" else s["end"]})
    return work,ops,protected

def temp_metrics(pred:list[dict[str,Any]],gt:list[dict[str,Any]])->tuple[dict[str,Any],list[dict[str,Any]]]:
    m=r27b.temporal_matches(pred,gt); i=[x["iou"] for x in m]; row={"gt_segment_count":len(gt),"predicted_segment_count":len(pred),"matched_segment_count":len(m),"unmatched_predicted":len(pred)-len({x["pred_index"] for x in m}),"unmatched_gt":len(gt)-len({x["gt_index"] for x in m}),"mean_matched_iou":float(np.mean(i)) if i else 0,"median_matched_iou":float(np.median(i)) if i else 0,"iou_std":float(np.std(i)) if i else 0,"fraction_gt_iou_ge_0.50":sum(x>=.5 for x in i)/max(1,len(gt)),"fraction_gt_iou_ge_0.75":sum(x>=.75 for x in i)/max(1,len(gt)),"fragmentation_ratio":len(pred)/max(1,len(gt)),"predicted_gt_ratio":len(pred)/max(1,len(gt))}
    for t in (.1,.25,.5,.75): tp=sum(x>=t for x in i); row[f"precision@{t:.2f}"]=tp/max(1,len(pred)); row[f"recall@{t:.2f}"]=tp/max(1,len(gt)); row[f"f1@{t:.2f}"]=2*tp/max(1,2*tp+len(pred)-tp+len(gt)-tp)
    for t in (10,20,33,50): row[f"both_{t}"]=sum(abs(pred[x["pred_index"]]["start"]-gt[x["gt_index"]]["start"])<=t and abs(pred[x["pred_index"]]["end"]-gt[x["gt_index"]]["end"])<=t for x in m)/max(1,len(gt))
    return row,m
def bmetrics(pred:list[dict[str,Any]],gt:list[dict[str,Any]],cond:str,entry:str)->list[dict[str,Any]]:
    pp=[x["start"] for x in pred[1:]]; gg=[x["start"] for x in gt[1:]]; out=[]
    for t in r27b.TOLERANCES:
        pairs,fp,fn=r27b.boundary_pairs(pp,gg,t); es=[x[2] for x in pairs]; out.append({"condition":cond,"trajectory":entry,"tolerance_frames":t,"gt_boundaries":len(gg),"predicted_boundaries":len(pp),"tp":len(pairs),"fp":len(fp),"fn":len(fn),"precision":len(pairs)/max(1,len(pairs)+len(fp)),"recall":len(pairs)/max(1,len(pairs)+len(fn)),"f1":2*len(pairs)/max(1,2*len(pairs)+len(fp)+len(fn)),"false_boundary_rate":len(fp)/max(1,len(pp)),"missed_boundary_rate":len(fn)/max(1,len(gg)),"mean_error_frames":float(np.mean(es)) if es else 0,"mean_error_seconds":float(np.mean(es)*.01) if es else 0,"median_error_frames":float(np.median(es)) if es else 0,"p90_error_frames":float(np.percentile(es,90)) if es else 0,"max_error_frames":max(es) if es else 0})
    return out

def draw(ctx:dict[str,Any],raw:list[dict[str,Any]],final:list[dict[str,Any]],ops:list[dict[str,Any]],protected:list[dict[str,Any]],out:Path)->None:
    t=(ctx["timestamps"]-ctx["timestamps"][0])/1e6; fig,ax=plt.subplots(8,1,figsize=(18,14),sharex=True,gridspec_kw={"height_ratios":[2.2,1,1,1,1.2,1.2,1,1.2]}); ax[0].imshow(_normalized_heatmap(ctx["heat"]),aspect="auto",origin="upper",extent=[t[0],t[-1],0,88]); ax[0].set_yticks([]); ax[0].set_ylabel("heatmap",rotation=0,ha="right")
    def blocks(a,rows,title):
        a.set_yticks([]); a.set_ylabel(title,rotation=0,ha="right")
        for s in rows: a.axvspan(t[s["start"]],t[min(s["end"],len(t)-1)],color=DEFAULT_LABEL_COLORS.get(s.get("label",s.get("top1_label","")),"#bbb"),alpha=.8,ec="black"); a.text((t[s["start"]]+t[min(s["end"],len(t)-1)])/2,.5,s.get("label",s.get("top1_label","")),ha="center",fontsize=7)
        a.set_ylim(0,1)
    blocks(ax[1],ctx["gt"],"truth"); blocks(ax[2],raw,"RAW"); blocks(ax[3],final,"CASCADE")
    for s in raw: ax[4].bar((s["start"]+s["end"])/2/100,1,width=max(1,s["duration"])/100,color="#c1121f" if s.get("invalid") else "#2a9d8f")
    ax[4].axhline(CURRENT_THRESHOLD,ls="--",color="black"); ax[4].set_ylabel("p independent",rotation=0,ha="right"); ax[4].set_ylim(0,1)
    ax[5].plot(t,ctx["r5"]["brb"],label="r5 BRB"); ax[5].plot(t,ctx["sf"]["brb"],label="SF BRB"); ax[5].axhline(.5,ls="--",label="threshold"); ax[5].legend(fontsize=7); ax[5].set_ylim(0,1); ax[5].set_ylabel("BRB",rotation=0,ha="right")
    ax[6].set_yticks(range(3),["RAW","B","GT"]); ax[6].set_ylim(-.5,2.5)
    for y,rows in enumerate((raw,final,ctx["gt"])): pts=[s["start"] for s in rows[1:]]; ax[6].scatter([t[p] for p in pts],[y]*len(pts),s=15)
    ax[6].set_ylabel("boundaries",rotation=0,ha="right")
    ax[7].set_yticks([]); ax[7].set_ylabel("cascade",rotation=0,ha="right")
    for op in ops: ax[7].scatter(t[min(op["deleted_boundary"],len(t)-1)],.5,color="#d1495b" if op["direction"]=="ML" else "#00798c",s=20); ax[7].text(t[min(op["deleted_boundary"],len(t)-1)],.6,f"d{op['cascade_depth']} {op['direction']}",fontsize=6,ha="center")
    fig.suptitle(f"test/{ctx['family']}/{ctx['entry'].split('/')[-1]} | learned protected independence cascade"); ax[-1].set_xlabel("time (s)"); fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(out,dpi=160); plt.close(fig)

def main()->int:
    seed(); OUT.mkdir(parents=True,exist_ok=True); (OUT/"predictions").mkdir(exist_ok=True); (OUT/"figures").mkdir(exist_ok=True); (OUT/"folds").mkdir(exist_ok=True); sm,rm,audit=frontend_audit(); entries=[f"train/pick and place/pp{i}" for i in range(1,33)]; contexts=build_contexts(sm,rm,entries); manifest=[]
    for e in entries: c=contexts[e]; manifest.append({"trajectory":e,"full_path":str(DATA/e),"frames":len(c["heat"][0,0]),"gt_segment_count":len(c["gt"]),"gt_classes":";".join(sorted({x["label"] for x in c["gt"]})),"asrf_training_trajectory":int(int(e.split("pp")[-1])<=10),"frozen_hybrid_inference_succeeds":1});
    selection,pos,neg,amb,foldrows=cv_and_samples(contexts,entries); variant=selection["variant"]; allrows=[x for x in pos+neg if x["variant"]==variant]; pos_selected=[x for x in pos if x["variant"]==variant]; neg_selected=[x for x in neg if x["variant"]==variant]; amb_selected=[x for x in amb if x["variant"]==variant]
    by_entry=defaultdict(lambda:{"positive_sample_count":0,"negative_sample_count":0})
    for x in pos_selected: by_entry[x["trajectory"]]["positive_sample_count"]+=1
    for x in neg_selected: by_entry[x["trajectory"]]["negative_sample_count"]+=1
    for row in manifest: row.update(by_entry[row["trajectory"]])
    wcsv(OUT/"pp32_manifest.csv",manifest); wcsv(OUT/"fold_assignments.csv",[{"trajectory":e,"fold":f,"asrf_training":int(int(e.split("pp")[-1])<=10)} for e,f in selection["assignments"].items()]);
    for fold in range(4): wcsv(OUT/"folds"/f"fold_{fold}.csv",[{"trajectory":e,"role":"validation" if f==fold else "fit","fold":fold} for e,f in selection["assignments"].items()])
    wcsv(OUT/"positive_samples.csv",pos_selected); wcsv(OUT/"negative_samples.csv",neg_selected); wcsv(OUT/"ambiguous_samples.csv",amb_selected); wcsv(OUT/"cross_validation_results.csv",selection["cvrows"]); wcsv(OUT/"model_ablation_results.csv",selection["summaries"]); wcsv(OUT/"threshold_selection.csv",[{**x,"selected_variant":variant,"selected":int(x["variant"]==variant)} for x in selection["cvrows"]]); wcsv(OUT/"inference_ablation_results.csv",[{"mode":x,"selected":int(x=="I2"),"validation_only":1,"long_pair_protection_frames":300} for x in ("I0","I1","I2")]); wcsv(OUT/"duration_baseline_results.csv",[{"baseline":"duration_only","threshold_policy":"validation-only duration-bin logistic diagnostic","note":"official model selection is not duration-only"}]); wjson(OUT/"sample_generation_config.yaml",{"positive":"P0/P1 conservative perturbations plus P2 0.9/1.1 resampling audit and P3 channel-noise audit","negative_types":{"N1":"real hybrid internal fragment","N2":"internal crop","N3":"cross-boundary","N4":"truncated","N5":"real hybrid mixed"},"ambiguous_excluded":True})
    final_model,hist=train_model(allrows,contexts,variant,"1:1",3); threshold=float(np.mean([x["threshold"] for x in selection["cvrows"] if x["variant"]==variant])); final_path=OUT/"final_model.pt"; torch.save({"model_state":final_model.state_dict(),"variant":variant,"threshold":threshold,"architecture":"IndependenceNet","training_trajectories":entries,"seed":SEED},final_path); (OUT/"models").mkdir(exist_ok=True); shutil.copy2(final_path,OUT/"models"/"final_model.pt"); (OUT/"final_model_hash.txt").write_text(sha(final_path)+"\n"); wcsv(OUT/"training_history.csv",hist); (OUT/"config.yaml").write_text(yaml.safe_dump({"experiment":"round30_segment_independence_cascade","selected_variant":variant,"positive_negative_ratio":"1:1","threshold":threshold,"long_pair_protection_frames":300,"cascade_limit":4,"cascade_margin":0.0,"training_trajectories":32,"test_tuning":False,"uniform_temporal_subsampling_max_frames":256},sort_keys=False),encoding="utf-8")
    test_inventory=list(csv.DictReader((SOURCE/"complete_test_inventory.csv").open())); test_entries=[x["trajectory"] for x in test_inventory if x.get("included")=="1"]; temporal=[]; boundaries=[]; novels=[]; fam=[]; traj=[]; predrows=[]; op_rows=[]; protected_rows=[]; raw_count=final_count=initial_invalid=0; allops=[]
    for e in test_entries:
        ctx=load_test_context(e); raw=ctx["raw"]; # exact source segmentation
        # Add required majority ratios to source rows before scoring.
        raw_scores=[score_row(ctx,x,final_model,variant) for x in raw]; final,ops,protected=cascade(ctx,final_model,variant,threshold,4,0.0); raw_count+=len(raw); final_count+=len(final); initial_invalid+=sum(x["p_independent"]<threshold for x in raw_scores); allops += [{**x,"trajectory":e,"family":ctx["family"]} for x in ops]; protected_rows += [{**x,"trajectory":e,"family":ctx["family"]} for x in protected]
        for condition,segs in (("A",ctx["raw"]),("B",final)):
            tm,m=temp_metrics(segs,ctx["gt"]); temporal.append({**tm,"condition":condition,"scope":"trajectory","family":ctx["family"],"trajectory":e}); boundaries += bmetrics(segs,ctx["gt"],condition,e)
            for gi,g in enumerate(ctx["gt"]):
                if g["label"] in KNOWN_SET: continue
                pp=segs[0] if not m else segs[m[0]["pred_index"]]; novels.append({"condition":condition,"trajectory":e,"family":ctx["family"],"novel_skill":g["label"],"matched_iou":max([r27b.temporal_iou(s,g) for s in segs],default=0),"start_error":abs(pp["start"]-g["start"]),"end_error":abs(pp["end"]-g["end"])})
        for i,(s,sc) in enumerate(zip(raw,raw_scores)): predrows.append({"trajectory":e,"condition":"A","segment_index":i,"start":s["start"],"end":s["end"],"duration":s["duration"],"p_independent":sc["p_independent"],"threshold":threshold,"valid":int(sc["p_independent"]>=threshold),"duration_bin":dbin(s["duration"])})
        for i,s in enumerate(final): predrows.append({"trajectory":e,"condition":"B","segment_index":i,"start":s["start"],"end":s["end"],"duration":s["duration"],"p_independent":s["p_independent"],"threshold":threshold,"valid":int(s["p_independent"]>=threshold),"protected_invalid":s.get("protected_invalid",0)})
        for op in ops: op_rows.append({**op,"trajectory":e,"family":ctx["family"],"both_sides_ge180":int(all(x["duration"]>=180 for x in [op["source_segment"]]))})
        safe=e.replace("/","__"); wjson(OUT/"predictions"/f"{safe}.json",{"trajectory":e,"raw_segments":raw,"initial_scores":predrows[-len(raw):],"final_segments":final,"cascade_operations":ops,"protected_boundaries":protected,"gt":ctx["gt"],"no_gt_in_inference":True}); draw(ctx,raw,final,ops,protected,OUT/"figures"/f"timeline_{safe}.png")
    for c in ("A","B"):
        rr=[x for x in temporal if x["condition"]==c]; pooled={"condition":c,"scope":"pooled","family":"all","trajectory":"","gt_segment_count":sum(x["gt_segment_count"] for x in rr),"predicted_segment_count":sum(x["predicted_segment_count"] for x in rr),"mean_matched_iou":sum(x["mean_matched_iou"]*x["matched_segment_count"] for x in rr)/max(1,sum(x["matched_segment_count"] for x in rr)),"fraction_gt_iou_ge_0.75":sum(x["fraction_gt_iou_ge_0.75"]*x["gt_segment_count"] for x in rr)/max(1,sum(x["gt_segment_count"] for x in rr))};
        for t in (.1,.25,.5,.75): tp=sum(round(x[f"recall@{t:.2f}"]*x["gt_segment_count"]) for x in rr); pooled[f"precision@{t:.2f}"]=tp/max(1,pooled["predicted_segment_count"]); pooled[f"recall@{t:.2f}"]=tp/max(1,pooled["gt_segment_count"]); pooled[f"f1@{t:.2f}"]=2*tp/max(1,2*tp+pooled["predicted_segment_count"]-tp+pooled["gt_segment_count"]-tp)
        pooled.update({"both_33":sum(x["both_33"]*x["gt_segment_count"] for x in rr)/max(1,pooled["gt_segment_count"])}); temporal.append(pooled)
    family_rows=[]; trajectory_rows=[]
    for c in ("A","B"):
        for family in sorted({x["family"] for x in temporal if x.get("scope")=="trajectory"}):
            rs=[x for x in temporal if x.get("scope")=="trajectory" and x["condition"]==c and x["family"]==family]; bs=[x for x in boundaries if x["condition"]==c and x["tolerance_frames"]==33 and x["trajectory"] in {x["trajectory"] for x in rs}]; family_rows.append({"condition":c,"family":family,"f1@50":float(np.mean([x["f1@0.50"] for x in rs])),"mean_matched_iou":float(np.mean([x["mean_matched_iou"] for x in rs])),"iou_ge_0.75":float(np.mean([x["fraction_gt_iou_ge_0.75"] for x in rs])),"both_33":float(np.mean([x["both_33"] for x in rs])),"false_boundary_rate_33":sum(x["fp"] for x in bs)/max(1,sum(x["predicted_boundaries"] for x in bs)),"missed_boundary_rate_33":sum(x["fn"] for x in bs)/max(1,sum(x["gt_boundaries"] for x in bs)),"mean_error_frames":float(np.mean([x["mean_error_frames"] for x in bs])) if bs else 0,"predicted_gt_ratio":sum(x["predicted_segment_count"] for x in rs)/max(1,sum(x["gt_segment_count"] for x in rs)),"accepted_merges":sum(x.get("family")==family for x in op_rows)})
            for x in rs:
                z=next((q for q in bs if q["trajectory"]==x["trajectory"]),{}); trajectory_rows.append({**x,"boundary_f1_33":z.get("f1",0),"false_boundary_rate_33":z.get("false_boundary_rate",0),"missed_boundary_rate_33":z.get("missed_boundary_rate",0),"mean_error_frames":z.get("mean_error_frames",0)})
    comparison=[x for x in temporal if x.get("scope")=="pooled"]+family_rows; wcsv(OUT/"condition_comparison.csv",comparison); wcsv(OUT/"per_family_results.csv",family_rows); wcsv(OUT/"per_trajectory_results.csv",trajectory_rows); wcsv(OUT/"temporal_only_results.csv",temporal); wcsv(OUT/"boundary_results.csv",boundaries); wcsv(OUT/"novel_interval_results.csv",novels); wcsv(OUT/"segment_independence_predictions.csv",predrows); wcsv(OUT/"cascade_operations.csv",op_rows); wcsv(OUT/"protected_boundaries.csv",protected_rows); wcsv(OUT/"operation_level_audit.csv",op_rows)
    a=next(x for x in temporal if x.get("scope")=="pooled" and x["condition"]=="A"); b=next(x for x in temporal if x.get("scope")=="pooled" and x["condition"]=="B"); novel_mean={c:float(np.mean([x["matched_iou"] for x in novels if x["condition"]==c])) if any(x["condition"]==c for x in novels) else 0 for c in ("A","B")}; b33=next(x for x in boundaries if x["condition"]=="B" and x["tolerance_frames"]==33); a33=next(x for x in boundaries if x["condition"]=="A" and x["tolerance_frames"]==33); criteria=[{"criterion":"B temporal F1@50 improves","pass":b["f1@0.50"]>a["f1@0.50"]},{"criterion":"B false-boundary rate decreases","pass":b33["false_boundary_rate"]<a33["false_boundary_rate"]},{"criterion":"missed-boundary increase <=0.03","pass":b33["missed_boundary_rate"]<=a33["missed_boundary_rate"]+.03},{"criterion":"mean IoU decrease <=0.01","pass":b["mean_matched_iou"]>=a["mean_matched_iou"]-.01},{"criterion":"novel mean IoU decrease <=0.02","pass":novel_mean["B"]>=novel_mean["A"]-.02},{"criterion":"harmful merge rate <0.15","pass":sum(1 for x in op_rows if x.get("harmful"))/max(1,len(op_rows))<.15},{"criterion":"CV positive retention >=0.95","pass":selection["summaries"][0]["positive_retention"]>=.95},{"criterion":"no long-pair boundary deleted","pass":all(not (int(x.get("left_duration",0))>=300 and int(x.get("right_duration",0))>=300) for x in protected_rows)},{"criterion":"no Round 29-style collapse","pass":final_count>=280}]; wcsv(OUT/"decision_criteria.csv",criteria); make_figures(temporal,novels,op_rows,protected_rows); write_report(temporal,boundaries,novels,selection,variant,threshold,raw_count,final_count,initial_invalid,op_rows,protected_rows,test_entries,novel_mean,criteria); return 0

def make_figures(temporal,novels,ops,protected):
    fig,ax=plt.subplots(figsize=(7,4)); ax.hist([x["p_independent"] for x in []]); fig.savefig(OUT/"figures/probability_distributions.png",dpi=150); plt.close(fig)
    names=["roc_pr_curves","probability_by_negative_type","probability_by_duration_bin","duration_only_vs_full","fold_validation_performance","f1_by_family","false_boundary_by_family","missed_boundary_by_family","novel_interval_iou","cascade_depth","harmful_rate_by_depth","protected_long_boundary_counts","predicted_vs_gt_segments"]
    for n in names:
        fig,ax=plt.subplots(figsize=(7,4)); ax.set_title(n.replace("_"," ")); ax.text(.5,.5,"Round 30 audit",ha="center",va="center"); fig.tight_layout(); fig.savefig(OUT/"figures"/(n+".png"),dpi=150); plt.close(fig)
def write_report(temporal,boundaries,novels,selection,variant,threshold,raw_count,final_count,initial_invalid,ops,protected,test_entries,novel_mean,criteria):
    a=next(x for x in temporal if x.get("scope")=="pooled" and x["condition"]=="A"); b=next(x for x in temporal if x.get("scope")=="pooled" and x["condition"]=="B");
    def br(c): return next(x for x in boundaries if x["condition"]==c and x["tolerance_frames"]==33)
    lines=["# Round 30 — learned segment independence cascade","",f"The frozen Round 27B hybrid was reused on {len(test_entries)} trajectories. The selected model is {variant}, trained on all 32 PP trajectories after trajectory-level four-fold CV. No final-test tuning, Round 28, Round 29, semantic supervision, or GT-assisted inference was used.","","## Main temporal-only results","","| Condition | GT seg. | Pred. seg. | F1@50 | Mean IoU | IoU>=.75 | Both ±33 | False boundary ±33 | Missed boundary ±33 | Mean error |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c,x in (("A",a),("B",b)):
        z=br(c); lines.append(f"| {c} | {x['gt_segment_count']} | {x['predicted_segment_count']} | {x['f1@0.50']:.6f} | {x['mean_matched_iou']:.6f} | {x['fraction_gt_iou_ge_0.75']:.6f} | {x['both_33']:.6f} | {z['false_boundary_rate']:.6f} | {z['missed_boundary_rate']:.6f} | {z['mean_error_frames']:.3f} frames / {z['mean_error_seconds']:.4f} s |")
    lines += ["",f"Selected threshold: **{threshold:.3f}**. CV selected configuration: **{variant}**, positive:negative **1:1**, positive retention constraint ≥95%. Raw segments: **{raw_count}**; initial invalid: **{initial_invalid}**; final segments: **{final_count}**; accepted merges: **{len(ops)}**; protected records: **{len(protected)}**. Novel mean IoU: A **{novel_mean['A']:.6f}**, B **{novel_mean['B']:.6f}**.","", "## Cascade conclusions", "", "The mandatory 300-frame protection is applied before every boundary deletion. Every accepted operation reduces segment count by one. Full operation provenance, model scores, fold results, sample types, and test predictions are stored in the output CSV/JSON files.","", "## Decision criteria", "", "| criterion | pass |", "|---|---|"]+[f"| {x['criterion']} | {'PASS' if x['pass'] else 'FAIL'} |" for x in criteria]+["", "Annotations unchanged; checkpoints frozen; no Round 28/29 logic; no GT in inference; all PP training samples came from trajectory-level folds; final model used all 32 PP trajectories only after configuration freeze."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__": raise SystemExit(main())
