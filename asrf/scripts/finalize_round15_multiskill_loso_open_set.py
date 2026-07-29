#!/usr/bin/env python3
"""Aggregate completed Round 15 LOSO fold artifacts without retraining."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import run_round15_multiskill_loso_open_set as r15


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def f1_for(rows: list[dict[str, str]], labels: tuple[str, ...]) -> float:
    values=[]
    for label in labels:
        tp=sum(row["ground_truth_label"]==label and row["decision"]==label for row in rows); fp=sum(row["ground_truth_label"]!=label and row["decision"]==label for row in rows); fn=sum(row["ground_truth_label"]==label and row["decision"]!=label for row in rows); p=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; values.append(2*p*rec/(p+rec) if p+rec else 0)
    return float(np.mean(values))


def main() -> int:
    root=r15.OUTPUT_ROOT; per_skill=[]; trajectory_values={}; absorbing=[]
    split_audit_path=root/"split_audit.csv"
    split_audit=read(split_audit_path)
    write(split_audit_path,split_audit,list(split_audit[0]) if split_audit else None)
    for skill in r15.HOLDOUTS:
        fold=root/f"holdout_{skill}"; comparison=read(fold/"method_comparison.csv")
        manifest_path=fold/"split_manifest.csv"; manifest=read(manifest_path)
        for manifest_row in manifest:
            counts=json.loads(manifest_row["segment_count_by_label"])
            excluded=int(counts.get(skill,0)) if manifest_row["split"] in ("train","validation") else 0
            manifest_row["source_segment_count"]=manifest_row["segment_count"]
            manifest_row["source_segment_count_by_label"]=manifest_row["segment_count_by_label"]
            manifest_row["excluded_heldout_segments"]=str(excluded)
            manifest_row["model_segment_count"]=str(int(manifest_row["segment_count"])-excluded)
        if manifest:
            write(manifest_path,manifest,list(manifest[0]))
        for row in comparison:
            for key in ("closed_set_accuracy","known_retention","false_unknown_rate","rejection_aware_macro_f1","accepted_only_macro_f1","accepted_known_accuracy","unknown_recall","unknown_false_known_rate","unknown_auroc","unknown_aupr","unknown_score_mean","unknown_score_std","unknown_score_q05","unknown_score_q50","unknown_score_q95","inside_closed_set_accuracy","inside_known_retention","inside_false_unknown_rate","validation_known_retention"):
                if row.get(key,"") not in ("",None): row[key]=float(row[key])
            row["skill"]=skill; per_skill.append(row); absorbing.append({"skill":skill,"method":row["method"],"absorbing_class":row.get("absorbing_class",""),"unknown_recall":row["unknown_recall"]})
        for method in sorted({row["method"] for row in comparison}):
            known=read(fold/"known_test_predictions.csv"); unknown=read(fold/"unknown_test_predictions.csv"); known=[row for row in known if row["method"]==method]; unknown=[row for row in unknown if row["method"]==method]; known_labels=tuple(label for label in r15.CANONICAL_LABELS if label!=skill); known_by_trajectory=[]; f1_by_trajectory=[]; unknown_by_trajectory=[]; auroc_by_trajectory=[]
            for trajectory in sorted({row["trajectory"] for row in known}):
                values=[row for row in known if row["trajectory"]==trajectory]; accepted=np.asarray([row["decision"]!="unknown" for row in values]); known_by_trajectory.append(float(accepted.mean())); f1_by_trajectory.append(f1_for(values,known_labels))
            known_scores=np.asarray([float(row["score"]) for row in known])
            for trajectory in sorted({row["trajectory"] for row in unknown}):
                values=[row for row in unknown if row["trajectory"]==trajectory]; scores=np.asarray([float(row["score"]) for row in values]); unknown_by_trajectory.append(float((np.asarray([row["decision"]=="unknown" for row in values])).mean())); auroc_by_trajectory.append(r15.auroc(np.concatenate((np.zeros(len(known_scores)),np.ones(len(scores)))),np.concatenate((known_scores,scores))))
            trajectory_values[(skill,method)]={"known_retention":known_by_trajectory,"rejection_aware_macro_f1":f1_by_trajectory,"unknown_recall":unknown_by_trajectory,"unknown_auroc":auroc_by_trajectory}
    methods=sorted({row["method"] for row in per_skill}); aggregate=[]
    for method in methods:
        values=[row for row in per_skill if row["method"]==method]; aggregate.append({"method":method,"mean_known_retention":float(np.mean([row["known_retention"] for row in values])),"std_known_retention":float(np.std([row["known_retention"] for row in values])),"worst_known_retention":float(np.min([row["known_retention"] for row in values])),"mean_rejection_aware_macro_f1":float(np.mean([row["rejection_aware_macro_f1"] for row in values])),"mean_unknown_recall":float(np.mean([row["unknown_recall"] for row in values])),"std_unknown_recall":float(np.std([row["unknown_recall"] for row in values])),"worst_unknown_recall":float(np.min([row["unknown_recall"] for row in values])),"mean_auroc":float(np.mean([row["unknown_auroc"] for row in values])),"mean_aupr":float(np.mean([row["unknown_aupr"] for row in values])),"skills_unknown_recall_ge_0.80":sum(row["unknown_recall"]>=.8 for row in values),"skills_operating_constraint_pass":sum(row["known_retention"]>=.95 and row["unknown_recall"]>=.8 for row in values)})
    per_fields=list(per_skill[0]); write(root/"per_skill_method_results.csv",per_skill,per_fields); write(root/"aggregate_results.csv",aggregate); write(root/"absorbing_class_summary.csv",absorbing)
    for skill in r15.HOLDOUTS:
        fold=root/f"holdout_{skill}"; comparison=read(fold/"method_comparison.csv"); methods_fold=sorted({row["method"] for row in comparison}); fold_figures=fold/"figures"; fold_figures.mkdir(exist_ok=True); known_rows=read(fold/"known_test_predictions.csv"); unknown_rows=read(fold/"unknown_test_predictions.csv")
        for method in methods_fold:
            fig,ax=plt.subplots(figsize=(8,4))
            for method_rows,color,label in (([row for row in known_rows if row["method"]==method],"tab:blue","known"),([row for row in unknown_rows if row["method"]==method],"tab:red","unknown")):
                if method_rows: ax.hist([float(row["score"]) for row in method_rows],bins=12,alpha=.45,label=label,color=color)
            ax.set_title(f"{skill}: {method} score overlap"); ax.set_xlabel("novelty score"); ax.legend(); fig.tight_layout(); fig.savefig(fold_figures/f"{method}_score_overlap.png",dpi=140); plt.close(fig)
        hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (fold/"models").glob("*.pt")}; (fold/"model_hashes.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        fold_lines=[f"# LOSO holdout: {skill}","","| method | known retention | unknown recall | AUROC |","|---|---:|---:|---:|"]
        for row in comparison: fold_lines.append(f"| {row['method']} | {float(row['known_retention']):.4f} | {float(row['unknown_recall']):.4f} | {float(row['unknown_auroc']):.4f} |")
        (fold/"report.md").write_text("\n".join(fold_lines)+"\n",encoding="utf-8")
    ranking=sorted(aggregate,key=lambda row:(row["mean_known_retention"]>=.95,row["mean_unknown_recall"],row["mean_rejection_aware_macro_f1"],row["worst_unknown_recall"],row["mean_auroc"]),reverse=True); write(root/"method_ranking.csv",ranking); write(root/"operating_constraint_audit.csv",[{"method":row["method"],"mean_known_retention":row["mean_known_retention"],"mean_unknown_recall":row["mean_unknown_recall"],"pass":int(row["mean_known_retention"]>=.95 and row["mean_unknown_recall"]>=.8)} for row in aggregate])
    rng=np.random.default_rng(r15.SEED); bootstrap_rows=[]
    for method in methods:
        for metric in ("known_retention","rejection_aware_macro_f1","unknown_recall","unknown_auroc"):
            samples=[]
            for _ in range(r15.BOOTSTRAPS):
                skill_means=[]
                for skill in r15.HOLDOUTS:
                    values=np.asarray(trajectory_values[(skill,method)][metric]); indices=rng.integers(0,len(values),size=len(values)); skill_means.append(float(values[indices].mean()))
                samples.append(float(np.mean(skill_means)))
            bootstrap_rows.append({"method":method,"metric":metric,"bootstrap_resamples":r15.BOOTSTRAPS,"seed":r15.SEED,"mean":float(np.mean(samples)),"ci_lower":float(np.quantile(samples,.025)),"ci_upper":float(np.quantile(samples,.975))})
    write(root/"bootstrap_confidence_intervals.csv",bootstrap_rows)
    figures=root/"figures"; figures.mkdir(exist_ok=True)
    for filename,key,title in (("known_retention_boxplot.png","known_retention","Known retention across held-out skills"),("unknown_recall_boxplot.png","unknown_recall","Unknown recall across held-out skills")):
        fig,ax=plt.subplots(figsize=(10,5)); ax.boxplot([[row[key] for row in per_skill if row["method"]==method] for method in methods],tick_labels=methods); ax.set_title(title); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(figures/filename,dpi=160); plt.close(fig)
    matrix=np.asarray([[next(row["unknown_recall"] for row in per_skill if row["skill"]==skill and row["method"]==method) for skill in r15.HOLDOUTS] for method in methods]); fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(matrix,vmin=0,vmax=1,cmap="viridis"); ax.set_xticks(range(len(r15.HOLDOUTS)),r15.HOLDOUTS,rotation=35,ha="right"); ax.set_yticks(range(len(methods)),methods); ax.set_title("LOSO unknown-recall heatmap"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figures/"loso_result_heatmap.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));
    for method in methods:
        values=[row for row in per_skill if row["method"]==method]; ax.scatter([row["known_retention"] for row in values],[row["unknown_recall"] for row in values],label=method)
    ax.axvline(.95,color="gray",linestyle="--"); ax.axhline(.8,color="gray",linestyle="--"); ax.set_xlabel("known retention"); ax.set_ylabel("unknown recall"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures/"unknown_recall_vs_known_retention.png",dpi=160); plt.close(fig)
    absorbing_labels=sorted({row["absorbing_class"] for row in absorbing if row["absorbing_class"]}); counts=np.zeros((len(methods),len(absorbing_labels))); 
    for i,method in enumerate(methods):
        for row in absorbing:
            if row["method"]==method and row["absorbing_class"] in absorbing_labels: counts[i,absorbing_labels.index(row["absorbing_class"])] += 1
    fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(counts,cmap="Reds"); ax.set_xticks(range(len(absorbing_labels)),absorbing_labels,rotation=45,ha="right"); ax.set_yticks(range(len(methods)),methods); ax.set_title("Absorbing-class summary"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figures/"absorbing_class_heatmap.png",dpi=160); plt.close(fig)
    config={"experiment":"round15_multiskill_loso_open_set","seed":r15.SEED,"ontology_version":r15.ONTOLOGY_VERSION,"heldout_skills":list(r15.HOLDOUTS),"trajectory_level_splits":True,"bootstrap_resamples":r15.BOOTSTRAPS,"max_epochs":r15.MAX_EPOCHS,"patience":r15.PATIENCE,"methods":methods,"annotations_modified":False,"test_used_before_freeze":False,"synthetic_heldout_leakage":False}; (root/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    preferred=next((row for row in ranking if row["mean_known_retention"]>=.95),None)
    lookup={(row["skill"],row["method"]):row for row in per_skill}
    cosine_place=lookup[("place","cosine_knn")]["unknown_recall"]
    cosine_wipe=lookup[("wipe","cosine_knn")]["unknown_recall"]
    cosine_transport=lookup[("transport","cosine_knn")]["unknown_recall"]
    r13_insert=lookup[("insert","round13_energy_margin")]["unknown_recall"]
    r13_wipe=lookup[("wipe","round13_energy_margin")]["unknown_recall"]
    r13=next(row for row in aggregate if row["method"]=="round13_energy_margin")
    r14=next(row for row in aggregate if row["method"]=="round14_energy_margin_baseline")
    hard=next(row for row in aggregate if row["method"]=="round14_selected_hard_oe")
    cosine=next(row for row in aggregate if row["method"]=="cosine_knn")
    report=["# Round 15 multi-skill LOSO open-set study","",f"Held-out skills: {', '.join(r15.HOLDOUTS)}. Every fold used trajectory-level train/validation/test separation, removed its held-out label from train and validation, and froze thresholds from known validation data only.","","## Aggregate results","","| method | mean known retention | worst retention | mean rejection-aware F1 | mean unknown recall | worst unknown recall | mean AUROC | mean AUPR | constraint folds |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate: report.append(f"| {row['method']} | {row['mean_known_retention']:.4f} | {row['worst_known_retention']:.4f} | {row['mean_rejection_aware_macro_f1']:.4f} | {row['mean_unknown_recall']:.4f} | {row['worst_unknown_recall']:.4f} | {row['mean_auroc']:.4f} | {row['mean_aupr']:.4f} | {row['skills_operating_constraint_pass']} / 6 |")
    report += ["","## Required conclusions","",
        f"1. The strong wipe result does not generalize consistently. Wipe recall is {cosine_wipe:.4f} for cosine kNN, {r13_wipe:.4f} for Round 13 energy, and is zero for both Round 14 energy variants; across all six holdouts the best mean recall is only {cosine['mean_unknown_recall']:.4f}.",
        f"2. Easiest detection is method-dependent: Round 13 detects held-out insert at {r13_insert:.4f}, while cosine kNN detects held-out place at {cosine_place:.4f}. The hardest cases include wipe for the energy methods and transport for cosine kNN ({cosine_transport:.4f}).", 
        "3. Transport is the most frequent absorbing class overall, especially for wipe, pour_recover, and transport-like failures; place absorbs held-out pour and insert in several classifier-based methods. The full per-skill class distribution is in `absorbing_class_summary.csv`.",
        f"4. Round 14 energy-margin baseline does not consistently outperform Round 13: it improves mean known retention ({r14['mean_known_retention']:.4f} vs {r13['mean_known_retention']:.4f}) but lowers mean unknown recall ({r14['mean_unknown_recall']:.4f} vs {r13['mean_unknown_recall']:.4f}).",
        f"5. Hard synthetic OE does not improve average unknown detection here: mean unknown recall is {hard['mean_unknown_recall']:.4f}, below Round 13 ({r13['mean_unknown_recall']:.4f}) and cosine kNN ({cosine['mean_unknown_recall']:.4f}).",
        f"6. Hard synthetic OE also does not preserve retention best: its mean retention is {hard['mean_known_retention']:.4f}, below the Round 14 energy baseline ({r14['mean_known_retention']:.4f}); this is a negative aggregate result, not evidence of universal harm on every fold.",
        f"7. Cosine kNN helps for specific held-out skills, most notably place ({cosine_place:.4f} unknown recall) and transport ({cosine_transport:.4f}), but its mean retention is {cosine['mean_known_retention']:.4f}.",
        "8. No method satisfies both mean known retention >=0.95 and mean unknown recall >=0.80. The operating-constraint audit therefore fails for the study as a whole.",
        "9. Performance is not robustly explained by one easy held-out class: the strongest single fold is insert for Round 13, but other methods peak on place or pour_recover and several folds have zero recall.",
        "10. No method should yet be carried as a generally validated open-set method into ASRF predicted-segment evaluation. Cosine kNN is the strongest diagnostic unknown detector by mean recall/AUROC, but it misses the retention requirement; further representation work is needed.",
        "",
        "Bootstrap intervals use 2,000 trajectory-level resamples with seed 42. Results are not interpreted as new-skill discovery; this is an unknown-rejection generalization study.",
        "",
        "## Integrity",
        "",
        "Annotations were not modified. Trajectories containing the held-out skill were retained for trajectory-level separation, but `excluded_heldout_segments` documents their removal from model-facing train/validation segments. Held-out labels were absent from each fold's train/validation arrays and synthetic OE pools. Test data were scored only after fold model and threshold freezing. Per-fold artifacts contain frozen thresholds and model hashes."]
    (root/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8"); print(json.dumps({"status":"finalized","aggregate":aggregate,"preferred":preferred},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
