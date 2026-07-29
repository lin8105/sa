import sys

sys.path.insert(0, "scripts")
import run_round22_protected_asb_boundary_refinement as round22  # noqa: E402


def _features(**overrides):
    values = {
        "short_skill": 0,
        "critical_transition": 0,
        "high_brb": 0,
        "two_sided_stability": 0,
        "semantic_incompatibility": 0,
        "sequential_incompatibility": 0,
    }
    values.update(overrides)
    return values


def test_critical_transition_is_hard_protected():
    protected, reason, _ = round22.protection_decision(_features(critical_transition=1), "P2_transition", soft=False, cfg={})
    assert protected == 1
    assert "critical_transition" in reason


def test_high_brb_protection_is_variant_specific():
    protected, reason, _ = round22.protection_decision(_features(high_brb=1), "P3_brb", soft=False, cfg={})
    assert protected == 1
    assert reason == "high_brb"
    unprotected, _, _ = round22.protection_decision(_features(high_brb=1), "P1_short", soft=False, cfg={})
    assert unprotected == 0


def test_soft_protection_reports_penalty_without_deleting_boundary():
    protected, reason, penalty = round22.protection_decision(_features(short_skill=1, critical_transition=1), "P9_full_soft", soft=True, cfg={"soft_penalty": 0.45})
    assert protected == 0
    assert reason.startswith("soft_penalty:")
    assert penalty > 0


def test_risk_feature_names_are_fixed_and_interpretable():
    names = round22.risk_names()
    assert "brb_probability" in names
    assert "short_skill" in names
    assert "semantic_incompatibility" in names
    assert len(names) >= 10
