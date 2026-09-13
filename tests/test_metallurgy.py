"""
Unit tests for Stainless Steel Metallurgy Engine, Grade Profiles, and Disposition Logic.
"""
import pytest
from backend.core.grade_profiles import get_grade_profile, list_supported_grades, GRADE_PROFILES
from backend.core.metallurgy_engine import (
    analyze_defect,
    compute_frame_disposition,
    EXTENDED_DEFECT_TAXONOMY,
    BASE_SEVERITY_SCORES
)

def test_grade_profiles_retrieval():
    """Verify grade profile loading and fallback mechanism."""
    p304 = get_grade_profile("SS_304")
    assert "AISI 304" in p304["grade_name"]
    assert "scratches" in p304["base_tolerances"]
    assert p304["base_tolerances"]["scratches"]["critical"] is True

    p316 = get_grade_profile("SS_316L")
    assert "AISI 316L" in p316["grade_name"]
    # In 316L, pitted_surface has weight 2.2 and is critical
    assert p316["base_tolerances"]["pitted_surface"]["weight"] == 2.2
    assert p316["base_tolerances"]["pitted_surface"]["critical"] is True

    p_duplex = get_grade_profile("DUPLEX_2205")
    assert p_duplex["base_tolerances"]["crazing"]["weight"] == 2.2

    # Fallback to SS_304 for unknown code
    p_unknown = get_grade_profile("UNKNOWN_GRADE_999")
    assert "AISI 304" in p_unknown["grade_name"]

    supported = list_supported_grades()
    assert len(supported) >= 5
    assert "SS_304" in supported

def test_analyze_defect_center_vs_edge():
    """Verify edge proximity risk multiplier is applied to strip edges."""
    img_w, img_h = 1000, 1000
    
    # Center defect (x: 450 to 550)
    center_box = [450.0, 450.0, 550.0, 550.0]
    center_res = analyze_defect(
        defect_class="patches",
        confidence=0.80,
        box=center_box,
        image_width=img_w,
        image_height=img_h,
        grade_code="SS_304"
    )
    assert center_res["is_edge_defect"] is False
    assert center_res["strip_zone"] == "Strip Center / Body"

    # Edge defect (x: 10 to 60 -> within 12% margin)
    edge_box = [10.0, 450.0, 60.0, 550.0]
    edge_res = analyze_defect(
        defect_class="patches",
        confidence=0.80,
        box=edge_box,
        image_width=img_w,
        image_height=img_h,
        grade_code="SS_304"
    )
    assert edge_res["is_edge_defect"] is True
    assert edge_res["strip_zone"] == "Strip Edge"
    # Edge defect severity should be strictly higher due to 1.35x multiplier
    assert edge_res["severity_score"] > center_res["severity_score"]

def test_analyze_defect_grade_sensitivity():
    """Verify SS 316L penalizes pitting much more severely than standard commercial grade SS 201."""
    box = [200.0, 200.0, 300.0, 300.0]
    res_316 = analyze_defect(
        defect_class="pitted_surface",
        confidence=0.85,
        box=box,
        image_width=1000,
        image_height=1000,
        grade_code="SS_316L"
    )
    res_201 = analyze_defect(
        defect_class="pitted_surface",
        confidence=0.85,
        box=box,
        image_width=1000,
        image_height=1000,
        grade_code="SS_201"
    )
    assert res_316["severity_score"] > res_201["severity_score"]
    assert res_316["is_critical_for_grade"] is True
    assert res_201["is_critical_for_grade"] is False

def test_compute_frame_disposition_prime():
    """Defect-free strip must receive PRIME / PASS disposition with A+ quality."""
    disp = compute_frame_disposition([])
    assert disp["disposition"] == "PRIME / PASS"
    assert disp["disposition_code"] == "PRIME"
    assert disp["quality_grade"] == "A+"
    assert disp["quality_score_pct"] == 100.0
    assert disp["total_defect_count"] == 0

def test_compute_frame_disposition_rework():
    """Superficial reworkable defects (like light roll patches) should route to REWORK."""
    light_patches = [{
        "defect_class": "patches",
        "severity_score": 2.2,
        "is_critical_for_grade": False,
        "reworkable": True
    }]
    disp = compute_frame_disposition(light_patches, grade_code="SS_304")
    assert disp["disposition_code"] == "REWORK"
    assert disp["quality_grade"] == "B"

def test_compute_frame_disposition_scrap():
    """Severe critical defects (crazing with high severity) must trigger SCRAP / REJECT."""
    severe_crazing = [{
        "defect_class": "crazing",
        "severity_score": 8.8,
        "is_critical_for_grade": True,
        "reworkable": False
    }]
    disp = compute_frame_disposition(severe_crazing, grade_code="SS_304")
    assert disp["disposition_code"] == "SCRAP"
    assert disp["quality_grade"] == "F"
    assert disp["critical_defect_count"] == 1

def test_extended_taxonomy_completeness():
    """Verify taxonomy contains all 25 industrial stainless steel defects with required fields."""
    assert len(EXTENDED_DEFECT_TAXONOMY) == 25, f"Expected 25 taxonomy categories, found {len(EXTENDED_DEFECT_TAXONOMY)}"
    for key, item in EXTENDED_DEFECT_TAXONOMY.items():
        assert "full_name" in item
        assert "category" in item
        assert "root_cause" in item
        assert "metallurgical_hazard" in item
        assert "corrective_actions" in item
        assert len(item["corrective_actions"]) > 0

def test_compute_frame_disposition_false_alarm_neutralized():
    """Verify that candidate defects flagged as false alarms do NOT cause coil scrap."""
    false_alarm_defect = [{
        "defect_class": "crazing",
        "severity_score": 0.0,
        "is_critical_for_grade": False,
        "is_potential_false_alarm": True,
        "severity_tier": "Suppressed / False Alarm",
        "reworkable": False
    }]
    disp = compute_frame_disposition(false_alarm_defect, grade_code="SS_304")
    assert disp["disposition"] == "PRIME / PASS"
    assert disp["disposition_code"] == "PRIME"
    assert disp["quality_grade"] == "A+"
    assert disp["total_defect_count"] == 0
    assert "false alarm" in disp["action_summary"].lower()
