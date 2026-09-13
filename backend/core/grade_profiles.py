"""
Stainless Steel Grade Tolerance Profiles for Jindal Stainless manufacturing line.
Defines grade-specific defect sensitivity, critical defect types, and metallurgical tolerances.
"""
from typing import Dict, Any

GRADE_PROFILES: Dict[str, Dict[str, Any]] = {
    "SS_304": {
        "grade_name": "AISI 304 / EN 1.4301 (Austenitic Standard)",
        "typical_applications": "Architectural trim, food processing equipment, consumer appliances, commercial kitchens",
        "surface_finish": "2B (Cold rolled, annealed, pickled, skin-passed) / BA (Bright Annealed)",
        "base_tolerances": {
            "crazing": {"weight": 1.2, "max_acceptable_area_pct": 0.5, "critical": True},
            "inclusion": {"weight": 1.3, "max_acceptable_area_pct": 0.4, "critical": True},
            "patches": {"weight": 0.8, "max_acceptable_area_pct": 2.0, "critical": False},
            "pitted_surface": {"weight": 1.2, "max_acceptable_area_pct": 0.8, "critical": False},
            "rolled-in_scale": {"weight": 1.4, "max_acceptable_area_pct": 0.5, "critical": True},
            "scratches": {"weight": 1.5, "max_acceptable_area_pct": 0.3, "critical": True},  # High aesthetic penalty
        },
        "metallurgy_note": "Aesthetic sensitivity is paramount for SS 304 in architectural and consumer use. Scratches and roll marks impair reflective uniformity and surface passivity."
    },
    "SS_316L": {
        "grade_name": "AISI 316L / EN 1.4404 (Marine & Chemical Grade)",
        "typical_applications": "Offshore platforms, pharmaceutical vessels, chemical reactors, marine hardware",
        "surface_finish": "2B / 2D Industrial Passivated",
        "base_tolerances": {
            "crazing": {"weight": 1.5, "max_acceptable_area_pct": 0.3, "critical": True},
            "inclusion": {"weight": 2.0, "max_acceptable_area_pct": 0.1, "critical": True},  # Inclusions initiate pitting
            "patches": {"weight": 0.9, "max_acceptable_area_pct": 1.5, "critical": False},
            "pitted_surface": {"weight": 2.2, "max_acceptable_area_pct": 0.05, "critical": True}, # Zero-tolerance for pitting
            "rolled-in_scale": {"weight": 1.6, "max_acceptable_area_pct": 0.3, "critical": True},
            "scratches": {"weight": 1.0, "max_acceptable_area_pct": 0.8, "critical": False},
        },
        "metallurgy_note": "Extreme sensitivity to pitting and non-metallic inclusions. Molybdenum-enriched passive layer is vulnerable to localized chloride pit nucleation at inclusion boundaries (PREN >= 24)."
    },
    "SS_430": {
        "grade_name": "AISI 430 / EN 1.4016 (Ferritic Stainless)",
        "typical_applications": "Automotive exhaust trim, dishwasher liners, interior panels, chimney flues",
        "surface_finish": "BA / 2B Ferritic",
        "base_tolerances": {
            "crazing": {"weight": 1.6, "max_acceptable_area_pct": 0.2, "critical": True},  # Crazing triggers roping failure
            "inclusion": {"weight": 1.1, "max_acceptable_area_pct": 0.6, "critical": False},
            "patches": {"weight": 1.0, "max_acceptable_area_pct": 1.5, "critical": False},
            "pitted_surface": {"weight": 1.1, "max_acceptable_area_pct": 0.8, "critical": False},
            "rolled-in_scale": {"weight": 1.3, "max_acceptable_area_pct": 0.6, "critical": False},
            "scratches": {"weight": 1.1, "max_acceptable_area_pct": 0.7, "critical": False},
        },
        "metallurgy_note": "Ferritic lattice has lower room-temperature ductility than austenitic grades. Surface micro-fissures (crazing) severely aggravate plastic anisotropy and forming splits."
    },
    "SS_201": {
        "grade_name": "AISI 201 / Low-Ni High-Mn (Commercial Austenitic)",
        "typical_applications": "Cookware, railway carriage interiors, structural tubing, decorative hardware",
        "surface_finish": "2B / No. 4 Polished",
        "base_tolerances": {
            "crazing": {"weight": 1.0, "max_acceptable_area_pct": 0.8, "critical": True},
            "inclusion": {"weight": 1.0, "max_acceptable_area_pct": 0.8, "critical": False},
            "patches": {"weight": 0.7, "max_acceptable_area_pct": 3.0, "critical": False},
            "pitted_surface": {"weight": 1.0, "max_acceptable_area_pct": 1.2, "critical": False},
            "rolled-in_scale": {"weight": 1.1, "max_acceptable_area_pct": 1.0, "critical": False},
            "scratches": {"weight": 1.0, "max_acceptable_area_pct": 1.0, "critical": False},
        },
        "metallurgy_note": "Standard commercial tolerance profile. Cost-optimized grade; minor superficial scratches and color patches are tolerable for non-critical forming applications."
    },
    "DUPLEX_2205": {
        "grade_name": "Duplex 2205 / EN 1.4462 (Austenitic-Ferritic Duplex)",
        "typical_applications": "Oil & gas subsea flowlines, chemical tankers, desalination high-pressure pumps",
        "surface_finish": "Hot rolled annealed pickled (1D / No. 1) or 2B",
        "base_tolerances": {
            "crazing": {"weight": 2.2, "max_acceptable_area_pct": 0.05, "critical": True},  # Stress corrosion cracking risk
            "inclusion": {"weight": 1.9, "max_acceptable_area_pct": 0.1, "critical": True},
            "patches": {"weight": 1.0, "max_acceptable_area_pct": 1.0, "critical": False},
            "pitted_surface": {"weight": 2.0, "max_acceptable_area_pct": 0.1, "critical": True},
            "rolled-in_scale": {"weight": 1.8, "max_acceptable_area_pct": 0.2, "critical": True},
            "scratches": {"weight": 1.2, "max_acceptable_area_pct": 0.5, "critical": False},
        },
        "metallurgy_note": "High-yield strength structural grade. Crazing and surface notch defects cause premature fatigue nucleation and hydrogen-induced stress corrosion cracking in sour gas services."
    }
}

def get_grade_profile(grade_code: str) -> Dict[str, Any]:
    """Retrieve tolerance profile for a given stainless steel grade code."""
    normalized = grade_code.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in GRADE_PROFILES:
        return GRADE_PROFILES[normalized]
    # Default to SS 304 if unrecognized
    return GRADE_PROFILES["SS_304"]

def list_supported_grades() -> Dict[str, str]:
    """List all available steel grades and descriptions."""
    return {k: v["grade_name"] for k, v in GRADE_PROFILES.items()}
