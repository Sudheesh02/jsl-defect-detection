"""
Metallurgy & Quality Assurance Engine for Stainless Steel Defect Analysis.
Computes defect severity, root-cause attribution, strip location risk,
and automated coil disposition (PRIME / REWORK / DOWNGRADE / SCRAP).
"""
from typing import Dict, List, Any, Optional
from backend.core.grade_profiles import get_grade_profile
from backend.config import DISPOSITION_THRESHOLDS

# Comprehensive 25-Defect Industrial Metallurgy Taxonomy (NEU-DET + SteelDefectX + ASTM/EN Stainless Standards)
EXTENDED_DEFECT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "crazing": {
        "full_name": "Crazing / Network Micro-Cracks",
        "category": "Thermal / Tensile Stress",
        "root_cause": "Uneven continuous casting secondary cooling, excessive thermal gradients, or severe cold reduction tensile stress.",
        "metallurgical_hazard": "Initiates brittle fatigue fracture, catastrophic strip tearing at high line tensions, and stress corrosion cracking.",
        "corrective_actions": [
            "Verify mold powder lubricating viscosity and secondary spray cooling nozzle alignment.",
            "Reduce cold-rolling line speed by 15% and inspect intermediate annealing furnace temperature curve.",
            "Audit casting mold copper tube wear profile."
        ],
        "reworkable": False
    },
    "inclusion": {
        "full_name": "Non-Metallic Smelting Inclusion",
        "category": "Steelmaking / Refining",
        "root_cause": "Tundish slag entrainment, submerged entry nozzle (SEN) clogging, or deoxidation alumina/silicate particles remaining in liquid melt.",
        "metallurgical_hazard": "Breaks passive chromium-oxide barrier; acts as primary nucleation center for localized pitting corrosion and micro-void coalescence.",
        "corrective_actions": [
            "Check ladle-to-tundish argon shroud integrity to prevent re-oxidation.",
            "Verify calcium-treatment ladle desulfurization and argon bubbling rinsing duration.",
            "Inspect tundish stopper rod position and SEN refractory wear."
        ],
        "reworkable": False
    },
    "patches": {
        "full_name": "Surface Patches / Roll Slippage Marks",
        "category": "Rolling Mill Mechanical",
        "root_cause": "Localized friction variation, work roll slippage during reduction, or incomplete localized pickling acid contact.",
        "metallurgical_hazard": "Aesthetic defect, non-uniform surface roughness (Ra) affecting subsequent coating, buffing, or stamping.",
        "corrective_actions": [
            "Inspect rolling mill emulsion concentration, temperature, and lubrication oil flow rate.",
            "Check roll bite entry tension and work roll surface roughness.",
            "Verify acid pickling line tank temperature and turbulence agitator operation."
        ],
        "reworkable": True
    },
    "pitted_surface": {
        "full_name": "Pitted Surface / Acid Attack",
        "category": "Chemical Pickling / Corrosion",
        "root_cause": "Over-pickling in mixed acid bath (HF/HNO3), chloride contamination in rinse cascade, or localized stagnant droplet corrosion.",
        "metallurgical_hazard": "Compromises wall thickness tolerance, acts as stress concentration notch, causes galvanic micro-cells.",
        "corrective_actions": [
            "Check pickling line free acid titration levels and strip immersion time.",
            "Inspect high-pressure rinse water conductivity (< 50 uS/cm) and squeegee roller pressure.",
            "Verify passivating bath oxidizing potential (Redox potential > +600 mV)."
        ],
        "reworkable": False
    },
    "rolled-in_scale": {
        "full_name": "Rolled-in Iron Oxide Scale",
        "category": "Hot Strip Mill / Descaling",
        "root_cause": "Incomplete primary or secondary descaling spray impact pressure (< 180 bar) allowing hot iron oxides (wustite/magnetite) to roll into the steel substrate.",
        "metallurgical_hazard": "Imbeds brittle oxides beneath strip surface, causing flaking during forming, seam weld porosity, and galvanic corrosion.",
        "corrective_actions": [
            "Inspect high-pressure descaling header nozzle wear, pressure drops, and filter clogging.",
            "Verify reheating furnace soaking zone atmosphere (control excess O2 < 1.5%).",
            "Schedule immediate grinding or roll change for affected mill stand."
        ],
        "reworkable": True
    },
    "scratches": {
        "full_name": "Mechanical Guide / Abrasion Scratches",
        "category": "Line Handling / Guide Mechanics",
        "root_cause": "Foreign metal chips or scale lodged in bridle rolls, deflector rolls, loop tower guides, or uncoiler guide knives.",
        "metallurgical_hazard": "Linear notch defect causing tear propagation during deep drawing; visual rejection for mirror/architectural finishes.",
        "corrective_actions": [
            "Clean and inspect line guide rolls, bridle rolls, and pinch rollers for embedded metal slivers.",
            "Verify strip pass line elevation and side guide roll clearances.",
            "Check pay-off reel back tension to eliminate coil clock-spring abrasion."
        ],
        "reworkable": True
    },
    "roll_printing": {
        "full_name": "Finishing Roll Printing / Roll Marks",
        "category": "Work Roll Damage",
        "root_cause": "Local work roll spalling, work roll dent from tail-end slap, or debris indentation on work rolls replicating at roll circumference pitch.",
        "metallurgical_hazard": "Periodic defect running full coil length; causes non-uniform strip gauge and aesthetic failure.",
        "corrective_actions": [
            "Measure repeating defect pitch to identify exact roll stand diameter.",
            "Trigger automated work roll changeover on identified rolling stand.",
            "Inspect entry strip cleaners for particulate contamination."
        ],
        "reworkable": True
    },
    "edge_cracks": {
        "full_name": "Edge Cracks / Trim Tearing",
        "category": "Edge Finishing / Trimming",
        "root_cause": "Excessive edge cooling in roughing mill, dull side trimmer knives, or excessive lateral spread deformation.",
        "metallurgical_hazard": "High risk of full strip breakage across tension loopers, damaging mill rolls and causing unplanned line downtime.",
        "corrective_actions": [
            "Check side trimmer knife sharpness, horizontal overlap, and vertical clearance.",
            "Verify edge heater induction power at roughing breakdown mill.",
            "Slow line speed through tension bridle to prevent catastrophic coil rupture."
        ],
        "reworkable": False
    },
    "crease": {
        "full_name": "Transverse Coil Crease / Coil Breaks",
        "category": "Strip Tension / Uncoiler",
        "root_cause": "Yield point elongation instability during uncoiling without adequate back-tension or improper anti-fluting roll engagement.",
        "metallurgical_hazard": "Visible transverse ridges across strip width; cannot be eliminated by standard skin-pass rolling.",
        "corrective_actions": [
            "Engage anti-fluting breaker roll on uncoiler mandrel.",
            "Optimize payoff reel back-tension to maintain uniform strip catenary.",
            "Inspect coil temperature prior to uncoiling (allow coil to reach room temperature)."
        ],
        "reworkable": False
    },
    "crescent_gap": {
        "full_name": "Crescent Gap / Shear Blade Notch",
        "category": "Shearing / Entry End",
        "root_cause": "Damaged flying shear knife blade or misaligned crop shear leaving semi-circular edge tears during coil head/tail cropping.",
        "metallurgical_hazard": "Severe notch stress concentration leading to instant edge tearing in cold rolling stands.",
        "corrective_actions": [
            "Index or replace notched entry crop shear blades.",
            "Recalibrate crop shear blade overlap and timing sequence.",
            "Trim strip past crescent notch before threading mill reduction stands."
        ],
        "reworkable": False
    },
    "waist_folding": {
        "full_name": "Longitudinal Waist Folding / Buckle",
        "category": "Tension Leveler / Plastic Strain",
        "root_cause": "Excessive inter-stand tension causing transverse Poisson contraction beyond elastic yield limit, inducing longitudinal folding ridges.",
        "metallurgical_hazard": "Irreversible localized thickness reduction and longitudinal ridge marks.",
        "corrective_actions": [
            "Reduce inter-stand looper strip tension profile by 10-15%.",
            "Check roll crown compensation and intermediate roll shifting alignment.",
            "Recalibrate tension leveler intermesh penetration depth."
        ],
        "reworkable": False
    },
    "oil_spot": {
        "full_name": "Carbonized Oil Spot / Residue",
        "category": "Annealing / Emulsion",
        "root_cause": "Rolling emulsion carryover not fully removed by squeegee rolls before bright annealing furnace, pyrolyzing into carbon spots.",
        "metallurgical_hazard": "Surface carbon contamination impairs corrosion resistance and local passivation barrier.",
        "corrective_actions": [
            "Inspect exit air knives and squeegee roller durometer wear.",
            "Audit degreasing line alkaline spray pressure and rinse water temp (> 65 C).",
            "Adjust bright annealing furnace protective hydrogen/nitrogen gas flow rate."
        ],
        "reworkable": True
    },
    "water_stain": {
        "full_name": "Rinse Water Stain / Mineral Spotting",
        "category": "Pickling / Post-Rinse",
        "root_cause": "High mineral TDS in final rinse water or inefficient dryer blower leaving drying droplets on strip surface.",
        "metallurgical_hazard": "Aesthetic blemish and localized ionic residue that may initiate atmospheric tarnishing in storage.",
        "corrective_actions": [
            "Check reverse osmosis (RO) demineralizer water conductivity (< 20 uS/cm).",
            "Inspect hot air dryer blower nozzles for balanced air knife velocity.",
            "Ensure squeegee roll nip pressure is uniform across entire strip width."
        ],
        "reworkable": True
    },
    "blister": {
        "full_name": "Subsurface Gas Blister / Hydrogen Pocket",
        "category": "Casting / Hydrogen Degassing",
        "root_cause": "Hydrogen supersaturation or subsurface gas porosity trapped during continuous casting expanding during hot rolling.",
        "metallurgical_hazard": "Loss of laminar structural cohesion; pops open into surface voids during cold rolling or deep drawing.",
        "corrective_actions": [
            "Verify vacuum degassing (VD/VOD) duration and hydrogen content (< 2.0 ppm in melt).",
            "Audit slab soaking furnace residence time to allow hydrogen effusing.",
            "Reject affected strip segment for critical pressure vessel applications."
        ],
        "reworkable": False
    },
    "seam_weld_defect": {
        "full_name": "Seam Weld Anomaly / Lack of Fusion",
        "category": "Strip Joining / Welder",
        "root_cause": "Inadequate weld current, laser beam misalignment, or edge gap mismatch at coil-joining welder.",
        "metallurgical_hazard": "Strip separation in continuous annealing-pickling furnace requiring catastrophic line re-threading.",
        "corrective_actions": [
            "Inspect laser weld optics, clamping bar pressure, and strip edge preparation shears.",
            "Perform automated weld bulge detection test before clearing coil joint into furnace looper.",
            "Reinforce weld parameter tracking algorithms for stainless alloy chemistry shifts."
        ],
        "reworkable": False
    },
    "laminations": {
        "full_name": "Internal Laminations / Pipe Segregation",
        "category": "Casting / Solidification",
        "root_cause": "Center-line chemical segregation or shrinkage cavity unhealed during breakdown rolling.",
        "metallurgical_hazard": "Planar separation splitting the strip into two sheets under through-thickness tensile stress.",
        "corrective_actions": [
            "Calibrate continuous caster dynamic soft reduction roll segment positioning.",
            "Ensure superheat at tundish remains within target (+15 to +25 C window).",
            "Quarantine coil for ultrasonic flaw verification."
        ],
        "reworkable": False
    },
    "heat_tint": {
        "full_name": "Surface Heat Tint / Oxidation Streak",
        "category": "Bright Annealing Atmosphere",
        "root_cause": "Oxygen or moisture ingress through furnace entry/exit seal curtains causing localized chromium oxide discolouration.",
        "metallurgical_hazard": "Depletes chromium from underlying metal substrate; reduces pitting resistance unless pickled.",
        "corrective_actions": [
            "Monitor bright annealing furnace dew point (must maintain < -55 C).",
            "Inspect entry seal nitrogen purge curtain integrity and gas seals.",
            "Check furnace atmospheric pressure differential to prevent ambient air aspiration."
        ],
        "reworkable": True
    },
    "chatter_marks": {
        "full_name": "Mill Chatter Marks / Harmonic Ripple",
        "category": "Cold Rolling Mill Dynamics",
        "root_cause": "Third or fifth-octave mill vibration resonance caused by roll bearing wear, improper roll grinding, or friction chatter.",
        "metallurgical_hazard": "Microscopic periodic gauge variations and optical zebra striping on reflective surfaces.",
        "corrective_actions": [
            "Reduce mill speed by 10% to move out of resonant vibration frequency envelope.",
            "Inspect backup roll bearings, chocks, and hydraulic cylinder pressure transducers.",
            "Re-grind work rolls with anti-chatter wheel balancing protocols."
        ],
        "reworkable": True
    },
    "pinch_marks": {
        "full_name": "Pinch Roll Indentation Marks",
        "category": "Bridle / Pinch Rollers",
        "root_cause": "Excessive pinch roll hydraulic clamping force or strip tracking overlap pinching strip edge.",
        "metallurgical_hazard": "Localized gouging and cold-work embrittlement inducing localized yield kinks.",
        "corrective_actions": [
            "Recalibrate proportional hydraulic pressure relief valves on pinch roll stands.",
            "Audit automatic strip steering centering units (steering roll transducers).",
            "Check pinch roll polyurethane coating for groove wear or embedded particles."
        ],
        "reworkable": True
    },
    "slivers": {
        "full_name": "Hot Rolled Slivers / Scabs",
        "category": "Breakdown Mill / Ingot Conditioning",
        "root_cause": "Torn metal tongues or slab corner cracks rolled flat and re-bonded superficially during breakdown mill passes.",
        "metallurgical_hazard": "Breaks loose during secondary cold forming; causes tooling punctures and surface ruptures.",
        "corrective_actions": [
            "Audit slab mechanical grinding inspection line for residual corner cracks.",
            "Inspect vertical breakdown mill edging roll draft pass schedules.",
            "Schedule slab conditioning scarfing audit."
        ],
        "reworkable": False
    },
    "gouges": {
        "full_name": "Heavy Mechanical Gouging",
        "category": "Material Handling / Mill Debris",
        "root_cause": "Severe mechanical interference with broken strip guides, coil stripper plates, or debris in looper cars.",
        "metallurgical_hazard": "Deep notch penetrating > 5% nominal strip gauge; triggers stress raisers and customer structural rejection.",
        "corrective_actions": [
            "Immediately stop line and inspect strip passline for foreign metal debris.",
            "Inspect looper car deflector rolls and bridle roll entry chutes.",
            "Quarantine affected coil footage for cut-out."
        ],
        "reworkable": False
    },
    "orange_peel": {
        "full_name": "Orange Peel / Coarse Grain Etch",
        "category": "Annealing Furnace Thermal Profile",
        "root_cause": "Excessive annealing temperature or over-soaking causing excessive austenite grain growth (ASTM grain size < 6).",
        "metallurgical_hazard": "Rough textured pebbled appearance upon forming; degrades cosmetic specular reflectivity.",
        "corrective_actions": [
            "Lower continuous annealing furnace heating zone temperatures by 25-40 C.",
            "Increase strip line speed through annealing furnace to shorten soaking cycle.",
            "Verify optical pyrometer calibration at furnace soaking zone."
        ],
        "reworkable": False
    },
    "edge_burr": {
        "full_name": "Protruding Edge Burr / Slitter Slag",
        "category": "Rotary Slitting Line",
        "root_cause": "Dull rotary slitter knives, incorrect knife horizontal clearance, or excessive knife penetration depth.",
        "metallurgical_hazard": "Burr punctures interleaving paper and scratches adjacent coil laps during winding; safety hazard.",
        "corrective_actions": [
            "Inspect rotary slitter knife cutting edges and redress cutting edges.",
            "Adjust horizontal knife clearance to 8-10% of strip thickness.",
            "Engage edge deburring or edge conditioning rollers on rewinder."
        ],
        "reworkable": True
    },
    "rust_stain": {
        "full_name": "Atmospheric Rust Stain / Iron Contamination",
        "category": "Storage / Warehouse Humidity",
        "root_cause": "Free carbon steel particulate contamination settling on strip in humid environments, creating galvanic micro-rust.",
        "metallurgical_hazard": "Compromises surface passivity; leads to localized tea-staining in architectural installations.",
        "corrective_actions": [
            "Segregate stainless handling equipment (nylon slings, polymer rolls) from carbon steel areas.",
            "Apply chemical passivation wash (citric or nitric acid solution) followed by DI water rinse.",
            "Control warehouse relative humidity (< 50% RH) and packaging barrier wrapping."
        ],
        "reworkable": True
    },
    "cross_bow": {
        "full_name": "Strip Cross-Bow / Flatness Distortion",
        "category": "Shape Control / Skin-Pass Mill",
        "root_cause": "Differential residual stress across strip thickness created during cold rolling or unbalanced tension leveler bending.",
        "metallurgical_hazard": "Strip curls across width, disrupting automated laser cutting tables and continuous stamping dies.",
        "corrective_actions": [
            "Adjust tension leveler anti-crossbow roll cluster penetration.",
            "Apply selective work roll zone cooling on cold rolling stand.",
            "Optimize skin-pass elongation reduction (0.5% - 1.2% target)."
        ],
        "reworkable": True
    }
}

BASE_SEVERITY_SCORES: Dict[str, float] = {
    "crazing": 8.5,
    "inclusion": 7.5,
    "patches": 4.5,
    "pitted_surface": 7.0,
    "rolled-in_scale": 8.0,
    "scratches": 6.0,
    "roll_printing": 6.5,
    "edge_cracks": 8.5,
    "crease": 6.0,
    "crescent_gap": 7.5,
    "waist_folding": 7.0,
    "oil_spot": 3.5,
    "water_stain": 2.5,
    "blister": 8.0,
    "seam_weld_defect": 9.0,
    "laminations": 9.0,
    "heat_tint": 4.0,
    "chatter_marks": 5.5,
    "pinch_marks": 5.0,
    "slivers": 7.5,
    "gouges": 8.5,
    "orange_peel": 4.0,
    "edge_burr": 5.5,
    "rust_stain": 4.5,
    "cross_bow": 5.0,
}


def analyze_defect(
    defect_class: str,
    confidence: float,
    box: List[float],
    image_width: int,
    image_height: int,
    grade_code: str = "SS_304"
) -> Dict[str, Any]:
    """
    Perform deep metallurgical and geometric analysis of a detected surface defect.
    
    Args:
        defect_class: Label name (e.g., 'scratches', 'rolled-in_scale')
        confidence: Model prediction confidence (0.0 to 1.0)
        box: Bounding box [xmin, ymin, xmax, ymax] in pixel coordinates
        image_width: Frame width in pixels
        image_height: Frame height in pixels
        grade_code: Stainless steel grade (e.g., 'SS_304', 'SS_316L')
    """
    xmin, ymin, xmax, ymax = box
    c_xmin = max(0.0, min(float(image_width), min(xmin, xmax)))
    c_xmax = max(0.0, min(float(image_width), max(xmin, xmax)))
    c_ymin = max(0.0, min(float(image_height), min(ymin, ymax)))
    c_ymax = max(0.0, min(float(image_height), max(ymin, ymax)))
    box_w = max(1.0, c_xmax - c_xmin)
    box_h = max(1.0, c_ymax - c_ymin)
    area_px = box_w * box_h
    total_area_px = max(1.0, float(image_width * image_height))
    area_pct = min(100.0, (area_px / total_area_px) * 100.0)
    aspect_ratio = box_w / box_h

    # Check strip zone: Edge vs Center
    # Defects within 12% of lateral border (x-axis) based on defect centroid are edge defects
    edge_margin_px = 0.12 * max(1.0, float(image_width))
    centroid_x = (c_xmin + c_xmax) / 2.0
    is_edge_defect = (centroid_x <= edge_margin_px) or (centroid_x >= (image_width - edge_margin_px))
    strip_zone = "Strip Edge" if is_edge_defect else "Strip Center / Body"

    # Edge proximity multiplier (edge defects escalate strip breakage risk)
    edge_risk_multiplier = 1.35 if is_edge_defect else 1.0

    # Retrieve Grade tolerance profile
    grade_prof = get_grade_profile(grade_code)
    grade_tolerances = grade_prof.get("base_tolerances", {}).get(defect_class, {
        "weight": 1.0,
        "max_acceptable_area_pct": 0.5,
        "critical": False
    })
    grade_weight = grade_tolerances.get("weight", 1.0)
    is_critical_for_grade = grade_tolerances.get("critical", False)
    max_acceptable_area = grade_tolerances.get("max_acceptable_area_pct", 0.5)

    # Base severity calculation (0 to 10 scale)
    base_score = BASE_SEVERITY_SCORES.get(defect_class, 5.0)
    
    # Area penalty factor
    area_penalty = min(2.0, (area_pct / max(0.01, max_acceptable_area)) * 0.5)
    
    # Confidence weight
    conf_weight = 0.5 + 0.5 * confidence

    raw_severity = (base_score * grade_weight + area_penalty) * edge_risk_multiplier * conf_weight
    normalized_severity = round(min(10.0, max(0.1, raw_severity)), 2)

    # Severity tier
    if normalized_severity < 3.5:
        severity_tier = "Minor"
    elif normalized_severity < 6.5:
        severity_tier = "Moderate"
    elif normalized_severity < 8.5:
        severity_tier = "Severe"
    else:
        severity_tier = "Critical"

    # Retrieve domain knowledge taxonomy
    meta_info = EXTENDED_DEFECT_TAXONOMY.get(defect_class, {
        "full_name": defect_class.replace("_", " ").replace("-", " ").title(),
        "category": "Surface Blemish",
        "root_cause": "Process anomaly on rolling or finishing line.",
        "metallurgical_hazard": "Surface quality degradation.",
        "corrective_actions": ["Inspect relevant rolling stand or line guides."],
        "reworkable": True
    })

    return {
        "defect_class": defect_class,
        "full_name": meta_info["full_name"],
        "confidence": round(confidence, 4),
        "bounding_box": [round(c_xmin, 1), round(c_ymin, 1), round(c_xmax, 1), round(c_ymax, 1)],
        "area_pixels": int(area_px),
        "area_pct": round(area_pct, 3),
        "aspect_ratio": round(aspect_ratio, 2),
        "strip_zone": strip_zone,
        "is_edge_defect": is_edge_defect,
        "severity_score": normalized_severity,
        "severity_tier": severity_tier,
        "is_critical_for_grade": is_critical_for_grade,
        "reworkable": meta_info["reworkable"],
        "category": meta_info["category"],
        "root_cause": meta_info["root_cause"],
        "metallurgical_hazard": meta_info["metallurgical_hazard"],
        "corrective_actions": meta_info["corrective_actions"],
        "is_potential_false_alarm": False
    }


def compute_frame_disposition(
    defects: List[Dict[str, Any]],
    grade_code: str = "SS_304",
    is_batch: bool = False,
    total_frames: int = 1
) -> Dict[str, Any]:
    """
    Compute aggregate quality score and disposition recommendation for an inspected frame / coil strip.
    Accurately filters out suppressed false alarms so non-critical candidate anomalies do not scrap prime steel.
    """
    # Exclude verified false alarms from disposition calculation
    active_defects = [
        d for d in defects 
        if not d.get("is_potential_false_alarm", False) and d.get("severity_tier") != "Suppressed / False Alarm"
    ]
    suppressed_count = len(defects) - len(active_defects)

    if not active_defects:
        action_msg = (
            f"All {suppressed_count} candidate anomaly regions verified as non-critical false alarms by deep feature analysis. Cleared for prime dispatch."
            if suppressed_count > 0 else
            "Surface meets prime stainless specification. Clear for downstream slitting/packaging."
        )
        return {
            "disposition": "PRIME / PASS",
            "disposition_code": "PRIME",
            "status_color": "#00e676",  # Vibrant green
            "overall_severity": 0.0,
            "quality_grade": "A+",
            "quality_score_pct": 100.0,
            "action_summary": action_msg,
            "total_defect_count": 0,
            "critical_defect_count": 0,
            "reworkable_defect_count": 0
        }

    total_defects = len(active_defects)
    severities = [d["severity_score"] for d in active_defects]
    max_severity = max(severities)
    # Composite severity considers max defect heavily + aggregate volume
    composite_severity = min(10.0, max_severity * 0.7 + (sum(severities) / total_defects) * 0.3)
    if is_batch and total_frames > 0:
        defect_density = total_defects / float(total_frames)
        density_penalty = min(3.0, defect_density * 0.3)
        composite_severity = min(10.0, composite_severity + density_penalty)
    composite_severity = round(composite_severity, 2)

    critical_count = sum(1 for d in active_defects if d.get("is_critical_for_grade") or d.get("severity_score", 0) >= 7.5)
    reworkable_count = sum(1 for d in active_defects if d.get("reworkable", False))

    quality_score_pct = max(0.0, round(100.0 - (composite_severity * 10.0), 1))

    # Disposition decision matrix
    if composite_severity <= DISPOSITION_THRESHOLDS["prime_max_severity"] and critical_count == 0:
        disposition = "PRIME / PASS"
        disposition_code = "PRIME"
        status_color = "#00e676"  # Emerald green
        quality_grade = "A"
        action_summary = "Minor superficial anomalies within acceptable grade tolerance. Approved for prime dispatch."
    elif composite_severity <= DISPOSITION_THRESHOLDS["rework_max_severity"] and reworkable_count == total_defects:
        disposition = "REWORK REQUIRED"
        disposition_code = "REWORK"
        status_color = "#ffb300"  # Industrial Amber
        quality_grade = "B"
        action_summary = "Defects are superficial (pickling/grinding reworkable). Route coil to skin-pass or conditioning line."
    elif composite_severity <= DISPOSITION_THRESHOLDS["downgrade_max_severity"] and (critical_count <= 1 or max_severity < 4.0):
        disposition = "DOWNGRADE TO COMMERCIAL"
        disposition_code = "DOWNGRADE"
        status_color = "#ff9100"  # Industrial Orange
        quality_grade = "C"
        action_summary = "Surface aesthetics do not meet prime architectural standard. Downgrade to commercial / non-exposed grade."
    else:
        disposition = "SCRAP / REJECT"
        disposition_code = "SCRAP"
        status_color = "#ff1744"  # Crimson Red
        quality_grade = "F"
        action_summary = "Critical structural defect or severe defect concentration detected. Strip fails quality gate; quarantine for scrap/melt recovery."

    return {
        "disposition": disposition,
        "disposition_code": disposition_code,
        "status_color": status_color,
        "overall_severity": composite_severity,
        "quality_grade": quality_grade,
        "quality_score_pct": quality_score_pct,
        "action_summary": action_summary,
        "total_defect_count": total_defects,
        "critical_defect_count": critical_count,
        "reworkable_defect_count": reworkable_count
    }
