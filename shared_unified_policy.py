from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CompactFocusGateConfig:
    min_confidence: float = 0.65
    max_focus_confidence: float = 0.78
    min_context_confidence: float = 0.55
    min_area_ratio: float = 0.010
    grasp_max_area_ratio: float = 0.0135
    place_max_area_ratio: float = 0.0135
    min_horizontal_separation: float = 0.20
    max_vertical_misalignment: float = 0.05
    max_overlap_ratio: float = 1.00
    activation_patience: int = 1
    release_patience: int = 2
    min_active_steps: int = 2
    cooldown_steps: int = 2


@dataclass
class CompactFocusGateState:
    active: bool = False
    support_count: int = 0
    release_count: int = 0
    active_steps: int = 0
    cooldown_remaining: int = 0
    phase: str = "grasp"

    def reset(self) -> None:
        self.active = False
        self.support_count = 0
        self.release_count = 0
        self.active_steps = 0
        self.cooldown_remaining = 0
        self.phase = "grasp"


@dataclass
class SharedTaskPolicyProfile:
    archetype: str = "generic"
    spatial_focus_enabled: bool = False
    spatial_phase_gate: str = "off"
    bg_weight: float = 1.0
    place_src_weight: float = 1.0
    grasp_fovea_weight: float = 1.2
    place_fovea_weight: float = 1.2
    use_grasp_focus: bool = True
    place_foveation_delay: int = 2
    min_grasp_steps: int = 10
    grasp_max_area_ratio: float = 0.95
    place_max_area_ratio: float = 0.95
    grasp_min_confidence: float = 0.30
    place_min_confidence: float = 0.30
    grasp_max_confidence: float = 1.05
    place_max_confidence: float = 1.05
    grasp_min_context_confidence: float = 0.0
    place_min_context_confidence: float = 0.30
    grasp_requires_geometry: bool = False
    place_requires_geometry: bool = True
    min_area_ratio: float = 0.0
    min_horizontal_separation: float = 0.18
    max_vertical_misalignment: float = 0.10
    max_overlap_ratio: float = 1.0
    activation_patience: int = 1
    release_patience: int = 2
    min_active_steps: int = 2
    cooldown_steps: int = 2
    univla_gate_peak_min: float = 0.65
    univla_gate_peak_max: float = 0.78
    univla_gate_context_min: float = 0.55
    univla_gate_area_min: float = 0.010
    univla_gate_area_max: float = 0.0135
    univla_gate_dx_ratio: float = 0.20
    univla_gate_dy_ratio: float = 0.05


def extract_source_dest_nouns(instruction: Optional[str]) -> Dict[str, str]:
    text = (instruction or "").lower().strip()
    pattern = (
        r"(?:put|place|move|stack|push)\s+"
        r"(?:the\s+)?(.+?)\s+"
        r"(?:on(?:\s+top\s+of)?|in(?:to)?|onto|inside)\s+"
        r"(?:the\s+)?(.+)"
    )
    match = re.match(pattern, text)
    if not match:
        return {"source": "", "target": ""}
    return {
        "source": match.group(1).strip().rstrip(".,"),
        "target": match.group(2).strip().rstrip(".,"),
    }


def infer_task_archetype(instruction: Optional[str]) -> str:
    nouns = extract_source_dest_nouns(instruction)
    source = nouns["source"]
    target = nouns["target"]
    text = (instruction or "").lower()

    if "stack" in text or "cube" in source or "block" in source:
        return "stack_alignment"
    if "spoon" in source:
        return "thin_tool"
    if any(token in target for token in ("basket", "bin", "bowl")):
        return "container_drop"
    if any(token in target for token in ("plate", "towel", "cloth", "napkin")):
        return "planar_placement"
    return "generic"


def shared_task_policy_profile(
    instruction: Optional[str],
    *,
    model_family: str,
) -> SharedTaskPolicyProfile:
    archetype = infer_task_archetype(instruction)
    profile = SharedTaskPolicyProfile(archetype=archetype)

    if archetype == "stack_alignment":
        profile.spatial_focus_enabled = True
        profile.spatial_phase_gate = "always"
        profile.place_foveation_delay = 5
        profile.min_grasp_steps = 15
        profile.grasp_max_area_ratio = 0.95
        profile.place_max_area_ratio = 0.95
        profile.grasp_fovea_weight = 1.2
        profile.place_fovea_weight = 1.2
        profile.place_src_weight = 1.0
    elif archetype == "planar_placement":
        profile.spatial_focus_enabled = False
        profile.spatial_phase_gate = "off"
        profile.place_foveation_delay = 2
        profile.min_grasp_steps = 10
        profile.grasp_max_area_ratio = 0.95
        profile.place_max_area_ratio = 0.95
        profile.grasp_fovea_weight = 1.2
        profile.place_fovea_weight = 1.2
        profile.place_src_weight = 1.0
    elif archetype == "container_drop":
        profile.spatial_focus_enabled = False
        profile.spatial_phase_gate = "off"
        profile.place_foveation_delay = 2
        profile.min_grasp_steps = 10
        profile.grasp_max_area_ratio = 0.5
        profile.place_max_area_ratio = 0.6
        profile.grasp_fovea_weight = 1.1
        profile.place_fovea_weight = 1.2
        profile.place_src_weight = 1.0
    elif archetype == "thin_tool":
        profile.spatial_focus_enabled = False
        profile.spatial_phase_gate = "off"
        profile.place_foveation_delay = 5
        profile.min_grasp_steps = 15
        profile.grasp_max_area_ratio = 0.95
        profile.place_max_area_ratio = 0.95
        profile.grasp_fovea_weight = 1.1
        profile.place_fovea_weight = 1.3
        profile.place_src_weight = 1.1

    if model_family == "univla":
        # Keep the UniVLA router conservative; it already has a validated gain.
        if archetype == "planar_placement":
            profile.univla_gate_peak_min = 0.62
            profile.univla_gate_peak_max = 0.80
            profile.univla_gate_context_min = 0.50
            profile.univla_gate_area_min = 0.008
            profile.univla_gate_area_max = 0.016
            profile.univla_gate_dx_ratio = 0.18
            profile.univla_gate_dy_ratio = 0.06
        elif archetype == "stack_alignment":
            profile.univla_gate_peak_min = 0.68
            profile.univla_gate_peak_max = 0.78
            profile.univla_gate_context_min = 0.58
            profile.univla_gate_area_min = 0.010
            profile.univla_gate_area_max = 0.014
            profile.univla_gate_dx_ratio = 0.22
            profile.univla_gate_dy_ratio = 0.05
    return profile


def _phase_gate_fields(
    phase: str,
    profile: SharedTaskPolicyProfile,
) -> Dict[str, float | bool]:
    phase_key = "place" if phase == "place" else "grasp"
    if phase_key == "grasp":
        return {
            "min_confidence": float(profile.grasp_min_confidence),
            "max_confidence": float(profile.grasp_max_confidence),
            "min_context_confidence": float(profile.grasp_min_context_confidence),
            "max_area_ratio": float(profile.grasp_max_area_ratio),
            "requires_geometry": bool(profile.grasp_requires_geometry),
        }
    return {
        "min_confidence": float(profile.place_min_confidence),
        "max_confidence": float(profile.place_max_confidence),
        "min_context_confidence": float(profile.place_min_context_confidence),
        "max_area_ratio": float(profile.place_max_area_ratio),
        "requires_geometry": bool(profile.place_requires_geometry),
    }


def phase_compact_focus_gate(
    *,
    phase: str,
    source_confidence: float,
    target_confidence: float,
    source_area_ratio: float,
    target_area_ratio: float,
    horizontal_separation: float = 0.0,
    vertical_misalignment: float = 1.0,
    overlap_ratio: float = 0.0,
    profile: SharedTaskPolicyProfile,
    state: Optional[CompactFocusGateState] = None,
) -> Dict[str, float | bool | str | int]:
    phase_key = "place" if phase == "place" else "grasp"
    if phase_key == "grasp":
        focus_confidence = float(source_confidence)
        focus_area_ratio = float(source_area_ratio)
        context_confidence = float(target_confidence)
    else:
        focus_confidence = float(target_confidence)
        focus_area_ratio = float(target_area_ratio)
        context_confidence = float(source_confidence)

    phase_fields = _phase_gate_fields(phase_key, profile)
    min_confidence = float(phase_fields["min_confidence"])
    max_confidence = float(phase_fields["max_confidence"])
    min_context_confidence = float(phase_fields["min_context_confidence"])
    max_area_ratio = float(phase_fields["max_area_ratio"])
    requires_geometry = bool(phase_fields["requires_geometry"])

    confidence_support = (
        min_confidence <= focus_confidence <= max_confidence
        and context_confidence >= min_context_confidence
    )
    geometry_support = (
        True
        if not requires_geometry
        else (
            horizontal_separation >= float(profile.min_horizontal_separation)
            and vertical_misalignment <= float(profile.max_vertical_misalignment)
            and overlap_ratio <= float(profile.max_overlap_ratio)
        )
    )
    area_support = (
        float(profile.min_area_ratio)
        <= focus_area_ratio
        <= max_area_ratio
    )
    support = bool(confidence_support and geometry_support and area_support)

    if state is None:
        enabled = support
        active_steps = 1 if enabled else 0
        cooldown_remaining = 0
    else:
        if state.phase != phase_key:
            state.phase = phase_key
            state.support_count = 0
            state.release_count = 0
            state.active_steps = 0
            if not state.active:
                state.cooldown_remaining = 0

        if not state.active and state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1

        if support:
            state.support_count += 1
            state.release_count = 0
        else:
            state.support_count = 0
            state.release_count += 1

        if (
            not state.active
            and state.cooldown_remaining <= 0
            and state.support_count >= int(profile.activation_patience)
        ):
            state.active = True
            state.active_steps = 0
            state.release_count = 0

        if state.active:
            state.active_steps += 1
            if (
                state.release_count >= int(profile.release_patience)
                and state.active_steps >= int(profile.min_active_steps)
            ):
                state.active = False
                state.cooldown_remaining = int(profile.cooldown_steps)
                state.support_count = 0
                state.release_count = 0
                state.active_steps = 0

        enabled = bool(state.active)
        active_steps = int(state.active_steps)
        cooldown_remaining = int(state.cooldown_remaining)

    return {
        "phase": phase_key,
        "focus_confidence": focus_confidence,
        "context_confidence": context_confidence,
        "focus_area_ratio": focus_area_ratio,
        "horizontal_separation": float(horizontal_separation),
        "vertical_misalignment": float(vertical_misalignment),
        "overlap_ratio": float(overlap_ratio),
        "confidence_support": bool(confidence_support),
        "geometry_support": bool(geometry_support),
        "area_support": bool(area_support),
        "support": bool(support),
        "enabled": bool(enabled),
        "active_steps": int(active_steps),
        "cooldown_remaining": int(cooldown_remaining),
    }


def compact_focus_gate(
    *,
    phase: str,
    source_confidence: float,
    target_confidence: float,
    source_area_ratio: float,
    target_area_ratio: float,
    horizontal_separation: float = 0.0,
    vertical_misalignment: float = 1.0,
    overlap_ratio: float = 0.0,
    config: CompactFocusGateConfig,
    state: Optional[CompactFocusGateState] = None,
) -> Dict[str, float | bool | str | int]:
    phase_key = "place" if phase == "place" else "grasp"
    if phase_key == "grasp":
        focus_confidence = float(source_confidence)
        focus_area_ratio = float(source_area_ratio)
        context_confidence = float(target_confidence)
        max_area_ratio = float(config.grasp_max_area_ratio)
    else:
        focus_confidence = float(target_confidence)
        focus_area_ratio = float(target_area_ratio)
        context_confidence = float(source_confidence)
        max_area_ratio = float(config.place_max_area_ratio)

    confidence_support = (
        float(config.min_confidence)
        <= focus_confidence
        <= float(config.max_focus_confidence)
        and context_confidence >= float(config.min_context_confidence)
    )
    geometry_support = (
        horizontal_separation >= float(config.min_horizontal_separation)
        and vertical_misalignment <= float(config.max_vertical_misalignment)
        and overlap_ratio <= float(config.max_overlap_ratio)
    )
    area_support = (
        float(config.min_area_ratio)
        <= focus_area_ratio
        <= max_area_ratio
    )
    support = bool(confidence_support and geometry_support and area_support)

    if state is None:
        enabled = support
        active_steps = 1 if enabled else 0
        cooldown_remaining = 0
    else:
        if state.phase != phase_key:
            state.phase = phase_key
            state.support_count = 0
            state.release_count = 0
            state.active_steps = 0
            if not state.active:
                state.cooldown_remaining = 0

        if not state.active and state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1

        if support:
            state.support_count += 1
            state.release_count = 0
        else:
            state.support_count = 0
            state.release_count += 1

        if (
            not state.active
            and state.cooldown_remaining <= 0
            and state.support_count >= int(config.activation_patience)
        ):
            state.active = True
            state.active_steps = 0
            state.release_count = 0

        if state.active:
            state.active_steps += 1
            if (
                state.release_count >= int(config.release_patience)
                and state.active_steps >= int(config.min_active_steps)
            ):
                state.active = False
                state.cooldown_remaining = int(config.cooldown_steps)
                state.support_count = 0
                state.release_count = 0
                state.active_steps = 0

        enabled = bool(state.active)
        active_steps = int(state.active_steps)
        cooldown_remaining = int(state.cooldown_remaining)

    return {
        "phase": phase_key,
        "focus_confidence": focus_confidence,
        "context_confidence": context_confidence,
        "focus_area_ratio": focus_area_ratio,
        "horizontal_separation": float(horizontal_separation),
        "vertical_misalignment": float(vertical_misalignment),
        "overlap_ratio": float(overlap_ratio),
        "confidence_support": bool(confidence_support),
        "geometry_support": bool(geometry_support),
        "area_support": bool(area_support),
        "support": bool(support),
        "enabled": bool(enabled),
        "active_steps": int(active_steps),
        "cooldown_remaining": int(cooldown_remaining),
    }
