from __future__ import annotations

import json
from pathlib import Path

from netsim_common.models import ProfileSpec


def load_builtin_profiles() -> dict[str, ProfileSpec]:
    profile_path = (
        Path(__file__).resolve().parent.parent
        / "netsim_agent"
        / "profiles"
        / "defaults.json"
    )
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = [ProfileSpec.model_validate(item) for item in raw]
    return {profile.name: profile for profile in profiles}
