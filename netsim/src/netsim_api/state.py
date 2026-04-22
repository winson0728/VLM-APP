from __future__ import annotations

from copy import deepcopy

from netsim_common.models import (
    BridgeBinding,
    InterfaceSummary,
    LinePatchRequest,
    LineSpec,
    RoutingBinding,
)
from netsim_common.profiles import load_builtin_profiles


class InMemoryStore:
    def __init__(self) -> None:
        self._profiles = load_builtin_profiles()
        self._lines: dict[str, LineSpec] = {}
        self._interfaces = [
            InterfaceSummary(name="mgmt0", role="management", kind="ethernet", managed=False),
            InterfaceSummary(name="br-lan", role="lan", kind="bridge", managed=True),
            InterfaceSummary(name="enp3s0", role="wan", kind="ethernet", managed=True),
            InterfaceSummary(name="enp4s0", role="line-port", kind="ethernet", managed=True),
            InterfaceSummary(name="enp5s0", role="line-port", kind="ethernet", managed=True),
            InterfaceSummary(name="enp6s0", role="line-port", kind="ethernet", managed=True),
        ]

    def list_interfaces(self) -> list[InterfaceSummary]:
        return deepcopy(self._interfaces)

    def list_profiles(self):
        return [profile.model_copy(deep=True) for profile in self._profiles.values()]

    def get_profile(self, name: str):
        return self._profiles[name].model_copy(deep=True)

    def list_lines(self) -> list[LineSpec]:
        return [line.model_copy(deep=True) for line in self._lines.values()]

    def get_line(self, line_id: str) -> LineSpec:
        return self._lines[line_id].model_copy(deep=True)

    def create_line(self, line: LineSpec) -> LineSpec:
        candidate = line.model_copy(deep=True)
        if candidate.profile:
            candidate.impairments = self.get_profile(candidate.profile).impairments
        self._lines[candidate.id] = candidate
        return candidate.model_copy(deep=True)

    def patch_line(self, line_id: str, patch: LinePatchRequest) -> LineSpec:
        current = self._lines[line_id].model_copy(deep=True)
        if patch.description is not None:
            current.description = patch.description
        if patch.enabled is not None:
            current.enabled = patch.enabled
        if patch.profile is not None:
            current.profile = patch.profile
            current.impairments = self.get_profile(patch.profile).impairments
        if patch.impairments is not None:
            current.impairments = patch.impairments
        if patch.routing is not None:
            current.routing = RoutingBinding.model_validate(patch.routing)
        if patch.bridge is not None:
            current.bridge = BridgeBinding.model_validate(patch.bridge)
        self._lines[line_id] = current
        return current.model_copy(deep=True)

    def apply_profile(self, line_id: str, profile_name: str) -> LineSpec:
        profile = self.get_profile(profile_name)
        current = self._lines[line_id].model_copy(deep=True)
        current.profile = profile_name
        current.impairments = profile.impairments
        self._lines[line_id] = current
        return current.model_copy(deep=True)

    def set_enabled(self, line_id: str, enabled: bool) -> LineSpec:
        current = self._lines[line_id].model_copy(deep=True)
        current.enabled = enabled
        self._lines[line_id] = current
        return current.model_copy(deep=True)
