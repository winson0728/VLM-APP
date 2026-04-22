from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class LineMode(str, Enum):
    ROUTING = "routing"
    BRIDGE = "bridge"


class DistributionMode(str, Enum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    NORMAL = "normal"


class DisconnectMethod(str, Enum):
    NFT_DROP = "nft_drop"
    TC_LOSS = "tc_loss"


class ValueRange(BaseModel):
    min: float = Field(..., ge=0)
    max: float = Field(..., ge=0)
    base: Optional[float] = Field(default=None, ge=0)

    def resolved(self) -> float:
        if self.base is not None:
            return self.base
        return (self.min + self.max) / 2.0


class RandomizationPolicy(BaseModel):
    enabled: bool = True
    update_interval_sec: int = Field(default=5, ge=1, le=3600)
    distribution: DistributionMode = DistributionMode.UNIFORM
    hysteresis_pct: float = Field(default=10.0, ge=0, le=100)


class DisconnectPolicy(BaseModel):
    enabled: bool = False
    probability_per_hour: float = Field(default=0.0, ge=0)
    duration_sec_min: int = Field(default=1, ge=1)
    duration_sec_max: int = Field(default=5, ge=1)
    method: DisconnectMethod = DisconnectMethod.NFT_DROP


class ImpairmentPolicy(BaseModel):
    up_mbps: ValueRange
    down_mbps: ValueRange
    delay_ms: ValueRange
    jitter_ms: ValueRange
    reorder_pct: ValueRange
    reorder_gap: int = Field(default=5, ge=1, le=1000)
    disconnect: DisconnectPolicy = Field(default_factory=DisconnectPolicy)
    randomization: RandomizationPolicy = Field(default_factory=RandomizationPolicy)


class RoutingBinding(BaseModel):
    lan_if: str
    lan_cidr: str
    wan_if: str
    wan_gateway: str
    route_table: int = Field(..., ge=1, le=65535)
    fwmark: Optional[int] = Field(default=None, ge=1, le=65535)


class BridgeBinding(BaseModel):
    port_a: str
    port_b: str
    bridge_name: Optional[str] = None
    stp: bool = False


class ProfileSpec(BaseModel):
    name: str
    description: str
    impairments: ImpairmentPolicy


class LineSpec(BaseModel):
    id: str
    description: str = ""
    mode: LineMode
    profile: Optional[str] = None
    enabled: bool = True
    routing: Optional[RoutingBinding] = None
    bridge: Optional[BridgeBinding] = None
    impairments: ImpairmentPolicy


class LinePatchRequest(BaseModel):
    description: Optional[str] = None
    profile: Optional[str] = None
    enabled: Optional[bool] = None
    routing: Optional[RoutingBinding] = None
    bridge: Optional[BridgeBinding] = None
    impairments: Optional[ImpairmentPolicy] = None


class InterfaceSummary(BaseModel):
    name: str
    role: str
    kind: str
    managed: bool = True


class CommandStep(BaseModel):
    tool: str
    argv: list[str]
    shell: str
    rationale: str


class CommandPhase(BaseModel):
    name: str
    commands: list[CommandStep] = Field(default_factory=list)


class LinePlan(BaseModel):
    line_id: str
    mode: LineMode
    apply: CommandPhase
    disconnect: CommandPhase
    reconnect: CommandPhase
    destroy: CommandPhase
    notes: list[str] = Field(default_factory=list)


class ActionResponse(BaseModel):
    status: str
    line: LineSpec
