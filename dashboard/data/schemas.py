"""
Typed Schema Definitions for the Frozen Step 5.7 Policy Configuration,
Dashboard Policy Record, and Scenario Record (Sections 14, 15, 16).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dashboard.config.policy_config import (
    ALLOWED_FALLBACK_SOURCES,
    ALLOWED_LEAD_TIMES,
    ALLOWED_POLICY_STATUSES,
    ALLOWED_SERVICE_LEVELS,
    FROZEN_BASE_LEAD_TIME_DAYS,
    FROZEN_BASE_SERVICE_LEVEL,
    FROZEN_BASE_Z_VALUE,
    FROZEN_POLICY_VERSION,
)


@dataclass(frozen=True)
class PolicyConfigSchema:
    """Typed model representing the frozen Step 5.7 policy configuration (Section 14)."""

    policy_name: str
    policy_version: str
    base_lead_time_days: int
    base_service_level: float
    base_z_value: float
    base_z_score: float
    scenario_lead_times: List[int]
    scenario_service_levels: List[float]
    scenario_z_values: Dict[str, float]
    mean_min_history: int
    std_min_history: int
    fallback_hierarchy: List[str]
    inventory_level_time_semantics: str
    order_quantity_supported: bool
    demand_forecast_used_in_policy: bool
    units_ordered_used_as_on_order: bool
    raw_payload: Dict[str, Any]

    def validate(self) -> None:
        """Validate that the loaded configuration strictly adheres to the frozen policy."""
        if self.base_lead_time_days != FROZEN_BASE_LEAD_TIME_DAYS:
            raise ValueError(
                f"Invalid base lead time {self.base_lead_time_days}; expected {FROZEN_BASE_LEAD_TIME_DAYS}"
            )
        if abs(self.base_service_level - FROZEN_BASE_SERVICE_LEVEL) > 1e-9:
            raise ValueError(
                f"Invalid base service level {self.base_service_level}; expected {FROZEN_BASE_SERVICE_LEVEL}"
            )
        if abs(self.base_z_value - FROZEN_BASE_Z_VALUE) > 1e-9:
            raise ValueError(
                f"Invalid base Z value {self.base_z_value}; expected {FROZEN_BASE_Z_VALUE}"
            )
        if sorted(self.scenario_lead_times) != sorted(ALLOWED_LEAD_TIMES):
            raise ValueError(
                f"Invalid scenario lead times {self.scenario_lead_times}; expected {ALLOWED_LEAD_TIMES}"
            )
        if sorted(self.scenario_service_levels) != sorted(ALLOWED_SERVICE_LEVELS):
            raise ValueError(
                f"Invalid scenario service levels {self.scenario_service_levels}; expected {ALLOWED_SERVICE_LEVELS}"
            )
        if self.mean_min_history != 7 or self.std_min_history != 30:
            raise ValueError(
                f"Invalid history requirements ({self.mean_min_history}, {self.std_min_history}); expected (7, 30)"
            )
        if self.inventory_level_time_semantics != "unknown":
            raise ValueError(
                f"Invalid inventory_level_time_semantics '{self.inventory_level_time_semantics}'; expected 'unknown'"
            )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyConfigSchema":
        base = payload.get("base_policy", {})
        scen = payload.get("scenario_parameters", {})
        hist = payload.get("history_requirements", {})
        fb_raw = payload.get(
            "fallback_hierarchy",
            ["STORE_PRODUCT", "PRODUCT", "STORE", "GLOBAL_TRAIN"],
        )
        if isinstance(fb_raw, dict):
            fb_list = [x for x in fb_raw.get("order", []) if x != "INSUFFICIENT_DATA"]
        else:
            fb_list = list(fb_raw)

        z_val = float(
            base.get("z_value", base.get("service_level_z", FROZEN_BASE_Z_VALUE))
        )
        inst = cls(
            policy_name=str(
                payload.get("policy_name", "Risk-Based Inventory Replenishment Framework")
            ),
            policy_version=FROZEN_POLICY_VERSION,
            base_lead_time_days=int(
                base.get("lead_time_days", FROZEN_BASE_LEAD_TIME_DAYS)
            ),
            base_service_level=float(
                base.get("service_level", FROZEN_BASE_SERVICE_LEVEL)
            ),
            base_z_value=z_val,
            base_z_score=z_val,
            scenario_lead_times=[
                int(x) for x in scen.get("lead_time_days", ALLOWED_LEAD_TIMES)
            ],
            scenario_service_levels=[
                float(x) for x in scen.get("service_levels", ALLOWED_SERVICE_LEVELS)
            ],
            scenario_z_values={
                str(k): float(v)
                for k, v in scen.get(
                    "z_values", {"0.90": 1.2816, "0.95": 1.6449, "0.99": 2.3263}
                ).items()
            },
            mean_min_history=int(hist.get("mean_min_history", 7)),
            std_min_history=int(hist.get("std_min_history", 30)),
            fallback_hierarchy=fb_list,
            inventory_level_time_semantics=str(
                payload.get("inventory_level_time_semantics", "unknown")
            ),
            order_quantity_supported=False,
            demand_forecast_used_in_policy=False,
            units_ordered_used_as_on_order=False,
            raw_payload=payload,
        )
        inst.validate()
        return inst


@dataclass(frozen=True)
class DashboardPolicyRecord:
    """Typed model representing a single Store x Product x Date dashboard decision (Section 15)."""

    date: str
    store_id: str
    product_id: str
    category: str
    region: str
    split_label: str
    inventory_level: float
    units_sold: float
    units_ordered_reference: float
    demand_forecast_reference: Optional[float]
    historical_demand_count: int
    demand_baseline: Optional[float]
    demand_baseline_source: str
    demand_std: Optional[float]
    demand_std_source: str
    lead_time_days: int
    service_level: float
    z_score: float
    lead_time_demand: Optional[float]
    safety_stock: Optional[float]
    reorder_point: Optional[float]
    inventory_gap_to_rop: Optional[float]
    days_of_supply_estimate: Optional[float]
    estimated_stockout_probability: Optional[float]
    policy_status: str
    order_quantity_supported: bool = False
    inventory_level_time_semantics: str = "unknown"

    def validate(self) -> None:
        if self.policy_status not in ALLOWED_POLICY_STATUSES:
            raise ValueError(
                f"Invalid policy_status '{self.policy_status}'; allowed: {ALLOWED_POLICY_STATUSES}"
            )
        if self.demand_baseline_source not in ALLOWED_FALLBACK_SOURCES:
            raise ValueError(
                f"Invalid demand_baseline_source '{self.demand_baseline_source}'"
            )
        if self.demand_std_source not in ALLOWED_FALLBACK_SOURCES:
            raise ValueError(
                f"Invalid demand_std_source '{self.demand_std_source}'"
            )
        if self.inventory_level_time_semantics != "unknown":
            raise ValueError(
                f"Expected inventory_level_time_semantics='unknown', got '{self.inventory_level_time_semantics}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioRecordSchema:
    """Typed model representing one scenario in the 3x3 frozen scenario matrix (Section 16)."""

    scenario_id: str
    lead_time_days: int
    service_level: float
    z_score: float
    is_base_policy: bool
    lead_time_demand: Optional[float]
    safety_stock: Optional[float]
    reorder_point: Optional[float]
    policy_status: str
    historical_reorder_rate: Optional[float] = None
    historical_stockout_coverage: Optional[float] = None
    historical_reorder_precision: Optional[float] = None
    historical_false_reorder_rate: Optional[float] = None
    historical_avg_safety_stock: Optional[float] = None
    historical_avg_rop: Optional[float] = None

    def validate(self) -> None:
        if self.lead_time_days not in ALLOWED_LEAD_TIMES:
            raise ValueError(f"Invalid scenario lead time: {self.lead_time_days}")
        if self.policy_status not in ALLOWED_POLICY_STATUSES:
            raise ValueError(f"Invalid scenario policy status: {self.policy_status}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
