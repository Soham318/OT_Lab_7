"""
Problems Package: All 14 multiobjective benchmark and engineering problems (P1 to P14).
"""

from __future__ import annotations
from moo.problems.base import Problem
from moo.problems.p01_factory import FactorySpeedProblem
from moo.problems.p02_fleet import FleetSizingProblem
from moo.problems.p03_zdt2 import ZDT2Problem
from moo.problems.p04_portfolio import PortfolioProblem
from moo.problems.p05_cantilever import CantileverBeamProblem
from moo.problems.p06_binh_korn import BinhKornProblem
from moo.problems.p07_water import WaterAllocationProblem
from moo.problems.p08_tsp import DeliveryRoutingTSPProblem
from moo.problems.p09_feature_selection import FeatureSelectionProblem
from moo.problems.p10_dtlz2 import DTLZ2Problem
from moo.problems.p11_production import ProductionPlanningProblem
from moo.problems.p12_supplier import SupplierSelectionProblem
from moo.problems.p13_multitask import MultiTaskToyProblem, SharedMultiTaskNetwork
from moo.problems.p14_laptop import LaptopDesignProblem


PROBLEMS_REGISTRY: dict[str, type[Problem]] = {
    "P1": FactorySpeedProblem,
    "P2": FleetSizingProblem,
    "P3": ZDT2Problem,
    "P4": PortfolioProblem,
    "P5": CantileverBeamProblem,
    "P6": BinhKornProblem,
    "P7": WaterAllocationProblem,
    "P8": DeliveryRoutingTSPProblem,
    "P9": FeatureSelectionProblem,
    "P10": DTLZ2Problem,
    "P10_M3": DTLZ2Problem,
    "P10_M5": lambda: DTLZ2Problem(M=5),
    "P11": ProductionPlanningProblem,
    "P12": SupplierSelectionProblem,
    "P13": MultiTaskToyProblem,
    "P13a": MultiTaskToyProblem,
    "P14": LaptopDesignProblem,
}


def get_problem(problem_id: str, **kwargs) -> Problem:
    """
    Factory helper to instantiate any of the 14 problems by identifier (e.g. 'P1', 'p01', 'P5', 'P10_M5').
    """
    key = problem_id.upper().strip()
    if key.startswith("P0") and len(key) >= 3 and key[2:].isdigit():
        key = "P" + str(int(key[2:]))
    elif key.startswith("P") and key[1:].isdigit():
        key = "P" + str(int(key[1:]))

    if key not in PROBLEMS_REGISTRY:
        raise ValueError(
            f"Unknown problem '{problem_id}'. Available: {list(PROBLEMS_REGISTRY.keys())}"
        )
    factory = PROBLEMS_REGISTRY[key]
    return factory(**kwargs) if callable(factory) else factory()


__all__ = [
    "Problem",
    "FactorySpeedProblem",
    "FleetSizingProblem",
    "ZDT2Problem",
    "PortfolioProblem",
    "CantileverBeamProblem",
    "BinhKornProblem",
    "WaterAllocationProblem",
    "DeliveryRoutingTSPProblem",
    "FeatureSelectionProblem",
    "DTLZ2Problem",
    "ProductionPlanningProblem",
    "SupplierSelectionProblem",
    "MultiTaskToyProblem",
    "SharedMultiTaskNetwork",
    "LaptopDesignProblem",
    "PROBLEMS_REGISTRY",
    "get_problem",
]
