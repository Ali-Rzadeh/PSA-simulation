"""Interfaces the PSA simulation with IPOPT via cyipopt."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cycle import psa_cycle


@dataclass
class PsaIpoptProblem:
    """Optimization wrapper for PSA operating pressures.

    The decision variables are the high, intermediate and purge pressures used by
    the PSA cycle.  The remaining process variables are held fixed using the
    provided template.
    """

    material: Sequence[Sequence[float]]
    n: int
    run_type: str
    base_process_variables: Sequence[float]
    purity_target: float = 0.90
    recovery_target: float = 0.90
    pressure_bounds: Optional[Sequence[Tuple[float, float]]] = None
    ipopt_options: Optional[Sequence[Tuple[str, float | int | str]]] = None

    template: List[float] = field(init=False)
    last_pressures: Optional[np.ndarray] = field(default=None, init=False)
    last_purity: Optional[float] = field(default=None, init=False)
    last_recovery: Optional[float] = field(default=None, init=False)
    last_energy: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.template = list(self.base_process_variables)
        if len(self.template) < 8:
            raise ValueError("Process variables template must include pressure slots.")
        if self.pressure_bounds is None:
            self.pressure_bounds = (
                (2.0e5, 1.0e6),   # High pressure bounds [Pa]
                (5.0e3, 5.0e5),   # Intermediate pressure bounds [Pa]
                (1.0e3, 5.0e4),   # Purge pressure bounds [Pa]
            )
        if len(self.pressure_bounds) != 3:
            raise ValueError("Pressure bounds must be provided for three decision variables.")
        self.ipopt_options = list(self.ipopt_options or [])

    # Utility -----------------------------------------------------------------
    def _evaluate_pressures(self, pressures: Sequence[float]) -> Tuple[float, float, float]:
        """Run the PSA cycle for a set of candidate pressures."""
        high_p, intermediate_p, purge_p = pressures
        process_variables = self.template.copy()
        process_variables[1] = float(high_p)
        process_variables[6] = float(intermediate_p)
        process_variables[7] = float(purge_p)

        purity, recovery, _productivity, energy_requirement, *_ = psa_cycle(
            process_variables,
            self.material,
            None,
            self.run_type,
            self.n,
        )

        self.last_pressures = np.array(pressures, dtype=float)
        self.last_purity = float(purity)
        self.last_recovery = float(recovery)
        self.last_energy = float(energy_requirement)
        return self.last_purity, self.last_recovery, self.last_energy

    def _ensure_latest(self, pressures: Sequence[float]) -> None:
        if self.last_pressures is None:
            self._evaluate_pressures(pressures)
            return
        if not np.allclose(self.last_pressures, pressures, rtol=0, atol=1e-9):
            self._evaluate_pressures(pressures)

    # IPOPT callbacks ---------------------------------------------------------
    def objective(self, pressures: Sequence[float]) -> float:
        """Return the specific energy requirement for the candidate pressures."""
        _, _, energy = self._evaluate_pressures(pressures)
        return energy

    def gradient(self, _pressures: Sequence[float]) -> Optional[np.ndarray]:  # type: ignore[override]
        """Let IPOPT approximate the gradient via finite differences."""
        return None

    def constraints(self, pressures: Sequence[float]) -> Iterable[float]:  # type: ignore[override]
        """Constraint residuals for purity and recovery targets."""
        self._ensure_latest(pressures)
        assert self.last_purity is not None and self.last_recovery is not None
        return [
            self.last_purity - self.purity_target,
            self.last_recovery - self.recovery_target,
        ]

    def jacobian(self, _pressures: Sequence[float]) -> Optional[np.ndarray]:  # type: ignore[override]
        """Let IPOPT approximate the Jacobian via finite differences."""
        return None

    def hessian(self, _pressures: Sequence[float], _lagrange: Sequence[float], _obj_factor: float) -> Optional[np.ndarray]:  # type: ignore[override]
        """Use IPOPT's internal Hessian approximation."""
        return None

    # Helpers -----------------------------------------------------------------
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower, upper = zip(*self.pressure_bounds)  # type: ignore[arg-type]
        return np.array(lower, dtype=float), np.array(upper, dtype=float)

    @property
    def constraint_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(2, dtype=float)
        upper = np.full(2, np.inf, dtype=float)
        return lower, upper

    def create_problem(self) -> "cyipopt.Problem":
        import cyipopt

        lb, ub = self.bounds
        cl, cu = self.constraint_bounds
        problem = cyipopt.Problem(
            n=3,
            m=2,
            problem_obj=self,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        problem.addOption("tol", 1e-6)
        problem.addOption("max_iter", 200)
        problem.addOption("constr_viol_tol", 1e-6)
        problem.addOption("print_level", 5)
        problem.addOption("hessian_approximation", "limited-memory")
        problem.addOption("mu_strategy", "adaptive")
        for key, value in self.ipopt_options:
            problem.addOption(str(key), value)
        return problem

    def solve(self, initial_pressures: Sequence[float]) -> Tuple[np.ndarray, dict]:
        problem = self.create_problem()
        solution, info = problem.solve(np.array(initial_pressures, dtype=float))
        self._ensure_latest(solution)
        return solution, info
