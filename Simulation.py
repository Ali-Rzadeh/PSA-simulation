from __future__ import annotations

import argparse
import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from cycle import psa_cycle


def default_material() -> list[list[float]]:
    material_property = [1.130000e03, -3.600000e04, -1.580000e04]
    isotherm_params = [
        3.090000e00,
        2.540000e00,
        8.650000e-07,
        2.630000e-08,
        -3.664121e04,
        -3.569066e04,
        5.840000e00,
        0.000000e00,
        2.500000e-06,
        0.000000e00,
        -1.580000e04,
        0.000000e00,
        1.000000e00,
    ]
    return [material_property, isotherm_params]


def default_process_variables() -> list[float]:
    opt_row = [7.63336e05, 4.60896e01, 9.51027e-02, 9.65920e-01, 1.0, 1.04979e04, 2.94]
    return [
        1.0,
        float(opt_row[0]),
        float(opt_row[0] * opt_row[3] / 8.314 / 313.15),
        float(opt_row[1]),
        float(opt_row[2]),
        float(opt_row[4]),
        1.0e4,
        float(opt_row[5]),
    ]


def run_psa_cycle(
    process_variables: Sequence[float],
    material: Sequence[Sequence[float]],
    run_type: str,
    n: int,
):
    return psa_cycle(process_variables, material, None, run_type, n)


def plot_profiles(b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray, n: int, prefix: str) -> None:
    b2 = b[2 * n + 4 : 3 * n + 6, -1]
    c2 = c[2 * n + 4 : 3 * n + 6, -1]
    b3 = b[2 * n + 4 : 3 * n + 6, 0]
    c3 = c[2 * n + 4 : 3 * n + 6, 0]

    d4 = d[2 * n + 4 : 3 * n + 6, -1]
    e4 = e[2 * n + 4 : 3 * n + 6, -1]
    d5 = d[2 * n + 4 : 3 * n + 6, 0]
    e5 = e[2 * n + 4 : 3 * n + 6, 0]
    concat = np.concatenate((b2, c2), axis=0)
    plt.plot(concat, label="CO2 concentration at adsorption")
    plt.title("CO2 concentration at adsorption step time = end")
    plt.savefig(f"{prefix}1solid.png")
    plt.show()

    concat2 = np.concatenate((b3, c3), axis=0)
    plt.plot(concat2, label="CO2 concentration at adsorption")
    plt.title("CO2 concentration at adsorption step time = 0")
    plt.savefig(f"{prefix}2solid.png")
    plt.show()

    concat3 = np.concatenate((d4, e4), axis=0)
    plt.plot(concat3, label="CO2 concentration at adsorption")
    plt.title("CO2 concentration at desorption step time = end")
    plt.savefig(f"{prefix}3solid.png")
    plt.show()

    concat4 = np.concatenate((d5, e5), axis=0)
    plt.plot(concat4, label="CO2 concentration at adsorption")
    plt.title("CO2 concentration at desorption step time = 0")
    plt.savefig(f"{prefix}4solid.png")
    plt.show()

    b2 = b[n + 2 : 2 * n + 4, -1]
    c2 = c[n + 2 : 2 * n + 4, -1]
    b3 = b[n + 2 : 2 * n + 4, 0]
    c3 = c[n + 2 : 2 * n + 4, 0]

    d4 = d[n + 2 : 2 * n + 4, -1]
    e4 = e[n + 2 : 2 * n + 4, -1]
    d5 = d[n + 2 : 2 * n + 4, 0]
    e5 = e[n + 2 : 2 * n + 4, 0]
    concat = np.concatenate((b2, c2), axis=0)
    plt.plot(concat, label="CO2 concentration at adsorption")
    plt.title("gas mole fraction at adsorption step time = end")
    plt.savefig(f"{prefix}1mole.png")
    plt.show()

    concat2 = np.concatenate((b3, c3), axis=0)
    plt.plot(concat2, label="CO2 concentration at adsorption")
    plt.title("gas mole fraction at adsorption step time = 0")
    plt.savefig(f"{prefix}2mole.png")
    plt.show()

    concat3 = np.concatenate((d4, e4), axis=0)
    plt.plot(concat3, label="CO2 concentration at adsorption")
    plt.title("gas mole fraction at desorption step time = end")
    plt.savefig(f"{prefix}3mole.png")
    plt.show()

    concat4 = np.concatenate((d5, e5), axis=0)
    plt.plot(concat4, label="CO2 concentration at adsorption")
    plt.title("gas mole fraction at desorption step time = 0")
    plt.savefig(f"{prefix}4mole.png")
    plt.show()


def run_base_case(process_variables: Sequence[float], material: Sequence[Sequence[float]], run_type: str, n: int, plot: bool) -> None:
    start_time = time.time()
    purity, recovery, productivity, energy_requirement, b, c, d, e = run_psa_cycle(
        process_variables,
        material,
        run_type,
        n,
    )
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")
    print("Purity:", purity)
    print("Recovery:", recovery)
    print("Productivity:", productivity)
    print("Energy Requirement:", energy_requirement)

    if plot:
        plot_profiles(b, c, d, e, n, prefix="")


def run_optimization(
    process_variables: Sequence[float],
    material: Sequence[Sequence[float]],
    run_type: str,
    n: int,
    purity_target: float,
    recovery_target: float,
    plot: bool,
) -> None:
    from optimization.ipopt_driver import PsaIpoptProblem

    initial_pressures = [process_variables[1], process_variables[6], process_variables[7]]
    psa_problem = PsaIpoptProblem(
        material=material,
        n=n,
        run_type=run_type,
        base_process_variables=process_variables,
        purity_target=purity_target,
        recovery_target=recovery_target,
    )

    solution, info = psa_problem.solve(initial_pressures)

    print("IPOPT status:", info.get("status"))
    print("Objective value (specific energy):", psa_problem.last_energy)
    print("Purity:", psa_problem.last_purity)
    print("Recovery:", psa_problem.last_recovery)
    print("Optimized pressures (high, intermediate, purge) [Pa]:", solution)

    if plot:
        optimized_variables = list(process_variables)
        optimized_variables[1] = float(solution[0])
        optimized_variables[6] = float(solution[1])
        optimized_variables[7] = float(solution[2])
        purity, recovery, productivity, energy_requirement, b, c, d, e = run_psa_cycle(
            optimized_variables,
            material,
            run_type,
            n,
        )
        print("Post-optimization productivity:", productivity)
        print("Post-optimization energy requirement:", energy_requirement)
        plot_profiles(b, c, d, e, n, prefix="optimized_")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or optimize the PSA simulation.")
    parser.add_argument("--optimize", action="store_true", help="Optimize operating pressures using IPOPT.")
    parser.add_argument("--purity-target", type=float, default=0.90, help="Minimum CO2 purity for optimization.")
    parser.add_argument(
        "--recovery-target",
        type=float,
        default=0.90,
        help="Minimum CO2 recovery for optimization.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting concentration profiles.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    material = default_material()
    process_variables = default_process_variables()
    run_type = "ProcessEvaluation"
    n = 30

    plot = not args.no_plots

    if args.optimize:
        run_optimization(
            process_variables,
            material,
            run_type,
            n,
            purity_target=args.purity_target,
            recovery_target=args.recovery_target,
            plot=plot,
        )
    else:
        run_base_case(process_variables, material, run_type, n, plot=plot)


if __name__ == "__main__":
    main()
