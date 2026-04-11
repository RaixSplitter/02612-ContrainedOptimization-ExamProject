from random_qp import RandomQPGenerator
from QP_solvers.QPsolver import QPsolver
from QP_solvers.PrimalActiveSet import PrimalActiveSetSolver
from QP_solvers.PrimalDualInteriorPoint import PrimalDualInteriorPointSolver
from utils import SolutionStats
import matplotlib.pyplot as plt

def run_experiment(n_values, density=0.1, alpha=1e-2):
    results = {
        "cvxpy": [],
        "active_set": [],
        "interior_point": []
    }

    for n in n_values:
        print(f"\nRunning problem with n = {n}")

        generator = RandomQPGenerator(n, alpha=alpha, density=density)
        generator.generate()

        # General problem
        H, g, bl, A, bu, l, u = generator.get_genral_problem()

        # Interior-point form
        H_ip, g_ip, A_eq, b_eq, C, d = generator.get_interior_point_problem()

        solvers = {
            "cvxpy": QPsolver(H, g, bl, A, bu, l, u),
            "active_set": PrimalActiveSetSolver(H, g, bl, A, bu, l, u),
            "interior_point": PrimalDualInteriorPointSolver(H_ip, g_ip, A_eq, b_eq, C, d)
        }

        for name, solver in solvers.items():
            stats = solver.solve()

            results[name].append(stats)

            print(f"{name}: obj={stats.obj:.4f}, time={stats.time:.4f}, iter={stats.iterations}")

    return results

def plot_results(results):
    plt.figure(figsize=(10,5))

    for name, data in results.items():
        n_vals = [d["n"] for d in data]
        times = [d["time"] for d in data]

        plt.plot(n_vals, times, marker='o', label=name)

    plt.xlabel("Problem size n")
    plt.ylabel("Time (seconds)")
    plt.title("Solver comparison (runtime)")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    n_values = [20, 50, 100, 200]

    results = run_experiment(n_values)

    plot_results(results)



if __name__ == "__main__":
    main()