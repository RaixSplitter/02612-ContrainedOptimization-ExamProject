from Generators.random_qp import RandomQPGenerator
from QP_solvers.QPsolver import QPsolver
from QP_solvers.PrimalActiveSet import PrimalActiveSetSolver
from QP_solvers.PrimalDualInteriorPoint import PrimalDualInteriorPointSolver
from utils import SolutionStats
from visualiser import plot_objective_error_vs_cvxpy, plot_runtime_logscale, plot_runtime_vs_n, plot_iterations_vs_n, create_stats_table, plot_time_per_iteration

def run_experiment(n_values, density=0.1, alpha=1e-2, beta=0.5):
    results = {
        "cvxpy": [],
        "active_set": [],
        "interior_point": []
    }

    num_trials = 3  

    for n in n_values:
        print(f"\nRunning problem with n = {n}")

        for trial in range(num_trials):  

            generator = RandomQPGenerator(n, alpha, density, beta=0.3, flag="sparse")
            generator.generate()

            # General problem
            H, g, bl, A, bu, l, u = generator.get_general_problem()

            # Interior-point form
            H_ip, g_ip, A_eq, b_eq, C, d = generator.get_interior_point_problem()

            solvers = {
                "cvxpy": QPsolver(H, g, bl, A, bu, l, u),
                "active_set": PrimalActiveSetSolver(H, g, bl, A, bu, l, u),
                "interior_point": PrimalDualInteriorPointSolver(H_ip, g_ip, A_eq, b_eq, C, d)
            }

            for name, solver in solvers.items():
                stats = solver.solve()

                results[name].append((n, stats)) 

                print(f"{name}: obj={stats.obj:.4f}, time={stats.time:.4f}, iter={stats.iterations}")

    return results



def main():
    n_values = [20, 50, 100, 200]

    results = run_experiment(n_values)

    plot_runtime_vs_n(results, n_values)
    plot_runtime_logscale(results)
    plot_iterations_vs_n(results, n_values)
    plot_objective_error_vs_cvxpy(results)
    plot_time_per_iteration(results)

    create_stats_table(results, format="default")

if __name__ == "__main__":
    main()