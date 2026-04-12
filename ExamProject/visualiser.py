import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

def plot_runtime_vs_n(results, n_values):
    plt.figure(figsize=(10,5))

    for name, data in results.items():
        grouped = {}
        for n, d in data:
            grouped.setdefault(n, []).append(d.time)

        n_vals = sorted(grouped.keys())
        means = [np.mean(grouped[n]) for n in n_vals]
        stds  = [np.std(grouped[n]) for n in n_vals]

        plt.errorbar(n_vals, means, yerr=stds, marker='o', capsize=5, label=name)

    plt.xlabel("Problem size n")
    plt.ylabel("Time (seconds)")
    plt.title("Solver comparison (runtime)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_iterations_vs_n(results, n_values):
    plt.figure(figsize=(10,5))

    for name, data in results.items():
        grouped = {}
        for n, d in data:
            grouped.setdefault(n, []).append(d.iterations)

        n_vals = sorted(grouped.keys())
        means = [np.mean(grouped[n]) for n in n_vals]
        stds  = [np.std(grouped[n]) for n in n_vals]

        plt.errorbar(n_vals, means, yerr=stds, marker='o', capsize=5, label=name)

    plt.xlabel("Problem size n")
    plt.ylabel("Iterations")
    plt.title("Solver comparison (iterations)")
    plt.legend()
    plt.grid()
    plt.show()

def create_stats_table(results, format="default"):
    rows = []

    # 🔹 Precompute CVXPY reference (for objective error)
    cvxpy_grouped = {}
    for n, d in results["cvxpy"]:
        cvxpy_grouped.setdefault(n, []).append(d.obj)

    cvxpy_mean = {n: np.mean(vals) for n, vals in cvxpy_grouped.items()}

    for solver, data in results.items():
        grouped = {}

        # group by n
        for n, d in data:
            grouped.setdefault(n, []).append(d)

        for n in sorted(grouped.keys()):
            runs = grouped[n]

            times = np.array([r.time for r in runs])
            iters = np.array([r.iterations for r in runs])
            objs  = np.array([r.obj for r in runs])

            feas = [r.feasibility for r in runs if r.feasibility is not None]

            # 🔹 New metrics
            time_per_iter = np.mean([
                r.time / r.iterations for r in runs if r.iterations > 0
            ])

            obj_mean = np.mean(objs)
            obj_error = abs(obj_mean - cvxpy_mean[n]) if solver != "cvxpy" else 0.0

            time_norm = np.mean(times) / (n**3)  # rough scaling indicator

            rows.append({
                "solver": solver,
                "n": n,

                # existing
                "time_mean": np.mean(times),
                "time_std": np.std(times),
                "iter_mean": np.mean(iters),
                "iter_std": np.std(iters),
                "feasible_rate": np.mean(feas) if len(feas) > 0 else np.nan,

                # 🔹 added
                "time_per_iter": time_per_iter,
                "obj_error_vs_cvxpy": obj_error,
                "time_over_n3": time_norm
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["solver", "n"])

    # nicer formatting
    pd.set_option("display.float_format", "{:.6f}".format)

    if format == "latex":
        df.to_latex("results_exercise_2.tex", index=False)
    elif format == "markdown":
        df.to_markdown("results_exercise_2.md", index=False)
    elif format == "jupyter":
        display(df)
    else:
        print(df.to_string(index=False))

    return df

def plot_objective_error_vs_cvxpy(results):
    plt.figure(figsize=(10,5))

    cvxpy_data = {}
    for n, d in results["cvxpy"]:
        cvxpy_data.setdefault(n, []).append(d.obj)

    cvxpy_mean = {n: np.mean(vals) for n, vals in cvxpy_data.items()}

    for solver, data in results.items():
        if solver == "cvxpy":
            continue

        grouped = {}
        for n, d in data:
            grouped.setdefault(n, []).append(d.obj)

        n_vals = sorted(grouped.keys())
        errors = []

        for n in n_vals:
            mean_obj = np.mean(grouped[n])
            ref = cvxpy_mean[n]
            err = abs(mean_obj - ref)
            errors.append(err)

        plt.plot(n_vals, errors, marker='o', label=solver)

    plt.xlabel("Problem size n")
    plt.ylabel("Objective error vs CVXPY")
    plt.title("Solution accuracy comparison")
    plt.legend()
    plt.grid()
    plt.show()

def plot_time_per_iteration(results):
    plt.figure(figsize=(10,5))

    for solver, data in results.items():
        grouped = {}

        for n, d in data:
            if d.iterations > 0:
                grouped.setdefault(n, []).append(d.time / d.iterations)

        n_vals = sorted(grouped.keys())
        means = [np.mean(grouped[n]) for n in n_vals]

        plt.plot(n_vals, means, marker='o', label=solver)

    plt.xlabel("Problem size n")
    plt.ylabel("Time per iteration")
    plt.title("Efficiency per iteration")
    plt.legend()
    plt.grid()
    plt.show()

def plot_runtime_logscale(results):
    plt.figure(figsize=(10,5))

    for solver, data in results.items():
        grouped = {}

        for n, d in data:
            grouped.setdefault(n, []).append(d.time)

        n_vals = sorted(grouped.keys())
        means = [np.mean(grouped[n]) for n in n_vals]

        plt.plot(n_vals, means, marker='o', label=solver)

    plt.yscale("log")  

    plt.xlabel("Problem size n")
    plt.ylabel("Time (log scale)")
    plt.title("Runtime scaling (log scale)")
    plt.legend()
    plt.grid()
    plt.show()

    