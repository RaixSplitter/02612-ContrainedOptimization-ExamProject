import matplotlib.pyplot as plt


def plot_runtime_vs_n(results, n_values):
    plt.figure(figsize=(10,5))

    for name, data in results.items():
        times = [d.time for d in data]

        plt.plot(n_values[:len(times)], times, marker='o', label=name)

    plt.xlabel("Problem size n")
    plt.ylabel("Time (seconds)")
    plt.title("Solver comparison (runtime)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_iterations_vs_n(results, n_values):
    plt.figure(figsize=(10,5))

    for name, data in results.items():
        iterations = [d.iterations for d in data]

        plt.plot(n_values[:len(iterations)], iterations, marker='o', label=name)

    plt.xlabel("Problem size n")
    plt.ylabel("Iterations")
    plt.title("Solver comparison (iterations)")
    plt.legend()
    plt.grid()
    plt.show()
