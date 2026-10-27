import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14})
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # discrete dynamical systems

    a system is characterized by state vector $\mathbf{x}_t\in\mathbb{R}^n$ at discrete time $t=0,1,\dots$, and governed by the linear difference equation:

    $$\mathbf{x}_{t+1}=A\mathbf{x}_{t}$$

    with $A$ an $n\times n$ matrix specifying the dynamics. the dynamics are subject to an initial condition $\mathbf{x}_0$.
    """)
    return


@app.cell
def _(np):
    def simulate_trajectory(A, x0, T):
        """
        simulate the trajectory of a discrete dynamical system 
        for time t = 0, 1, .., T. returns the sequence of states.
        """
        xs = [x0]
        for t in range(T - 1):
            xs.append(A @ xs[-1])
        return np.array(xs)

    return (simulate_trajectory,)


@app.cell
def _(np, plt):
    def viz_trajectory(xs, state_var_1, state_var_2, equal_axes=True):
        """
        visualize the trajectory in the 2D state space.
        """
        assert len(xs[0]) == 2
        T = len(xs)

        fig, ax = plt.subplots()
        plt.xlabel(state_var_1)
        plt.ylabel(state_var_2)
        cmap = plt.get_cmap("viridis", T)          # T discrete colors
        sc = plt.scatter(xs[:, 0], xs[:, 1], c=np.arange(T),
                         cmap=cmap, vmin=-0.5, vmax=T - 0.5, zorder=100)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        if equal_axes:
            plt.gca().set_aspect('equal', 'box')
        plt.colorbar(sc, label="year, $t$", ticks=np.arange(T))
        return fig

    return (viz_trajectory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## the frog and midges example

    (from David Lay's textbook _Linear Algebra and its Applications_.)
    """)
    return


@app.cell
def _(np):
    A = np.array([
        [0.7, 0.1],
        [-0.5, 1.3]
    ])
    return (A,)


@app.cell
def _(np):
    x0 = np.array([7, 9])
    return (x0,)


@app.cell
def _(A, simulate_trajectory, x0):
    T = 15
    xs = simulate_trajectory(A, x0, 15)
    return (xs,)


@app.cell
def _(plt, viz_trajectory, xs):
    viz_trajectory(xs, "# frogs / 100", "# midges / 100,000")
    plt.axline((0, 0), slope=5, linestyle="--", color="black")
    return


@app.cell
def _(A, np):
    eig_res = np.linalg.eig(A)
    lambda_1, lambda_2 = eig_res.eigenvalues
    v1, v2 = eig_res.eigenvectors.T # cuz eigenvectors in columns
    return lambda_1, lambda_2, v1, v2


@app.cell
def _(lambda_1, lambda_2):
    lambda_1, lambda_2 # eigenvalues
    return


@app.cell
def _(v1, v2):
    v1, v2 # eigenvectors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    check
    """)
    return


@app.cell
def _(A, v1):
    A @ v1
    return


@app.cell
def _(lambda_1, v1):
    lambda_1 * v1
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
