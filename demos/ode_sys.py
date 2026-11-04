import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    plt.style.use("petroff10")
    plt.rcParams.update({"font.size": 14})
    return mo, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # numerical solution to a system of ODEs

    an initial state problem:

    $\dfrac{\rm d}{{\rm d}t}[\mathbf{u}(t)]=A\mathbf{u}(t)$ for $t \geq 0$

    $\mathbf{u}(0)=\mathbf{u}_0$

    ## tanks exchanging fertilizer solution
    """)
    return


@app.cell
def _(np, solve_ivp):
    A = 1/50 * np.array(
        [
            [-1,  1],
            [1, -1]
        ]
    )
    u0 = np.array(
        [
            0.0, 
            1.5
        ]
    )      # initial condition u(0)
    t_span = (0.0, 100.0)           # (t_start, t_end)
    n_points = 1000                # number of output time points
 
    def rhs(t, u):
        """right-hand side of du/dt = A u."""
        return A @ u

    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(
        rhs, t_span, u0, t_eval=t_eval
    )
    return A, sol, u0


@app.cell
def _(A, np, plt, u0):
    def viz_phase_plane(A, u_max, u0=None):
        eigvals, eigvecs = np.linalg.eig(A)

        fig, ax = plt.subplots()
 
        x = np.linspace(0, u_max, 25)
        X, Y = np.meshgrid(x, x)
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y

        ax.streamplot(X, Y, U, V, color="gray")

        if u0 is not None:
            ax.scatter(u0[0], u0[1], clip_on=False, color="C2")

        for lam, vec in zip(eigvals, eigvecs.T):
            v = np.real(vec) / np.linalg.norm(np.real(vec))
            ax.axline((0, 0), slope=v[1] / v[0], ls="--")
        plt.xlim([0, u_max])
        plt.ylim([0, u_max])
        plt.xlabel("$u_1(t)$")
        plt.ylabel("$u_2(t)$")
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    viz_phase_plane(A, 3.0, u0=u0)
    return


@app.cell
def _(np, plt, sol):
    def viz_soln(sol):
        fig, ax = plt.subplots()
        plt.plot(sol.t, sol.y[0], label=r"$u_1(t)$", lw=2)
        ax.plot(sol.t, sol.y[1], label=r"$u_2(t)$", lw=2)
        ax.set_xlabel("time, $t$")
        plt.xlim([0, np.max(sol.t)])
        plt.ylim(ymin=0)
        ax.set_ylabel("solution, $u_i(t)$")
        plt.legend()
        plt.show()

    viz_soln(sol)
    return


@app.cell
def _(plt, sol):
    def viz_trajectory(sol):
        fig, ax = plt.subplots()
        ax.plot(sol.y[0], sol.y[1])
        ax.scatter(sol.y[0][0], sol.y[1][0], clip_on=False, color="C2")
        ax.axline((0, 0), slope=1.0, ls="--", color="gray")
        ax.set_xlabel("time, $t$")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.xlabel("$u_1(t)$")
        plt.ylabel("$u_2(t)$")
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    viz_trajectory(sol)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A->B->C in batch reactor
    """)
    return


@app.cell
def _(np):
    kappa = 2.0

    A_reactor = np.array(
        [
            [-1, 0, 0],
            [1, -kappa, 0],
            [0, kappa, 0]
        ]
    )
    A_reactor
    return A_reactor, kappa


@app.cell
def _(A_reactor, np):
    lambdas, V = np.linalg.eig(A_reactor)
    lambdas
    return V, lambdas


@app.cell
def _(V):
    V
    return


@app.cell
def _(np):
    x0 = np.array([1, 0, 0])
    return (x0,)


@app.cell
def _(V, np, x0):
    c = np.linalg.solve(V, x0)
    c
    return (c,)


@app.cell
def _(V, c, lambdas, np):
    def x_of_t(t):
        return V @ (c * np.exp(lambdas * t))

    return (x_of_t,)


@app.cell
def _(np, x_of_t):
    ts = np.linspace(0, 2.5, 250)
    xs = [x_of_t(t_i) for t_i in ts]
    return ts, xs


@app.cell
def _(B_max, kappa, np, plt, tau_stop, ts, xs):
    plt.figure()
    plt.xlabel("time, $\\tau$")
    plt.ylabel("concentration")
    plt.title("$\kappa$=" + f"{kappa:.1f}")
    plt.plot(ts, [x[0] for x in xs], label="A")
    plt.plot(ts, [x[1] for x in xs], label="B")
    plt.plot(ts, [x[2] for x in xs], label="C")
    plt.scatter(tau_stop, B_max, color="black")
    plt.xlim([0, np.max(ts)])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, ts, xs):
    tau_stop = ts[np.argmax([x[1] for x in xs])]
    tau_stop
    return (tau_stop,)


@app.cell
def _(np, xs):
    B_max = np.max([x[1] for x in xs])
    B_max
    return (B_max,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pendulum
    """)
    return


@app.cell
def _(np):
    g = 9.8      # m/s^2
    m = 0.1      # kg
    gamma = 0.05 # 1/s
    L = 1        # m

    A_pendulum = np.array(
        [
            [0, 1],
            [-g/L, -gamma/m]
        ]
    )
    A_pendulum
    return (A_pendulum,)


@app.cell
def _(A_pendulum, np):
    lambda_pendulum, V_pendulum = np.linalg.eig(A_pendulum)
    lambda_pendulum
    return V_pendulum, lambda_pendulum


@app.cell
def _(lambda_pendulum):
    alpha, beta = lambda_pendulum[0].real, lambda_pendulum[0].imag
    alpha, beta
    return alpha, beta


@app.cell
def _(V_pendulum):
    V_pendulum
    return


@app.cell
def _(V_pendulum):
    a, b = V_pendulum[:, 0].real, V_pendulum[:, 0].imag
    a, b
    return a, b


@app.cell
def _(a, b, np):
    P = np.column_stack([a, b])
    P
    return (P,)


@app.cell
def _(np):
    u0_pendulum = np.array([np.pi/8, 0])
    return (u0_pendulum,)


@app.cell
def _(P, np, u0_pendulum):
    C = np.linalg.solve(P, u0_pendulum)
    C
    return (C,)


@app.cell
def _(C, P):
    P * C
    return


@app.cell
def _(np):
    ts_pendulum = np.linspace(0, 25.0, 500)
    return (ts_pendulum,)


@app.cell
def _(C, P, alpha, beta, np):
    def u_pendulum(t):
        X = np.array(
            [
                [np.cos(beta * t), np.sin(beta * t)],
                [-np.sin(beta * t), np.cos(beta * t)]
            ]
        )
        return np.exp(alpha * t) * P @ X @ C

    return (u_pendulum,)


@app.cell
def _(ts_pendulum, u_pendulum):
    us_pendulum = [u_pendulum(t_i) for t_i in ts_pendulum]
    return (us_pendulum,)


@app.cell
def _(us_pendulum):
    us_pendulum[0]
    return


@app.cell
def _(np, plt, ts_pendulum, us_pendulum):
    plt.figure()
    plt.plot(ts_pendulum, [u[0] for u in us_pendulum], lw=2)
    plt.axhline(0, color="gray", lw=0.5)
    plt.xlim([0, np.max(ts_pendulum)])
    plt.xlabel("time, $t$ [s]")
    plt.ylabel(r"angle, $\theta(t)$ [rad]")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
