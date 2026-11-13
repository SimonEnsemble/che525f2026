import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from matplotlib.ticker import MaxNLocator

    plt.style.use("petroff10")
    plt.rcParams.update({"font.size": 14})
    return MaxNLocator, mo, np, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # weighted sampling
    """)
    return


@app.cell
def _():
    states = ["Pizza", "Salad", "Mac & Cheese"]
    probs = [0.5, 0.1, 0.4]
    return probs, states


@app.cell
def _(probs, random, states):
    random.choices(states, weights=probs)[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Markov chains
    """)
    return


@app.cell
def _(np):
    A = np.array(
        [
            [0.9, 0.2],
            [0.1, 0.8]
        ]
    )
    A
    return (A,)


@app.cell
def _(np):
    def check_left_stochastic(A):
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError("A must be a square matrix.")
        if np.any(A < 0) or not np.allclose(A.sum(axis=0), 1):
            raise ValueError("Each col of A must be non-negative and sum to 1")

    return (check_left_stochastic,)


@app.cell
def _(A, check_left_stochastic):
    check_left_stochastic(A)
    return


@app.cell
def _(np):
    def stationary_distribution(A):
        """
        for left-stochastic A: 
            right eigenvector for eigenvalue 1, normalised to sum to 1.
        """
        eigvals, eigvecs = np.linalg.eig(A)
        pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
        return pi / pi.sum()

    return (stationary_distribution,)


@app.cell
def _(A, stationary_distribution):
    stationary_distribution(A)
    return


@app.cell
def _(check_left_stochastic, np, random):
    def simulate_markov_chain(
        A, init_state, n_steps, seed=None, state_map=None
    ):
        """
        simulate a discrete-time, finite-space, time-homogeneous Markov chain.

        A           : left-stochastic transition matrix.
                      A[i, j] = P(next = i | current = j)
        init_state  : initial state in {0, ..., n-1}
        n_steps     : number of transitions to simulate
        seed        : optional random seed for reproducibility
        state_map   : maps ints to string for interpretable state

        Returns an array of n_steps + 1 visited states (including the start).
        """
        n = A.shape[0] # number of states
        check_left_stochastic(A)
 
        rng = random.Random(seed)
        chain = np.empty(n_steps + 1, dtype=int)
        chain[0] = init_state
        for t in range(1, n_steps + 1):
            chain[t] = rng.choices(range(n), weights=A[:, chain[t-1]])[0]

        if state_map:
            return " ".join(state_map[s] for s in chain)
        return chain

    return (simulate_markov_chain,)


@app.cell
def _(A, simulate_markov_chain):
    simulate_markov_chain(A, 1, 35, state_map=["A", "B"])
    return


@app.cell
def _(A, MaxNLocator, check_left_stochastic, np, plt, stationary_distribution):
    def viz_dynamics(A, init_state, n_steps, state_map):
        """
        plot P(X_t = i) for each state i, 
            for t = 0, ..., n_steps, starting from init_state.

        uses the exact evolution p_{t+1} = A p_t (no sampling), 
            with p_0 = e_{init_state}.
        """
        check_left_stochastic(A)
        p_inf = stationary_distribution(A)
        n = A.shape[0]
 
        P = np.zeros((n_steps + 1, n))
        P[0, init_state] = 1.0
        for t in range(n_steps):
            P[t + 1, :] = A @ P[t, :]
 
        t = np.arange(n_steps + 1)
 
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i in range(n):
            ax.plot(
                t, P[:, i], marker="o", markersize=6, label=state_map[i], lw=2
            )
        for s in range(n):
            ax.axhline(y=p_inf[s], color=f"C{s}", linestyle="--")
        ax.set_xlabel("time, $t$")
        ax.set_ylabel("probability in state $P(X_t = i)$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        plt.xlim(-0.5, n_steps+0.5)
        ax.legend()
        fig.tight_layout()
        plt.show()

    viz_dynamics(A, 0, 20, ["A", "B"])
    return (viz_dynamics,)


@app.cell
def _(check_left_stochastic, np):
    A2 = np.array(
        [
            [0.84, 0.20, 0.0],
            [0.14, 0.75, 0.0],
            [0.02, 0.05, 1.0]
        ]
    )
    check_left_stochastic(A2)
    A2
    return (A2,)


@app.cell
def _(A2, np):
    lambda2, V2 = np.linalg.eig(A2)
    np.real(lambda2)
    return (V2,)


@app.cell
def _(V2, np):
    np.real(V2)
    return


@app.cell
def _(A2, viz_dynamics):
    viz_dynamics(A2, 0, 100, ["A", "B", "T"])
    return


@app.cell
def _(A2):
    A2
    return


@app.cell
def _(A2):
    Q = A2[:2, :2]
    Q
    return (Q,)


@app.cell
def _(A2):
    R = A2[2, :2]
    R
    return (R,)


@app.cell
def _(Q, R, np, plt):
    def viz_chain_length_distn(Q, R, init_state, n_steps):
        n = Q.shape[0]

        # initial state
        q = np.zeros(n)
        q[init_state] = 1.0

        p_absorb = np.zeros(n_steps)
        for t in range(1, n_steps):
            p_absorb[t] = R @ q
            q = Q @ q

        t = np.arange(n_steps)

        mode = np.argmax(p_absorb)
        mean = np.linalg.inv(np.eye(n) - Q).sum(axis=0)[init_state]

        plt.figure()
        plt.xlabel("chain length")
        plt.ylabel("probability")
        plt.plot(
            t, p_absorb, marker="s", markersize=6, lw=2, 
            color="C4", clip_on=False
        )
        plt.axvline(mode, color="k", linestyle=":", label=f"mode = {mode}")
        plt.axvline(mean, color="k", linestyle="--", label=f"mean = {mean}")
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()

    viz_chain_length_distn(Q, R, 0, 150)
    return


if __name__ == "__main__":
    app.run()
