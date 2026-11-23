import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize_scalar

    plt.style.use("petroff6")
    plt.rcParams.update({
        "font.size": 14,
        "lines.linewidth": 2.5,
    })
    return minimize_scalar, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modeling heat transfer to a lime

    Data from F. Waqar, S. Patel, C. Simon, "A tutorial on the Bayesian statistical
    approach to inverse problems," *APL Machine Learning* (2023),
    [link](https://pubs.aip.org/aip/aml/article/1/4/041101/2919936/A-tutorial-on-the-Bayesian-statistical-approach-to).

    a lime sat in a refrigerator for several days. at time $t=0$, it was taken out of the refrigerator, placed onto a styrofoam slab, and left to warm in room air by natural convection---while its internal temperature was logged.

    <img src="https://raw.githubusercontent.com/SimonEnsemble/CHE361_W2024/main/studios/lime_setup.jpeg" width="320">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## data
    """)
    return


@app.cell
def _(pd):
    theta_air = 18.7  # °C, room air
    theta_0 = 6.5     # °C, refrigerator air

    data = pd.read_csv("train_data.csv")
    data
    return data, theta_0, theta_air


@app.cell
def _(plt, theta_0, theta_air):
    def viz_data(data):
        fig = plt.figure()
        plt.scatter(data["t [hr]"], data["θ [°C]"], label="data", zorder=3)
        plt.axhline(theta_air, color="gray", linestyle="--")
        plt.axhline(theta_0, color="gray", linestyle="--")
        plt.xlabel("time [hr]")
        plt.ylabel("lime temp. [°C]")
        return fig

    return (viz_data,)


@app.cell
def _(data, viz_data):
    viz_data(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## model
    """)
    return


@app.cell
def _(np, theta_0, theta_air):
    def theta(t, tau):
        """
        lime temperature [°C] at time t [hr] for time constant tau [hr].
        """
        return theta_0 + (theta_air - theta_0) * (1 - np.exp(-t / tau))

    return (theta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## conceptual parameter exploration
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0.1, stop=5, step=0.1, label="tau [hr]")
    slider
    return (slider,)


@app.cell
def _(slider):
    tau = slider.value
    tau
    return (tau,)


@app.cell
def _(data, plt, tau, tau_opt, theta, ts, viz_data):
    viz_data(data)
    plt.plot(
        ts, theta(ts, tau), color="C1", label=rf"model, $\tau$ = {tau_opt:.2f} hr"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## parameter identification
    """)
    return


@app.cell
def _(data, np, theta):
    def loss(tau):
        """
        sum of squared residuals between the model and the data.
        """
        return np.sum((data["θ [°C]"] - theta(data["t [hr]"], tau)) ** 2)

    return (loss,)


@app.cell
def _(loss, minimize_scalar):
    tau_opt = minimize_scalar(
        loss, bounds=(0.1, 2.0), method="bounded"
    ).x
    print(f"tau_opt = {tau_opt:.3f} hr")
    return (tau_opt,)


@app.cell
def _(loss, np, plt, tau_opt):
    taus = np.linspace(0.1, 2.0, 150)

    plt.figure()
    plt.plot(taus, [loss(_tau) for _tau in taus])
    plt.axvline(tau_opt, color="gray", linestyle="--")
    plt.xlabel(r"$\tau$ [hr]")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(data, np, plt, tau_opt, theta, viz_data):
    ts = np.linspace(0, 10, 150)

    viz_data(data)
    plt.plot(
        ts, theta(ts, tau_opt), color="C1", label=rf"model, $\tau$ = {tau_opt:.2f} hr"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    return (ts,)


if __name__ == "__main__":
    app.run()
