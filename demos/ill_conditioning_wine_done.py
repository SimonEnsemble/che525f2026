import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator
    import pandas as pd
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    import numpy as np
    import mpltern

    plt.style.use("petroff10")
    return AutoMinorLocator, MultipleLocator, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::streamline-emojis:white-wine-glass:: un-blending problem with UQ

    suppose a Bordeaux-style white wine was produced from a post-fermentation blend of Sémillon, Sauvignon Blanc, and Mauzac pure-varietal wines with the measured attributes below.

    | wine | acid [g/L] | sugar [g/L] |
    | -- | -- | --  |
    | Sémillon | 5.4 | 4.3 |
    | Sauv. Blanc | 6.2 | 5.2 |
    | Mauzac | 5.2 | 4.8 |
    | blend | 5.7 | 4.9 |

    based on these attributes, predict the make-up of the blend (volume %) in terms of the pure-varietal wines.


    **reference** for problem setup: C. Simon, T. Onufer, A. G. Andrade, E. Tomasino. "Estimating the make-up of a blend in terms of its parent mixtures (e.g., reverse-engineering a wine blend) based on chemical fingerprints." _ChemRxiv_. 2026. [link](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001815/v1).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## build linear system

    assumptions:

    * well-mixed blend
    * no acid-based chemistry or chemical reactions in general
    * no excess volume of mixing
    * no evaporation over the course of blending
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## solve linear system
    """)
    return


@app.cell
def _(np):
    A = np.array([
        [5.4, 6.2, 5.2],
        [4.3, 5.2, 4.8],
        [1, 1, 1]
    ])
    A
    return (A,)


@app.cell
def _(np):
    b = np.array([
        5.7,
        4.9,
        1
    ])
    b
    return (b,)


@app.cell
def _(A, b, np):
    x = np.linalg.solve(A, b)
    x
    return (x,)


@app.cell
def _(plt, x):
    wines = ["Sémillon", "Sauv. Blanc", "Mauzac"]

    fig, ax = plt.subplots()
    ax.pie(x, labels=wines, autopct='%1.1f%%')
    plt.show()
    return (wines,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## quantify uncertainty in the solution

    suppose we can measure the acid and sugar [g/L] only within $\pm$0.05 g/L. fill in the confidence region of the blend make-up using Monte Carlo uncertainty propogation.

    (assume independent, additive Gaussian measurement noise with $\sigma=0.05/2$ g/L.)
    """)
    return


@app.cell
def _(np, plt):
    plt.figure()
    plt.hist(0.05 / 2 * np.random.randn(1000)) # unit normal
    plt.xlabel("error [g/L]")
    plt.title("distribution of errors")
    return


@app.cell
def _(np):
    def perturbed_problem(A, b):
        sigma = 0.05 / 2 # g/L

        m = A.shape[0]
        n = A.shape[1]
    
        assert len(b) == n

        # # clear but inefficnet
        # A_new = np.copy(A)
        # b_new = np.copy(b)
        # for i in range(n):
        #     b_new[i] = b[i] + sigma * np.random.randn()
        #     for j in range(m):
        #         A_new[i, j] = A[i, j] + sigma * np.random.randn()

        # simple
        b_new = b + sigma * np.random.randn(3)
        A_new = A + sigma * np.random.randn(m, n)
    
        return A_new, b_new

    return (perturbed_problem,)


@app.cell
def _(A, b, np, perturbed_problem):
    xs = []
    n_sims = 250
    for s in range(n_sims):
        A_new, b_new = perturbed_problem(A, b)
        x_new = np.linalg.solve(A_new, b_new)
        xs.append(x_new)
    # xs
    return (xs,)


@app.cell
def _(AutoMinorLocator, MultipleLocator, plt, wines, xs):
    ax2 = plt.subplot(projection="ternary")
    ax2.scatter(
        [x[0] for x in xs], 
        [x[1] for x in xs], 
        [x[2] for x in xs], 
        s=64.0, c="C1", edgecolors="k", alpha=0.6
    )

    ax2.set_tlabel(wines[0])
    ax2.set_llabel(wines[1])
    ax2.set_rlabel(wines[2])

    ax2.taxis.set_major_locator(MultipleLocator(0.25))
    ax2.laxis.set_major_locator(MultipleLocator(0.20))
    ax2.raxis.set_major_locator(MultipleLocator(0.10))

    ax2.laxis.set_minor_locator(MultipleLocator(0.1))
    ax2.raxis.set_minor_locator(AutoMinorLocator(5))

    ax2.grid(axis='t')
    ax2.grid(axis='l', which='minor', linestyle='--')
    ax2.grid(axis='r', which='both', linestyle=':')

    plt.show()
    return


if __name__ == "__main__":
    app.run()
