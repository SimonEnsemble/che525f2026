import marimo

__generated_with = "0.25.0"
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
    return (mo,)


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

    *
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## solve linear system
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## quantify uncertainty in the solution

    suppose we can measure the alcohol and sugar [g/L] only within $\pm$0.05 g/L. fill in the confidence region of the blend make-up using Monte Carlo uncertainty propogation.

    (assume independent, additive Gaussian measurement noise with $\sigma=0.05/2$ g/L.)
    """)
    return


if __name__ == "__main__":
    app.run()
