import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from dataclasses import dataclass
    from matplotlib.patches import Ellipse

    plt.style.use("petroff10")
    return Ellipse, mo, np, pd, plt, sns


@app.cell
def _(sns):
    colors = sns.color_palette("Set2")
    colors
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ellipses

    :question: what is an ellipse?

    > an ellipse is a plane curve surrounding two focal points, such that for all points on the curve, the sum of both distances to the two focal points is a constant. it generalizes a circle, which is the special type of ellipse in which the two focal points are the same. -[Wikipedia]

    <img src="https://thumb.wikimedia.org/wikipedia/commons/thumb/a/ae/Ellipse-def-e.svg/3840px-Ellipse-def-e.svg.png" width=450>

    🖌️ see how to draw an ellipse [here](https://www.youtube.com/shorts/c-MO1gM2BKE).

    an ellipse is a _conic section_ i.e. obtained by an intersection of a plane with the surface of a cone in 3D space.

    <img src="https://upload.wikimedia.org/wikipedia/commons/1/11/Conic_Sections.svg?utm_source=en.wikipedia.org&utm_campaign=imageinfo&utm_content=original" width=300>

    as an implicit equation for an ellipse, all points $(x, y)$ in the plane must satisfy
    $$x^2 + B x y + C y^2 + D x + E y+ F = 0$$
    with $B,C,D,E,F$ constants that satisfy two conditions:
    1. the discriminant $\Delta := B^2-4C<0$
    2. [non-degeneracy](https://en.wikipedia.org/wiki/Conic_section#Degenerate_cases).

    the constants $B,C,D,E,F$ determine:
    * the two focal points in the plane
    * the sum of distances from the two focal points
    and, thus:
    * the length of the semi-major and semi-minor axes, $a$ & $b$
    * the center of the ellipse, $(x_o, y_o)$
    * the orientation/tilt/angle of the ellipse, $\theta$

    see [Wikipedia](https://en.wikipedia.org/wiki/Ellipse#General_ellipse).

    ## ellipse fitting

    :question: given a set of points tracing out an ellipse-like shape, how do I find the ellipse that fits them best?


    /// note | learning resource
    for background on least-squares fitting of an ellipse, see the "best-fit ellipse" example in: D. Margalit, J. Rabinoff. "The Method of Least Squares". _Interactive Linear Algebra_. [link.](https://textbooks.math.gatech.edu/ila/least-squares.html)
    ///

    ## application: fitting an ellipse to the outline of biological cells

    images of biological cells can be analyzed for statistics on their size and shape. this can help diagnose disease, track health, and/or inform research. given ellipse-shaped cells, fitting an ellipse to each cell in the image is one method to characterize the distribution of the cell sizes and shapes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## let's do an example!
    🩸 read in the data tracing the outlines of six different red blood cells of a patient with hereditary elliptocytosis. the data are stored in the CSV file `cell_outlines.csv`.


    > Elliptocytes, also known as ovalocytes or cigar cells, are abnormally shaped red blood cells that appear oval or elongated, from slightly egg-shaped to rod or pencil forms. They have normal central pallor with the hemoglobin appearing concentrated at the ends of the elongated cells when viewed through a light microscope.
    > -- "Elliptocyte". Wikipedia. [link](https://en.wikipedia.org/wiki/Elliptocyte)

    /// note
    I plot-digitized the outlines of a few cells from this image here. the length-scales are reasonable, but I made them up since they were not provided via a scale bar in the image.
    ///

    <img src="https://thumb.wikimedia.org/wikipedia/commons/thumb/7/7c/Hereditary_Elliptocytosis_in_a_70-year-old_man.tif/lossy-page1-1920px-Hereditary_Elliptocytosis_in_a_70-year-old_man.tif.jpg?utm_source=en.wikipedia.org&utm_campaign=imageinfo&utm_content=thumbnail" width=300>
    """)
    return


@app.cell
def _(pd):
    data = pd.read_csv("https://raw.githubusercontent.com/SimonEnsemble/che525f2026/refs/heads/main/demos/cell_outlines.csv")
    data
    return (data,)


@app.cell
def _(data):
    cells = data["cell"].unique()
    cells
    return (cells,)


@app.cell(hide_code=True)
def _(cells, mo):
    my_dropdown = mo.ui.dropdown(
        options=cells,
        value="A",
        label="choose a cell:"
    )
    my_dropdown
    return (my_dropdown,)


@app.cell
def _(data, my_dropdown):
    data.groupby("cell").get_group(my_dropdown.value)
    return


@app.cell
def _(data, sns):
    cell_to_color = dict(zip(data["cell"].unique(), sns.color_palette("Set2")))
    cell_to_color
    return (cell_to_color,)


@app.cell
def _(cell_to_color, cells, data, plt):
    def viz_data(data, cells):
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        plt.xlabel("x [µm]")
        plt.ylabel("y [µm]")

        # draw data
        for cell in cells:
            data_cell = data.groupby("cell").get_group(cell)
            plt.scatter(
                data_cell["x [µm]"], data_cell["y [µm]"],
                color=cell_to_color[cell],
                label=cell, s=12, zorder=12
            )

        plt.legend()
        return fig, ax

    viz_data(data, cells)
    return (viz_data,)


@app.cell
def _(np):
    def build_linear_system(data, cell):
        data_cell = data.groupby("cell").get_group(cell)

        x = data_cell["x [µm]"].to_numpy()
        y = data_cell["y [µm]"].to_numpy()

        A = np.column_stack([x * y, y**2, x, y, np.ones_like(x)])
        b = -x**2

        return A, b

    return (build_linear_system,)


@app.cell
def _(build_linear_system, data, my_dropdown):
    build_linear_system(data, my_dropdown.value)
    return


@app.cell
def _(build_linear_system, data, my_dropdown):
    A, b = build_linear_system(data, my_dropdown.value)
    A, b
    return A, b


@app.cell
def _(A, b, np):
    x = np.linalg.lstsq(A, b)[0]
    x # [B, C, D, E, F]
    return (x,)


@app.cell
def _(Ellipse, np):
    def conic_to_ellipse(B, C, D, E, F):
        # https://en.wikipedia.org/wiki/Ellipse#Parametric_representation
        Δ = B**2 - 4 * C
        assert Δ < 0.0
        K = E**2 + C*D**2 - B*D*E + Δ*F
        assert K > 0.0

        a = -np.sqrt(
            2*(E**2 + C*D**2 - B*D*E + Δ*F) * (1+C + np.sqrt((1-C)**2 + B**2))
        ) / Δ
        b = -np.sqrt(
            2*(E**2 + C*D**2 - B*D*E + Δ*F) * (1+C - np.sqrt((1-C)**2 + B**2))
        ) / Δ

        x0 = (2 * C * D - B * E) / Δ
        y0 = (2 * 1 * E - B * D) / Δ

        theta = 1/2 * np.atan2(-B, C - 1)

        return Ellipse((x0, y0), 2*a, 2*b, angle=np.degrees(theta),
                       fill=False, color="k", lw=2)

    return (conic_to_ellipse,)


@app.cell
def _(conic_to_ellipse, x):
    ellipse = conic_to_ellipse(*x)
    return


@app.cell
def _(build_linear_system, cells, conic_to_ellipse, data, np, plt, viz_data):
    def fit_all(data, cells):
        fig, ax = viz_data(data, cells)
        ax.set_aspect('equal', 'box')

        # draw data
        for cell in cells:
            # get data pertaining to this cell
            data_cell = data.groupby("cell").get_group(cell)
    
            # fit ellipse
            A, b = build_linear_system(data, cell)
            x = np.linalg.lstsq(A, b)[0]
            ellipse = conic_to_ellipse(*x)

            # draw fitted ellipse
            ax.add_patch(ellipse)

        plt.show()

    fit_all(data, cells)
    return


if __name__ == "__main__":
    app.run()
