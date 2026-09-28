import marimo

__generated_with = "0.25.0"
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
    return mo, sns


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

    :question: given a set of points tracing out an ellipse-like shape, how do I find the equation of the ellipse that fit them best?


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


if __name__ == "__main__":
    app.run()
