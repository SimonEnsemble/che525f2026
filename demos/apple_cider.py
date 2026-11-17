import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from scipy.optimize import brentq

    plt.style.use("petroff6")
    plt.rcParams.update({"font.size": 13, "lines.linewidth": 3})
    return brentq, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # warm-up: start-up of a CSTR

    mole balances on A and B, with the reactor initially full of pure solvent:

    $$\frac{dC_A}{dt} = \frac{q}{V}\left(C_{A,\mathrm{in}} - C_A\right) - k\,C_A, \qquad \frac{dC_B}{dt} = -\frac{q}{V}\,C_B + k\,C_A, \qquad C_A(0) = C_B(0) = 0$$
    """)
    return


@app.cell
def _():
    # ---------------- parameters ----------------
    q = 10.0        # flow rate [L/min]
    cA_in = 2.0     # concentration of A in feed [mol/L]
    V = 100.0       # reactor volume [L]
    k = 0.2         # rate constant [1/min]
    return V, cA_in, k, q


@app.cell
def _(V, cA_in, k, np, q):
    tau_cstr = 1 / (q / V + k)              # time constant for A [min]
    cA_infty = cA_in * (q / V) * tau_cstr   # steady-state concentration of A [mol/L]

    def cs_of_t(t):
        """
        concentrations of A and B [mol/L] at time t [min] after start-up.
        """
        cA = tau_cstr * (1 - np.exp(-t / tau_cstr))
        cB = tau_cstr * (
            k * V / q - (k * V / q + 1) * np.exp(-q * t / V) + np.exp(-t / tau_cstr)
        )
        return cA, cB

    return (cs_of_t,)


@app.cell
def _(cs_of_t, np, plt):
    _t = np.linspace(0.0, 60.0, 300)
    _cA, _cB = cs_of_t(_t)

    plt.figure()
    plt.plot(_t, _cA, label="A")
    plt.plot(_t, _cB, label="B")
    plt.xlabel("time, $t$ [min]")
    plt.ylabel("concentration [mol/L]")
    plt.xlim([0, np.max(_t)])
    plt.ylim(ymin=0)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # batch pasteurization of apple cider

    energy balance on the well-mixed cider, with the bottom of the tank insulated:

    $$\rho V c_p \frac{dT}{dt} = \underbrace{U_{sc}\,(2\pi r L)}_{a}\,(T_s - T) - \underbrace{U_{ca}\,(2\pi R H)}_{b}\,(T - T_0), \qquad T(0) = T_0$$

    solution: $T(t) = T_\infty - (T_\infty - T_0)\,e^{-t/\tau}$, with $T_\infty = \dfrac{a\,T_s + b\,T_0}{a + b}$ and $\tau = \dfrac{\rho V c_p}{a + b}$.
    """)
    return


@app.cell
def _():
    # ---------------- parameters ----------------
    R = 0.40            # tank radius [m]
    H = 1.00            # height of cider in tank [m]
    rho = 1050.0        # density of cider [kg/m^3]
    cp = 3.85           # specific heat of cider [kJ/(kg C)]
    T0 = 20.0           # room temperature = initial cider temperature [C]
    Ts = 110.0          # temperature of saturated steam [C]
    lam = 2230.0        # latent heat of steam at Ts [kJ/kg]
    r = 0.0125          # outer radius of coil tube [m]
    L = 10.0            # length of coil [m]
    U_sc = 42.0         # overall coefficient, steam to cider [kJ/(min m^2 C)]
    U_ca = 0.6          # overall coefficient, cider to air [kJ/(min m^2 C)]
    T_star = 71.0       # pasteurization temperature [C]
    t_max = 40.0        # target time to pasteurize [min]
    return H, L, R, T0, T_star, Ts, U_ca, U_sc, cp, lam, r, rho, t_max


@app.cell
def _(H, L, R, T0, Ts, U_ca, U_sc, cp, np, r, rho):
    a = U_sc * 2 * np.pi * r * L
    b = U_ca * 2 * np.pi * R * H

    tau = rho * np.pi * R**2 * H * cp / (a + b)

    T_infty = (a * Ts + b * T0) / (a + b)
    return T_infty, a, tau


@app.cell
def _(T0, T_infty, Ts, a, lam, np, tau):
    def T_of_t(t):
        """
        cider temperature [C] at time t [min].
        """
        return T_infty + (T0 - T_infty) * np.exp(-t / tau)

    def steam_rate(t):
        """
        steam consumption [kg/min] at time t [min] = heat through coil / latent heat.
        """
        return a * (Ts - T_of_t(t)) / lam

    return T_of_t, steam_rate


@app.cell
def _(T_of_t, T_star, np, plt, steam_rate):
    _ts = np.linspace(0.0, 60.0, 300)
    _fig, _axs = plt.subplots(2, 1, sharex=True, figsize=(5, 5))

    # top: cider temperature
    _axs[0].plot(_ts, T_of_t(_ts))
    _axs[0].axhline(T_star, linestyle="--", color="gray", zorder=0)
    _axs[0].text(
        0.03, T_star, "pasteurization temperature",
        transform=_axs[0].get_yaxis_transform(),   # x in axes fraction, y in data units
        ha="left", va="center", color="gray", fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    _axs[0].set_ylabel("cider temperature\n$T$ [°C]")

    # bottom: steam consumption
    _axs[1].plot(_ts, steam_rate(_ts), color="C1")
    _axs[1].set_ylabel("steam use\n[kg/min]")
    _axs[1].set_ylim(bottom=0)
    _axs[1].set_xlabel("time, $t$ [min]")
    _axs[1].set_xlim([0, np.max(_ts)])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(T0, T_infty, T_star, mo, np, tau):
    # (1) time to pasteurize
    t_star = tau * np.log((T_infty - T0) / (T_infty - T_star)) # closed-form

    mo.md(
        f"**(1)** the batch is pasteurized after **{t_star:.1f} min**"
    )
    return (t_star,)


@app.cell
def _(T0, T_infty, T_star, Ts, a, lam, mo, t_star, tau):
    # (2) steam consumption over the batch
    m_steam = a / lam * ((Ts - T_infty) * t_star + tau * (T_star - T0))
    mo.md(
        f"(2) one batch uses **{m_steam:.1f} kg** of steam."
    )
    return


@app.cell
def _(H, L, R, T0, T_star, Ts, U_ca, U_sc, brentq, cp, mo, np, r, rho, t_max):
    # (3) coil length needed to pasteurize within t_max
    def time_to_pasteurize(L):
        """
        time [min] for the cider to reach T_star with a coil of length L_coil [m].
        """
        a = U_sc * 2 * np.pi * r * L
        b = U_ca * (np.pi * R**2 + 2 * np.pi * R * H)

        tau = rho * np.pi * R**2 * H * cp / (a + b)

        T_infty = (a * Ts + b * T0) / (a + b)

        t_star = tau * np.log((T_infty - T0) / (T_infty - T_star))
        return t_star

    # bracket: the current coil is too slow, and a 50 m coil is fast enough
    L_needed = brentq(lambda L_coil: time_to_pasteurize(L_coil) - t_max, L, 50.0)

    mo.md(
        f"**(3)** to pasteurize within {t_max:.0f} min, the coil must be at least "
        f"**{L_needed:.1f} m** long (it is {L:.0f} m now)."
    )
    return


@app.cell
def _(T_infty, T_of_t, brentq, mo, tau):
    # (4) steam left on
    _T_boil = 100.0   # approximate boiling point of cider [C]
    _t_boil = brentq(lambda t: T_of_t(t) - _T_boil, 0.0, 10 * tau)

    mo.md(
        f"**(4)** with the steam left on, the cider approaches **{T_infty:.1f} °C**, "
        f"passing {_T_boil:.0f} °C after {_t_boil:.0f} min."
        f"the model does not capture boiling or evaporation, so it breaks down there."
    )
    return


if __name__ == "__main__":
    app.run()
