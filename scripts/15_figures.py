"""Figuras del manuscrito. Todas generadas por codigo, ninguna a mano.

Salida en figures/ a 600 puntos por pulgada, formato PNG y PDF vectorial.
Paleta y marcadores elegidos para que las figuras sigan siendo legibles al
imprimirse en escala de grises.

Fuente de datos, results/primary/confirmatory_runs.csv y los archivos
derivados en tables/. Ninguna figura vuelve a simular.
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "primary")
TAB = os.path.join(ROOT, "tables")
FIG = os.path.join(ROOT, "figures")
SUBMIT = os.path.join(ROOT, "submission_figures")
DPI = 600

MAIN = ["EKF_nom", "EKF_mis", "CovMatch", "SageHusa", "Huber", "Propuesto"]
LABELS = {"EKF_nom": "Nominal EKF", "EKF_mis": "Miscalibrated EKF",
          "CovMatch": "Covariance matching", "SageHusa": "Sage-Husa",
          "Huber": "Robust Huber", "Propuesto": "Proposed method"}
SCEN_LABEL = {"S0_nominal": "S0 nominal", "S1_noise": "S1 noise",
              "S2_bias": "S2 bias", "S3_outlier": "S3 outliers",
              "S4_dropout": "S4 data loss", "S5_combined": "S5 combined"}

STYLE = {
    "EKF_nom": dict(color="0.15", marker="o", ls="-"),
    "EKF_mis": dict(color="0.45", marker="v", ls=":"),
    "CovMatch": dict(color="#3b6ea5", marker="s", ls="--"),
    "SageHusa": dict(color="#8a5fa8", marker="^", ls="-."),
    "Huber": dict(color="#c08a2e", marker="D", ls="--"),
    "Propuesto": dict(color="#b03030", marker="*", ls="-"),
}


def setup():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(SUBMIT, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "lines.linewidth": 1.1, "lines.markersize": 3.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, name, number):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=DPI)
        fig.savefig(os.path.join(SUBMIT, "Figure_%02d.%s" % (number, ext)),
                    dpi=DPI)
    plt.close(fig)
    print("  escrita figures/%s.png y .pdf" % name)


def fig1_consistency(df):
    """NEES frente a severidad, separado para mejorar la lectura."""
    groups = [
        (["S1_noise", "S2_bias", "S3_outlier"],
         "fig1_consistencia_ruido_sesgo_atipicos", 1),
        (["S4_dropout", "S5_combined"],
         "fig2_consistencia_perdida_combinado", 2),
    ]
    for scens, name, number in groups:
        fig, axes_grid = plt.subplots(
            2, len(scens), figsize=(7.2, 4.8), squeeze=False,
            gridspec_kw={"height_ratios": [4.0, 1.25], "hspace": 0.0},
            constrained_layout=False)
        axes = axes_grid[0]
        for legend_ax in axes_grid[1]:
            legend_ax.axis("off")
        for ax, scen in zip(axes, scens):
            sub = df[(df.escenario == scen) & df.estimador.isin(MAIN)]
            piv = sub.groupby(["estimador", "severidad"])["nees"].mean().unstack()
            for est in MAIN:
                ax.plot(piv.columns, piv.loc[est], label=LABELS[est],
                        **STYLE[est])
            ax.axhspan(4.975, 7.120, color="0.75", alpha=0.35, lw=0,
                       zorder=0)
            ax.set_yscale("log")
            ax.set_title(SCEN_LABEL[scen])
            ax.set_xlabel("Severity")
        axes[0].set_ylabel("Mean NEES")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="center",
                   bbox_to_anchor=(0.5, 0.105), frameon=False, ncol=3,
                   fontsize=8, labelspacing=0.15, columnspacing=0.8,
                   handlelength=1.8, handletextpad=0.35)
        fig.subplots_adjust(bottom=0.04, top=0.94, wspace=0.16)
        save(fig, name, number)


def fig2_rank(df):
    """H1. Correlacion de ordenamientos con tres objetivos de lazo cerrado."""
    d = pd.read_csv(os.path.join(TAB, "h1_rank_correlations.csv"))
    names = {"track_rmse": "Mean tracking\nerror",
             "track_max": "Maximum\nexcursion",
             "recovery": "Recovery\ntime"}
    order = list(names)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                                  gridspec_kw={"width_ratios": [1.0, 1.4]})
    data = [d[d.objetivo == k]["spearman"].to_numpy() for k in order]
    bp = ax.boxplot(data, tick_labels=[names[k] for k in order],
                    widths=0.55, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("0.85")
        patch.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("#b03030")
        med.set_linewidth(1.4)
    ax.axhline(1.0, color="0.4", lw=0.6, ls=":")
    ax.set_ylabel("Spearman between RMSE rank\nand closed-loop rank")
    ax.set_title("Thirty scenario-severity cells")

    piv = (d[d.objetivo == "recovery"]
           .pivot(index="escenario", columns="severidad", values="spearman"))
    im = ax2.imshow(piv.to_numpy(), cmap="RdYlBu", vmin=-1, vmax=1,
                    aspect="auto")
    ax2.set_xticks(range(piv.shape[1]))
    ax2.set_xticklabels(["%.2f" % c for c in piv.columns])
    ax2.set_yticks(range(piv.shape[0]))
    ax2.set_yticklabels([SCEN_LABEL.get(i, i) for i in piv.index])
    ax2.set_xlabel("Severity")
    ax2.set_title("Recovery time by cell")
    ax2.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            if np.isfinite(v):
                ax2.text(j, i, "%.2f" % v, ha="center", va="center",
                         fontsize=6, color="0.1")
    fig.colorbar(im, ax=ax2, shrink=0.85, label="Spearman correlation")
    save(fig, "fig3_ordenamiento_h1", 3)


def fig3_bias(df):
    """Punto ciego ante sesgo, mediana y rango intercuartilico."""
    arms = ["EKF_nom", "CovMatch", "SageHusa", "Huber", "Propuesto",
            "Abl_sin_media"]
    sub = df[(df.escenario == "S2_bias") & df.estimador.isin(arms)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    metrics = [("nees", "Mean NEES", True),
               ("pos_rmse", "Position RMSE, m", False),
               ("track_max", "Maximum excursion, m", False)]
    for ax, (m, lab, logy) in zip(axes, metrics):
        grouped = sub.groupby(["estimador", "severidad"])[m]
        med = grouped.median().unstack()
        q1 = grouped.quantile(0.25).unstack()
        q3 = grouped.quantile(0.75).unstack()
        for est in [e for e in MAIN if e in med.index]:
            ax.plot(med.columns, med.loc[est], label=LABELS[est], **STYLE[est])
            ax.fill_between(med.columns, q1.loc[est], q3.loc[est],
                            color=STYLE[est]["color"], alpha=0.08, lw=0)
        est = "Abl_sin_media"
        ax.plot(med.columns, med.loc[est], color="#b03030", ls=":",
                marker="x", label="Proposed method without\nfirst-moment test")
        ax.fill_between(med.columns, q1.loc[est], q3.loc[est],
                        color="#b03030", alpha=0.08, lw=0)
        if logy:
            ax.set_yscale("log")
            ax.axhspan(4.975, 7.120, color="0.75", alpha=0.35, lw=0, zorder=0)
        ax.set_xlabel("Bias severity")
        ax.set_ylabel(lab)
    axes[2].legend(loc="upper left", frameon=False, fontsize=6)
    fig.suptitle("Magnetometer bias with drift", y=1.05)
    save(fig, "fig4_sesgo_deriva", 4)


def fig4_ablation(df):
    """Escalera de ablacion. Precio en nominal frente a beneficio en sesgo."""
    arms = ["EKF_nom", "Abl_solo_inflacion", "Abl_sin_media", "Abl_sin_zona",
            "Abl_sin_permanencia", "Propuesto"]
    names = {"EKF_nom": "Nominal EKF", "Abl_solo_inflacion": "Inflation only",
             "Abl_sin_media": "Without first-moment\ntest",
             "Abl_sin_zona": "Without dead\nzone",
             "Abl_sin_permanencia": "Without minimum\ndwell time",
             "Propuesto": "Complete method"}
    nom = (df[(df.escenario == "S0_nominal") & (df.severidad == 0.0)]
           .groupby("estimador")["pos_rmse"].median())
    bias = (df[(df.escenario == "S2_bias") & (df.severidad == 1.0)]
            .groupby("estimador")["pos_rmse"].median())
    sw = (df[(df.escenario == "S0_nominal") & (df.severidad == 0.0)]
          .groupby("estimador")["sw_mag"].mean())
    arms = [a for a in arms if a in nom.index and a in bias.index]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = np.arange(len(arms))
    ax.bar(x - 0.2, [nom[a] for a in arms], 0.38, color="0.6",
           label="Nominal")
    ax.bar(x + 0.2, [bias[a] for a in arms], 0.38, color="#b03030",
           label="Bias, severity one")
    ax.set_xticks(x)
    ax.set_xticklabels([names[a] for a in arms], rotation=35, ha="right",
                       fontsize=6)
    ax.set_ylabel("Position RMSE, m")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("Benefit and cost of each component")

    ax2.bar(x, [sw.get(a, 0.0) for a in arms], 0.6, color="0.35")
    ax2.set_xticks(x)
    ax2.set_xticklabels([names[a] for a in arms], rotation=35, ha="right",
                        fontsize=6)
    ax2.set_ylabel("Switches per run")
    ax2.set_title("Spurious switching under nominal conditions")
    save(fig, "fig5_ablacion", 5)


def fig5_cost(df):
    """Costo relativo y resultado de la comparacion pareada."""
    ct = pd.read_csv(os.path.join(TAB, "computational_cost.csv"),
                     index_col=0)
    hh = pd.read_csv(os.path.join(TAB, "head_to_head_sev1.csv"))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ct = ct.loc[[e for e in MAIN if e in ct.index]]
    ax.barh([LABELS[i] for i in ct.index], ct["relativo"], color="0.5")
    ax.axvline(1.0, color="0.2", lw=0.8, ls=":")
    ax.set_xlabel("Cost relative to nominal EKF")
    ax.set_title("Computational cost")

    mets = ["pos_rmse", "track_rmse", "track_max"]
    mlab = ["Estimation\nRMSE", "Mean\ntracking", "Maximum\nexcursion"]
    win, lose, tie = [], [], []
    for m in mets:
        s = hh[hh.metrica == m]
        win.append(int(((s.mejor == "Propuesto") & s.significativo).sum()))
        lose.append(int(((s.mejor != "Propuesto") & s.significativo).sum()))
        tie.append(len(s) - win[-1] - lose[-1])
    y = np.arange(len(mets))
    ax2.barh(y, win, color="#3b6ea5", label="Proposed wins")
    ax2.barh(y, tie, left=win, color="0.75", label="No difference")
    ax2.barh(y, lose, left=np.array(win) + np.array(tie), color="#b03030",
             label="Proposed loses")
    ax2.set_yticks(y)
    ax2.set_yticklabels(mlab)
    ax2.set_xlabel("Number of comparisons, thirty per metric")
    ax2.set_title("Paired comparison at severity one")
    ax2.legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "fig6_costo_y_comparacion", 6)


def fig6_compensation(df):
    """Frontera de compromiso entre exclusion y compensacion de sesgo."""
    arms = ["EKF_nom", "Propuesto", "Compensado", "Comp_siempre"]
    labels = {"EKF_nom": "Nominal EKF", "Propuesto": "Exclude",
              "Compensado": "Triggered compensation",
              "Comp_siempre": "Always compensate"}
    colors = {"EKF_nom": "0.35", "Propuesto": "#b03030",
              "Compensado": "#3b6ea5", "Comp_siempre": "#4f8a55"}
    nom = df[(df.escenario == "S0_nominal") & (df.severidad == 0.0)]
    bias = df[(df.escenario == "S2_bias") & (df.severidad == 1.0)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for arm in arms:
        x = nom[nom.estimador == arm].pos_rmse
        y = bias[bias.estimador == arm].pos_rmse
        ax.scatter(x.median(), y.median(), s=34, color=colors[arm],
                   label=labels[arm], zorder=3)
        ax.errorbar(x.median(), y.median(),
                    xerr=[[x.median() - x.quantile(0.25)],
                          [x.quantile(0.75) - x.median()]],
                    yerr=[[y.median() - y.quantile(0.25)],
                          [y.quantile(0.75) - y.median()]],
                    fmt="none", color=colors[arm], capsize=2, lw=0.8)
    ax.set_xlabel("Nominal RMSE, m")
    ax.set_ylabel("Bias RMSE, m")
    ax.set_title("Median and interquartile range")
    ax.legend(frameon=False, fontsize=6)

    data = [bias[bias.estimador == a].pos_rmse.to_numpy() for a in arms]
    bp = ax2.boxplot(data, tick_labels=[labels[a] for a in arms],
                     patch_artist=True, showfliers=True)
    for patch, arm in zip(bp["boxes"], arms):
        patch.set_facecolor(colors[arm])
        patch.set_alpha(0.35)
    ax2.set_ylabel("Bias RMSE, m")
    ax2.set_title("Variation across forty repetitions")
    ax2.tick_params(axis="x", rotation=25)
    save(fig, "fig7_compensacion_sesgo", 7)


def main():
    setup()
    df = pd.read_csv(os.path.join(RES, "confirmatory_runs.csv"))
    df2 = pd.read_csv(os.path.join(RES, "confirmatory_v2_runs.csv"))
    print("generando figuras a %d puntos por pulgada" % DPI)
    fig1_consistency(df)
    fig2_rank(df)
    fig3_bias(df2)
    fig4_ablation(df2)
    fig5_cost(df)
    fig6_compensation(df2)
    print("listo")


if __name__ == "__main__":
    main()
