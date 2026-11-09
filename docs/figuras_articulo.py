"""Figuras compactas para el artículo (columna de 7,5 cm), a partir de los registros de las pruebas."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datos_documentos import R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pruebas"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ia_mano as ia  # noqa: E402
import simulador as sm  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "articulo_figuras"
SALIDA.mkdir(exist_ok=True)
# figuras en escala de grises legible (la revista se imprime monocroma) con un acento
NEGRO, GRIS, CLARO = "#222222", "#8a8a8a", "#cfcfcf"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "lines.linewidth": 1.2, "legend.frameon": False})
ANCHO = 3.35                                                     # pulgadas ≈ 8,5 cm (se inserta a 7,5 cm)


def guardar(fig, nombre):
    fig.savefig(SALIDA / f"{nombre}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def corte():
    fig = plt.figure(figsize=(ANCHO, 3.0))
    a = fig.add_subplot(projection="polar")
    a.set_theta_zero_location("E")
    for k, m in enumerate(sm.NOMBRES_MUSCULOS):
        a.scatter(np.radians(sm.ANG[k]), sm.PROF[k], s=170, color="white", edgecolor=NEGRO, lw=0.8, zorder=3)
        a.text(np.radians(sm.ANG[k]), sm.PROF[k], m, ha="center", va="center", fontsize=5.6, zorder=4)
    for c in range(10):
        a.scatter(np.radians(sm.ELECTRODOS_NOMINAL[c]), 1.12, s=40, color=NEGRO if c < 5 else GRIS, zorder=3)
        a.text(np.radians(sm.ELECTRODOS_NOMINAL[c]), 1.32, ia.CANALES[c], ha="center", va="center", fontsize=7)
    a.set_ylim(0, 1.42)
    a.set_yticklabels([])
    a.set_xticks(np.radians([0, 90, 180, 270]), ["radial", "dorso", "cubital", "palma"], fontsize=7)
    a.grid(alpha=0.3)
    guardar(fig, "corte")


def dos_vs_diez():
    r = R["C1"]["resultado"]
    fig, ax = plt.subplots(figsize=(ANCHO, 1.9))
    k = ("acierto", "pseudo_r2", "r2_pose")
    x = np.arange(3)
    for j, (n, col) in enumerate((("2", CLARO), ("10", NEGRO))):
        v = [r[m][n] for m in k]
        e = [r[m][f"{n}_sd"] for m in k]
        ax.bar(x + (j - 0.5) * 0.36, v, 0.36, yerr=e, color=col, edgecolor=NEGRO, lw=0.6, capsize=2,
               label=f"{n} electrodos")
    ax.set_xticks(x, ["Acierto", "Pseudo-R² (gesto)", "R² (pose)"])
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    guardar(fig, "dos_vs_diez")


def robustez():
    r = R["C4"]["resultado"]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.0))
    g = [x["giro_deg"] for x in r["giro"]]
    ax.plot(g, [x["acierto"] * 100 for x in r["giro"]], "o-", color=GRIS, ms=3.5, label="sin recalibrar")
    ax.plot([x["giro_antes_deg"] for x in r["recalibrado"]], [x["acierto"] * 100 for x in r["recalibrado"]], "s-", color=NEGRO,
            ms=3.5, label="recalibrando")
    ax.axhline(85, color=GRIS, ls=":", lw=0.8)
    ax.set_xlabel("giro de los electrodos (°)")
    ax.set_ylabel("acierto (%)")
    ax.legend(loc="lower left")
    guardar(fig, "robustez")


def falla():
    r = R["C5"]["resultado"]["por_electrodo"]
    fig, ax = plt.subplots(figsize=(ANCHO, 1.9))
    x = np.arange(10)
    ax.bar(x - 0.2, [f["campeona"] * 100 for f in r], 0.4, color=CLARO, edgecolor=NEGRO, lw=0.5, label="red actual")
    ax.bar(x + 0.2, [f["mlp_con_apagados"] * 100 for f in r], 0.4, color=NEGRO, label="entrenada con electrodos apagados")
    ax.set_xticks(x, [f["electrodo"] for f in r])
    ax.set_ylabel("acierto (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    guardar(fig, "falla")


def neat():
    r = R["C6"]["resultado"]
    fig, ax = plt.subplots(figsize=(ANCHO, 1.8))
    nombres = ["NEAT", "Logística", "MLP"]
    medias = [r["neat_media"] * 100, r["logistica_media"] * 100, r["mlp_media"] * 100]
    sd = [r["neat_sd"] * 100, np.std([b["acierto"] for b in r["logistica"]]) * 100, np.std([b["acierto"] for b in r["mlp"]]) * 100]
    ax.barh(nombres, medias, xerr=sd, color=[GRIS, CLARO, NEGRO], edgecolor=NEGRO, lw=0.6, capsize=2)
    for k, v in enumerate(medias):
        ax.text(v - 0.6, k, f"{v:.1f} %".replace(".", ","), va="center", ha="right", color="white" if k != 1 else NEGRO, fontsize=7)
    ax.set_xlim(80, 100)
    ax.set_xlabel("acierto en 8 personas nuevas (%)")
    guardar(fig, "neat")


def confusion():
    cm = np.array(R["C3"]["resultado"]["matriz"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    nombres = ["Rep.", "Abi.", "Puño", "Pinza", "Señ.", "Flex.", "Ext."]
    for i in range(7):
        for j in range(7):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i, j] * 100:.1f}".replace(".", ","), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else NEGRO)
    ax.set_xticks(range(7), nombres)
    ax.set_yticks(range(7), nombres)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    guardar(fig, "confusion")


if __name__ == "__main__":
    for f in (corte, dos_vs_diez, robustez, falla, neat, confusion):
        f()
    print("ok", sorted(p.name for p in SALIDA.glob("*.png")))
