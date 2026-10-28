"""Figuras propias del documento técnico (las de las pruebas las genera cada prueba)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pruebas"))
from comun import ACENTO, AZULES, GRIS, MOTOR, NARANJA, VERDES, figura, plt  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import ia_mano as ia  # noqa: E402


def _caja(ax, x, y, w, h, texto, color, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=color, ec="#2b3a47", lw=0.9))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, wrap=True)


def _flecha(ax, a, b, estilo="-|>", color="#2b3a47", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=estilo, mutation_scale=9, color=color, lw=1, ls=ls))


def arquitectura():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fisico = ["Persona\n10 músculos del\nantebrazo", "10 electrodos\n(F1–F5, E1–E5)\n+ referencia", "AD620 ×10\nG = 1 + 49,4k/RG",
              "Pasa-banda\n16–482 Hz ×10", "Multiplexor\nCD74HC4067", "ADC ESP32\n12 bits · 1 kHz", "40 características\n+ calibración\n+ contexto (120)",
              "Red neuronal\ngesto + confianza", "Filtro de\nconfianza y\nmayoría", "5 servos de dedos\n+ servo de muñeca"]
    w, h, y1 = 0.088, 0.17, 0.74
    xs = np.linspace(0.005, 1 - w - 0.005, len(fisico))
    ax.text(0.005, 0.95, "PRÓTESIS (FÍSICA) · ESP32", fontsize=9, fontweight="bold", color=ACENTO)
    for k, (x, t) in enumerate(zip(xs, fisico)):
        _caja(ax, x, y1, w, h, t, "#dbe7f7" if k < 6 else "#fde7cf")
        if k:
            _flecha(ax, (xs[k - 1] + w, y1 + h / 2), (x, y1 + h / 2))
    ax.text(0.005, 0.56, "GEMELO DIGITAL (PC) · Python, PyTorch en la GPU", fontsize=9, fontweight="bold", color=VERDES[3])
    gemelo = [(0.005, "Personas virtuales\n(5 niveles: ruido, 60 Hz,\nfatiga, variabilidad)"),
              (0.15, "Geometría y acople\nmúsculo → electrodo\n(se corren por sesión)"),
              (0.295, "Cadena de adquisición\nsimulada por etapas\n(AD620, filtro, ADC)"),
              (0.44, "Mismas características,\ncalibración y contexto\n(ia_mano.py)")]
    for x, t in gemelo:
        _caja(ax, x, 0.36, 0.13, 0.16, t, "#d9f2e8")
    for (xa, _), (xb, _) in zip(gemelo[:-1], gemelo[1:]):
        _flecha(ax, (xa + 0.13, 0.44), (xb, 0.44))
    apps = [(0.60, 0.40, "Competencia (PBT)\n10 redes en la GPU\n→ campeonas"),
            (0.60, 0.21, "Neuroevolución (NEAT)\n150 redes desde cero\n→ campeones"),
            (0.80, 0.40, "modelo_red_calibrado.h\nred_neat.h\n(exportación a C)"),
            (0.80, 0.21, "Pruebas técnicas\n(gemelo, señal, modelos,\nfirmware, software)")]
    for x, y, t in apps:
        _caja(ax, x, y, 0.17, 0.14, t, "#eadff7" if x < 0.7 else "#fff3c4")
    _flecha(ax, (0.57, 0.44), (0.60, 0.47))
    _flecha(ax, (0.57, 0.44), (0.60, 0.28))
    _flecha(ax, (0.77, 0.47), (0.80, 0.47))
    _flecha(ax, (0.77, 0.28), (0.80, 0.47))
    _flecha(ax, (0.885, 0.54), (xs[7] + w / 2, y1), color=NARANJA)
    ax.text(0.83, 0.62, "se copia al ESP32", fontsize=7.5, color=NARANJA)
    for k in range(4):
        _flecha(ax, (xs[k + (k > 0) + (k > 1)] + w / 2, y1), (gemelo[k][0] + 0.065, 0.52), estilo="<|-|>", color=GRIS, ls="--")
    _caja(ax, 0.005, 0.03, 0.31, 0.13, "Mano bebé: 10 músculos, sin metas impuestas,\naprende sola qué puede hacer su cuerpo\n(espejo: 19.800 manos reales de HaGRID v2)", "#f7dfe5")
    _caja(ax, 0.34, 0.03, 0.40, 0.13, "Aplicaciones de escritorio (Qt + OpenGL): competencia, red, gemelo y\nelectrodos, campeonas, datos, neuroevolución y mano bebé, en vivo.\nTodo se guarda y se retoma al reabrir.", "#eef1f4")
    ax.text(0.77, 0.08, "— — equivalencia física ↔ gemelo", fontsize=7.5, color=GRIS)
    figura("T", "arquitectura", fig)


def red_competencia():
    c = sorted((MOTOR / "campeonas").glob("c*"))[-1]
    f = json.loads((c / "ficha.json").read_text(encoding="utf-8"))
    g = f["genoma"]
    capas = [120, g["h1"]] + ([g["h2"]] if g["h2"] else [])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    xs = np.linspace(0.06, 0.78, len(capas) + 1)
    textos = [f"Entrada\n120 = 40 × 3 ventanas\n(normalización)"] + [f"Oculta {k + 1}\n{n} ReLU" for k, n in enumerate(capas[1:])]
    for k, (x, n) in enumerate(zip(xs[:-1], capas)):
        m = min(n, 16)
        ys = np.linspace(0.12, 0.88, m)
        ax.scatter([x] * m, ys, s=26, color=AZULES[2] if k else GRIS, zorder=3)
        if k:
            for y0 in np.linspace(0.12, 0.88, min(capas[k - 1], 16)):
                for y1 in ys[::3]:
                    ax.plot([xs[k - 1], x], [y0, y1], color="#c9d6e3", lw=0.3, zorder=1)
        ax.text(x, 0.0, textos[k], ha="center", va="top", fontsize=8)
    xg = xs[-1]
    for y, n, col, t in ((0.72, 7, ACENTO, "Gesto (7)\nsoftmax"), (0.25, 6, VERDES[2], "Pose (6 GDL)\nlineal")):
        ys = np.linspace(y - 0.18, y + 0.18, n)
        ax.scatter([xg] * n, ys, s=30, color=col, zorder=3)
        for y0 in np.linspace(0.12, 0.88, min(capas[-1], 16)):
            for y1 in ys[::2]:
                ax.plot([xs[-2], xg], [y0, y1], color="#c9d6e3", lw=0.3, zorder=1)
        ax.text(xg + 0.04, y, t, va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1)
    ax.set_title(f"Red campeona #{f['campeona']} (generación {f['generacion']}): {f.get('parametros', '')} parámetros · "
                 f"lr {g['lr']:.1e} · olvido {g['olvido']} · L2 {g['l2']} · peso de la pose {g['peso_pose']}", fontsize=9)
    figura("T", "red_competencia", fig)
    return f


def red_neat():
    c = sorted((MOTOR / "neuro" / "campeones").glob("c*"))[-1]
    f = json.loads((c / "ficha.json").read_text(encoding="utf-8"))
    from gemelo.neuro import neat
    conex = [(a, b, w) for _, a, b, w, act in f["genoma"] if act]
    usados = sorted({a for a, _, _ in conex if a <= neat.SESGO})
    ocultas = sorted({n for a, b, _ in conex for n in (a, b) if n >= neat.PRIMER_OCULTO})
    prof = {n: 0 for n in usados}
    for _ in range(len(ocultas) + 2):
        for a, b, _ in conex:
            if b >= neat.PRIMER_OCULTO:
                prof[b] = max(prof.get(b, 1), prof.get(a, 0) + 1)
    maxp = max([prof.get(n, 1) for n in ocultas], default=0) + 1
    pos = {}
    for k, n in enumerate(usados):
        pos[n] = (0, 1 - (k + 0.5) / len(usados))
    cols = {}
    for n in ocultas:
        cols.setdefault(prof.get(n, 1), []).append(n)
    for d, ns in cols.items():
        for k, n in enumerate(ns):
            pos[n] = (d / maxp, 1 - (k + 0.5) / len(ns))
    for k, s in enumerate(neat.SALIDAS):
        pos[s] = (1, 1 - (k + 0.5) / 7)
    fig, ax = plt.subplots(figsize=(9, 5))
    for a, b, w in conex:
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=ACENTO if w > 0 else NARANJA,
                    lw=0.4 + 1.6 * min(abs(w), 3) / 3, alpha=0.6)
    for n in usados:
        ax.scatter(*pos[n], s=18, color=GRIS if n == neat.SESGO else (VERDES[2] if neat.NOMBRES_ENTRADAS[n].startswith("E") else AZULES[2]), zorder=3)
        ax.text(pos[n][0] - 0.015, pos[n][1], "sesgo" if n == neat.SESGO else neat.NOMBRES_ENTRADAS[n], ha="right", va="center", fontsize=6)
    for n in ocultas:
        ax.scatter(*pos[n], s=40, color="white", edgecolor="#333", zorder=3)
    for k, s in enumerate(neat.SALIDAS):
        ax.scatter(*pos[s], s=60, color=ACENTO, zorder=3)
        ax.text(1.02, pos[s][1], ["Reposo", "Abierta", "Puño", "Pinza", "Señalar", "Flexión", "Extensión"][k], va="center", fontsize=8)
    ax.set_xlim(-0.25, 1.15)
    ax.axis("off")
    m = f["metricas"]
    ax.set_title(f"Campeón NEAT #{f['campeon']} (generación {f['generacion']}): {len(ocultas)} neuronas ocultas, {len(conex)} conexiones, "
                 f"usa {len(usados) - 1} de 120 entradas · acierto {m['acierto'] * 100:.1f} %", fontsize=9)
    figura("T", "red_neat", fig)
    return f


def tendones():
    from gemelo.libre import cuerpo
    M = np.array([cuerpo.W_FDS, cuerpo.W_FDP, cuerpo.W_EDC, cuerpo.W_EIP])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.1))
    im = ax[0].imshow(M, cmap="Blues", vmin=0, vmax=1.8)
    ax[0].set_xticks(range(4), ["índice", "medio", "anular", "meñique"])
    ax[0].set_yticks(range(4), ["FDS (flexor sup.)", "FDP (flexor prof.)", "EDC (extensor común)", "EIP (extensor índice)"])
    ax[0].grid(False)
    for i in range(4):
        for j in range(4):
            ax[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 1 else "#333")
    ax[0].set_title("Reparto de cada tendón entre los dedos")
    plt.colorbar(im, ax=ax[0], fraction=0.04)
    a = np.linspace(0, 1, 50)
    for nombre, col, f in (("solo FDS", ACENTO, lambda v: np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + np.eye(10)[0] * v),
                           ("solo EDC", VERDES[2], lambda v: np.eye(10)[5] * v),
                           ("FDS + EIP (señalar)", NARANJA, lambda v: np.eye(10)[0] * v + np.eye(10)[6] * v)):
        P = cuerpo.pose(np.array([f(v) for v in a]))
        ax[1].plot(a, P[:, 1], color=col, label=f"{nombre}: índice")
        ax[1].plot(a, P[:, 3], color=col, ls="--", label=f"{nombre}: anular")
    ax[1].set_xlabel("activación del músculo (0 a 1)")
    ax[1].set_ylabel("flexión (0 estirado · 1 cerrado)")
    ax[1].set_title("Acoples: un músculo mueve varios dedos")
    ax[1].legend(fontsize=6.5)
    figura("T", "tendones", fig)


if __name__ == "__main__":
    arquitectura()
    red_competencia()
    red_neat()
    tendones()
    print("ok")
