"""
Ventana principal del gemelo de la mano (aplicación de escritorio)
==================================================================

Pestañas:
  Competencia de manos   las 10 manos InMoov en 3D, ranking, curva y lo que pasa
  Red neuronal           la red de la mano elegida: pesos y activaciones en vivo
  Gemelo y electrodos    corte del antebrazo, electrodos que se mueven, amplificador
  Campeonas              fotos de los 7 gestos de cada campeona
  Datos guardados        lo que el gemelo acumuló en disco

Todas reproducen la misma demostración a velocidad real; cada generación
nueva trae una persona nueva y las manos más entrenadas.
"""

import math
import queue
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .manos3d import EscenaManos

NOMBRES = ["Reposo", "Abierta", "Puño", "Pinza", "Señalar", "Flexión muñeca", "Extensión muñeca"]
DOF = ["pulgar", "índice", "medio", "anular", "meñique", "muñeca"]
C = {"bg": "#161c22", "panel": "#1d252c", "linea": "#2b363f", "fg": "#e6edf1", "muted": "#93a2ac",
     "acento": "#5b9bf0", "canal2": "#3fbf9a", "ok": "#3fbf7f", "aviso": "#e0a743", "mal": "#ef6b63", "neg": "#ec8350",
     "musculo": "#e25b52"}
QSS = f"""
QWidget {{ background: {C['bg']}; color: {C['fg']}; font-family: "Segoe UI"; font-size: 13px; }}
QTabWidget::pane {{ border: 0; }}
QTabBar::tab {{ background: transparent; color: {C['muted']}; padding: 9px 16px; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {C['acento']}; border-bottom-color: {C['acento']}; }}
QFrame#panel {{ background: {C['panel']}; border: 1px solid {C['linea']}; border-radius: 10px; }}
QLabel#titulo {{ color: {C['muted']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; background: transparent; }}
QLabel#chip {{ border: 1px solid {C['linea']}; border-radius: 10px; padding: 2px 9px; font-family: Consolas; font-size: 12px; background: {C['panel']}; }}
QLabel {{ background: transparent; }}
QTableWidget {{ background: {C['panel']}; gridline-color: {C['linea']}; border: 0; font-size: 12px; }}
QHeaderView::section {{ background: {C['panel']}; color: {C['muted']}; border: 0; border-bottom: 1px solid {C['linea']}; padding: 4px; }}
QListWidget {{ background: {C['panel']}; border: 0; font-size: 12px; color: {C['muted']}; }}
QPushButton {{ background: {C['panel']}; border: 1px solid {C['linea']}; border-radius: 7px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {C['acento']}; }}
QPushButton#primario {{ background: {C['acento']}; color: white; border-color: {C['acento']}; }}
QComboBox, QCheckBox {{ background: transparent; }}
QComboBox {{ border: 1px solid {C['linea']}; border-radius: 6px; padding: 3px 8px; background: {C['panel']}; }}
QProgressBar {{ background: {C['bg']}; border: 0; border-radius: 4px; height: 12px; text-align: right; font-size: 11px; }}
QProgressBar::chunk {{ background: {C['muted']}; border-radius: 4px; }}
QScrollArea {{ border: 0; }}
"""
pg.setConfigOptions(antialias=True, background=C["panel"], foreground=C["muted"])


def panel(titulo=None):
    f = QtWidgets.QFrame()
    f.setObjectName("panel")
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 12)
    if titulo:
        t = QtWidgets.QLabel(titulo.upper())
        t.setObjectName("titulo")
        t.setWordWrap(True)                                     # los títulos largos no ensanchan la ventana
        lay.addWidget(t)
    return f, lay


def qcolor(hexa, alfa=255):
    c = QtGui.QColor(hexa)
    c.setAlpha(int(alfa))
    return c


def pct(v):
    return f"{v * 100:.1f} %".replace(".", ",")


# ------------------------------------------------------------------
# Reloj: la demostración se reproduce a velocidad real y cambia al terminar
# ------------------------------------------------------------------
class Reloj:
    def __init__(self):
        self.demo, self.pendiente, self.t0, self.cambio = None, None, time.perf_counter(), False

    def recibir(self, demo):
        if self.demo is None:
            self.demo, self.t0, self.cambio = demo, time.perf_counter(), True
        else:
            self.pendiente = demo

    def ahora(self):
        if self.demo is None:
            return None
        dt = self.demo["vista"]["dt"]
        T = len(self.demo["gesto"])
        dur = T * dt
        t = time.perf_counter() - self.t0
        if t > dur + 0.6:
            if self.pendiente is not None:
                self.demo, self.pendiente, self.cambio = self.pendiente, None, True
            self.t0, t = time.perf_counter(), 0.0
        x = min(t / dt, T - 1)
        i = int(x)
        return i, x - i, min(t, dur), dur


# ------------------------------------------------------------------
# Competencia
# ------------------------------------------------------------------
class VistaCompetencia(QtWidgets.QWidget):
    elegir_mano = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        f, l = panel("Las 10 manos compitiendo · sólida = lo que hace · azul transparente = objetivo · arrastra para girar")
        self.objetivo = QtWidgets.QLabel("Esperando la primera generación…")
        self.objetivo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.objetivo.setStyleSheet("font-size: 14px;")
        l.addWidget(self.objetivo)
        self.escena = EscenaManos()
        self.escena.setMinimumHeight(380)
        l.addWidget(self.escena, 1)
        lay.addWidget(f, 0, 0)
        f, l = panel("Puntaje por generación · mejor (azul) y promedio (gris)")
        self.curva = pg.PlotWidget()
        self.curva.setMinimumHeight(130)
        self.curva.setMaximumHeight(210)
        self.curva.showGrid(y=True, alpha=0.15)
        self.l_mejor = self.curva.plot(pen=pg.mkPen(C["acento"], width=2))
        self.l_media = self.curva.plot(pen=pg.mkPen(C["muted"], width=1.5))
        self.curva.setLabel("bottom", "generación")
        l.addWidget(self.curva)
        lay.addWidget(f, 1, 0)
        f, l = panel("Ranking en el examen · personas que ninguna mano vio")
        self.tabla = QtWidgets.QTableWidget(10, 7)
        self.tabla.setHorizontalHeaderLabels(["#", "Mano", "Puntaje", "Acierto", "pseudo-R²", "R² pose", "Capas"])
        self.tabla.verticalHeader().hide()
        self.tabla.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabla.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tabla.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tabla.cellDoubleClicked.connect(lambda r, _c: self.elegir_mano.emit(int(self.tabla.item(r, 1).data(Qt.UserRole))))
        l.addWidget(self.tabla, 3)
        t = QtWidgets.QLabel("LO QUE PASA EN LA COMPETENCIA")
        t.setObjectName("titulo")
        l.addWidget(t)
        self.eventos = QtWidgets.QListWidget()
        self.eventos.setWordWrap(True)
        l.addWidget(self.eventos, 2)
        nota = QtWidgets.QLabel("Doble clic en una mano para ver su red neuronal.")
        nota.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        l.addWidget(nota)
        lay.addWidget(f, 0, 1, 2, 1)
        lay.setColumnStretch(0, 3)
        lay.setColumnStretch(1, 1)
        lay.setRowStretch(0, 4)
        lay.setRowStretch(1, 1)
        self.manos = []

    def estado(self, e):
        self.manos = e.get("manos", self.manos)
        for r, m in enumerate(sorted(self.manos, key=lambda m: m["rank"])):
            g = m["genoma"]
            vals = [str(m["rank"]), f"Mano {m['mano'] + 1}", f"{m['puntaje'] * 100:.1f}", pct(m["acierto"]),
                    f"{m['pseudo_r2']:.3f}", f"{m['r2_pose']:.3f}", f"{g['h1']}·{g['h2']}" if g["h2"] else str(g["h1"])]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                if c == 1:
                    it.setData(Qt.UserRole, m["mano"])
                if m["rank"] == 1:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(C["acento"])))
                self.tabla.setItem(r, c, it)
        self.eventos.clear()
        for ev in reversed(e.get("eventos", [])):
            it = QtWidgets.QListWidgetItem(f"gen {ev['gen']} · {ev['texto']}")
            if ev["texto"].startswith("★"):
                it.setForeground(QtGui.QBrush(QtGui.QColor(C["acento"])))
            self.eventos.addItem(it)
        h = e.get("historial")
        if h is not None and len(h):
            self.l_mejor.setData(h[:, 0], h[:, 1] * 100)
            self.l_media.setData(h[:, 0], h[:, 2] * 100)

    def cuadro(self, reloj):
        r = reloj.ahora()
        if r is None or not self.manos:
            return
        i, a, t, dur = r
        d = reloj.demo
        j = min(i + 1, len(d["gesto"]) - 1)
        poses = d["pose"][:, i] + (d["pose"][:, j] - d["pose"][:, i]) * a
        obj = d["objetivo"][i] + (d["objetivo"][j] - d["objetivo"][i]) * a
        info = {m["mano"]: m for m in self.manos}
        lider = next((m["mano"] for m in self.manos if m["rank"] == 1), -1)
        textos = []
        for k in range(len(poses)):
            pred = int(np.argmax(d["probs"][k, i]))
            ok = pred == d["gesto"][i]
            m = info.get(k, {})
            textos.append((f"#{m.get('rank', '–')} · Mano {k + 1} · {pct(m.get('acierto', 0))}", f"{'✓' if ok else '✗'} {NOMBRES[pred]}"))
        self.escena.actualizar(poses, obj, textos, lider)
        self.objetivo.setText(f"Objetivo: <b>{NOMBRES[d['gesto'][i]]}</b> &nbsp;·&nbsp; {t:.1f} / {dur:.1f} s "
                              f"&nbsp;·&nbsp; persona nueva de la generación {d['gen']}")


# ------------------------------------------------------------------
# Red neuronal
# ------------------------------------------------------------------
def adelante(p, x):
    a = (x - p["media"]) / p["escala"]
    acts = [a]
    ocultas = [c for c in p["capas"] if c["nombre"].startswith("oculta")]
    for c in ocultas:
        a = np.maximum(0, a @ c["W"] + c["b"])
        acts.append(a)
    g = next(c for c in p["capas"] if c["nombre"] == "gesto")
    po = next(c for c in p["capas"] if c["nombre"] == "pose")
    lg = a @ g["W"] + g["b"]
    e = np.exp(lg - lg.max())
    acts.append(np.r_[e / e.sum(), a @ po["W"] + po["b"]])
    return acts, ocultas, g, po


class RedWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(420)
        self.pesos, self.x, self.aporte = None, None, True

    def fijar(self, pesos, x):
        self.pesos, self.x = pesos, x
        self.update()

    def paintEvent(self, _):
        if self.pesos is None or self.x is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        acts, ocultas, g, po = adelante(self.pesos, self.x)
        tam = [24] + [len(c["b"]) for c in ocultas] + [13]
        nC = len(tam)
        xs = [150 + (W - 330) * c / (nC - 1) for c in range(nC)]

        def ys(n, c):
            if c == nC - 1:
                return [36 + (H - 60) * (0.55 * (k + 0.5) / 7 if k < 7 else 0.6 + 0.4 * (k - 6.5) / 6) for k in range(n)]
            return [30 + (H - 50) * (k + 0.5) / n for k in range(n)]

        Y = [ys(n, c) for c, n in enumerate(tam)]
        Ws = [c["W"] for c in ocultas] + [np.c_[g["W"], po["W"]]]
        for c, Wc in enumerate(Ws):
            a = acts[c]
            V = Wc * a[:, None] if self.aporte else Wc
            tope = max(np.abs(V).max(), 1e-6)
            for i0, j0 in zip(*np.nonzero(np.abs(V) / tope >= 0.05)):
                v = V[i0, j0]
                t = abs(v) / tope
                pen = QtGui.QPen(qcolor(C["acento"] if v >= 0 else C["neg"], 20 + 210 * t), 0.4 + 2.4 * t)
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(xs[c], Y[c][i0]), QtCore.QPointF(xs[c + 1], Y[c + 1][j0]))
        for c, a in enumerate(acts):
            tope = 1 if c in (0, nC - 1) else max(np.abs(a).max(), 1e-6)
            r = 5 if c == 0 else 7 if c == nC - 1 else max(2.5, min(6, 140 / tam[c]))
            for k, v in enumerate(a):
                if c == 0:
                    t, col = min(1, abs(v) / 2.5), C["acento"] if v >= 0 else C["neg"]
                elif c == nC - 1:
                    t, col = (v if k < 7 else min(1, abs(v))), C["acento"]
                else:
                    t, col = v / tope, C["acento"]
                p.setPen(QtGui.QPen(qcolor(C["linea"]), 1))
                p.setBrush(QtGui.QColor(C["panel"]))
                p.drawEllipse(QtCore.QPointF(xs[c], Y[c][k]), r, r)
                p.setBrush(qcolor(col, 40 + 215 * max(0, min(1, t))))
                p.drawEllipse(QtCore.QPointF(xs[c], Y[c][k]), r, r)
        p.setPen(QtGui.QColor(C["muted"]))
        p.setFont(QtGui.QFont("Consolas", 8))
        for k, n in enumerate(self.pesos["entradas"]):
            p.drawText(QtCore.QRectF(4, Y[0][k] - 8, xs[0] - 14, 16), Qt.AlignRight | Qt.AlignVCenter, n)
        sal = acts[-1]
        gana = int(np.argmax(sal[:7]))
        for k in range(7):
            f = QtGui.QFont("Segoe UI", 9)
            f.setBold(k == gana)
            p.setFont(f)
            p.setPen(QtGui.QColor(C["acento"] if k == gana else C["fg"]))
            p.drawText(QtCore.QPointF(xs[-1] + 14, Y[-1][k] + 4), f"{NOMBRES[k]} {sal[k] * 100:.0f} %")
        p.setPen(QtGui.QColor(C["muted"]))
        p.setFont(QtGui.QFont("Segoe UI", 9))
        for k in range(6):
            p.drawText(QtCore.QPointF(xs[-1] + 14, Y[-1][7 + k] + 4), f"{DOF[k]} {sal[7 + k]:.2f}")
        f = QtGui.QFont("Segoe UI", 9)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QtGui.QColor(C["fg"]))
        titulos = ["entrada · 8 × 3 ventanas"] + [f"oculta {k + 1} · {len(c['b'])} ReLU" for k, c in enumerate(ocultas)] + ["gesto (7) · pose (6)"]
        for c, t in enumerate(titulos):
            p.drawText(QtCore.QRectF(xs[c] - 90, 2, 180, 18), Qt.AlignCenter, t)
        p.end()


class VistaRed(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        f, l = panel()
        fila = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("RED DE LA MANO")
        t.setObjectName("titulo")
        fila.addWidget(t)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems([f"Mano {k + 1}" for k in range(10)])
        self.combo.activated.connect(lambda _: self.seguir.setChecked(False))
        fila.addWidget(self.combo)
        self.seguir = QtWidgets.QCheckBox("seguir a la líder")
        self.seguir.setChecked(True)
        fila.addWidget(self.seguir)
        self.aporte = QtWidgets.QCheckBox("líneas = peso × activación")
        self.aporte.setChecked(True)
        fila.addWidget(self.aporte)
        fila.addStretch(1)
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet(f"color: {C['muted']};")
        fila.addWidget(self.info)
        l.addLayout(fila)
        self.red = RedWidget()
        l.addWidget(self.red, 1)
        lay.addWidget(f, 3)
        f, l = panel("Genoma y resultados")
        self.ficha = QtWidgets.QFormLayout()
        self.ficha_lbl = {}
        for k in ("capas", "tasa de aprendizaje", "olvido (dropout)", "L2", "peso de la pose", "ruido de aumento", "linaje",
                  "puntaje", "acierto", "pseudo-R² gesto", "R² pose", "pasos entrenados"):
            v = QtWidgets.QLabel("—")
            v.setStyleSheet("font-family: Consolas;")
            self.ficha.addRow(QtWidgets.QLabel(k), v)
            self.ficha_lbl[k] = v
        l.addLayout(self.ficha)
        t = QtWidgets.QLabel("ACIERTO POR GESTO EN EL EXAMEN")
        t.setObjectName("titulo")
        l.addWidget(t)
        self.barras = []
        for n in NOMBRES:
            fila = QtWidgets.QHBoxLayout()
            fila.addWidget(QtWidgets.QLabel(n), 2)
            b = QtWidgets.QProgressBar()
            b.setRange(0, 1000)
            b.setFormat("%p %")
            fila.addWidget(b, 3)
            l.addLayout(fila)
            self.barras.append(b)
        nota = QtWidgets.QLabel("Azul: aporte positivo · naranja: negativo. Los pesos cambian en cada generación; "
                                "las activaciones siguen la señal de la persona de la demostración, a velocidad real.")
        nota.setWordWrap(True)
        nota.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        l.addWidget(nota)
        l.addStretch(1)
        lay.addWidget(f, 1)
        self.estado_ = {}
        self._i = -1

    def mano(self):
        if self.seguir.isChecked():
            k = next((m["mano"] for m in self.estado_.get("manos", []) if m["rank"] == 1), 0)
            self.combo.setCurrentIndex(k)
        return self.combo.currentIndex()

    def estado(self, e):
        self.estado_ = e
        self._i = -1
        k = self.mano()
        m = next((m for m in e.get("manos", []) if m["mano"] == k), None)
        if not m:
            return
        g = m["genoma"]
        vals = {"capas": f"24 → {g['h1']}" + (f" → {g['h2']}" if g["h2"] else "") + " → 7 + 6", "tasa de aprendizaje": f"{g['lr']:.2e}",
                "olvido (dropout)": g["olvido"], "L2": g["l2"], "peso de la pose": g["peso_pose"], "ruido de aumento": g["ruido"],
                "linaje": m["linaje"], "puntaje": f"{m['puntaje'] * 100:.1f}", "acierto": pct(m["acierto"]),
                "pseudo-R² gesto": f"{m['pseudo_r2']:.3f}", "R² pose": f"{m['r2_pose']:.3f}", "pasos entrenados": f"{m['pasos']:,}"}
        for k2, v in vals.items():
            self.ficha_lbl[k2].setText(str(v))
        for b, r in zip(self.barras, m.get("recall", [])):
            b.setValue(int(r * 1000))
            b.setFormat(f"{r * 100:.0f} %")
            b.setStyleSheet(f"QProgressBar::chunk {{ background: {C['acento'] if r >= 0.85 else C['muted']}; border-radius: 4px; }}")
        self.info.setText(f"Mano {k + 1} · puesto {m['rank']} de 10 · {m['parametros']:,} parámetros")

    def cuadro(self, reloj):
        r = reloj.ahora()
        if r is None or not self.estado_.get("pesos"):
            return
        i = r[0]
        k = self.mano()
        if i == self._i and k == getattr(self, "_k", -1) and self.red.aporte == self.aporte.isChecked():
            return
        self._i, self._k = i, k
        self.red.aporte = self.aporte.isChecked()
        self.red.fijar(self.estado_["pesos"][k], reloj.demo["X"][i])


# ------------------------------------------------------------------
# Gemelo y electrodos
# ------------------------------------------------------------------
class BrazoWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 360)
        self.vista, self.i = None, 0

    def fijar(self, vista, i):
        self.vista, self.i = vista, i
        self.update()

    def paintEvent(self, _):
        v = self.vista
        if not v:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        S = min(self.width(), self.height())
        cx, cy, R = self.width() / 2, self.height() / 2, S * 0.34
        ge = v["geometria"]
        act = v["activacion"][min(self.i, len(v["activacion"]) - 1)]

        def P(ang, r):
            return QtCore.QPointF(cx + math.cos(math.radians(ang)) * r, cy - math.sin(math.radians(ang)) * r)

        p.setPen(QtGui.QPen(qcolor(C["linea"]), 2))
        p.setBrush(QtGui.QColor("#26313a"))
        p.drawEllipse(QtCore.QPointF(cx, cy), R, R)
        p.setBrush(QtGui.QColor(C["panel"]))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(cx - R * 0.18, cy + R * 0.02), R * 0.13, R * 0.1)
        p.drawEllipse(QtCore.QPointF(cx + R * 0.2, cy - R * 0.02), R * 0.11, R * 0.09)
        p.setPen(QtGui.QColor(C["muted"]))
        p.setFont(QtGui.QFont("Segoe UI", 9))
        p.drawText(QtCore.QRectF(cx - R * 0.3, cy - 8, R * 0.24, 16), Qt.AlignCenter, "radio")
        p.drawText(QtCore.QRectF(cx + R * 0.1, cy - 10, R * 0.22, 16), Qt.AlignCenter, "cúbito")
        p.drawText(QtCore.QRectF(4, cy - R, 120, 16), Qt.AlignLeft, "↑ lado del dorso")
        p.drawText(QtCore.QRectF(4, cy + R - 14, 120, 16), Qt.AlignLeft, "↓ lado de la palma")
        colores = [C["acento"], C["canal2"]]
        ang_m = [a + d for a, d in zip(ge["angulos"], ge["dang_persona"])]
        for c, fila in enumerate(ge["acople"]):
            for k, w in enumerate(fila):
                if w < 0.08:
                    continue
                p.setPen(QtGui.QPen(qcolor(colores[c], 255 * 0.55 * w * (0.3 + act[k])), 1 + 2.5 * w))
                p.drawLine(P(ge["electrodos"][c], R + 6), P(ang_m[k], R * ge["profundidad"][k]))
        f = QtGui.QFont("Segoe UI", 8)
        f.setBold(True)
        for k, n in enumerate(v["musculos"]):
            q = P(ang_m[k], R * ge["profundidad"][k])
            r = R * (0.08 + 0.05 * ge["profundidad"][k])
            p.setPen(QtGui.QPen(qcolor(C["linea"]), 1))
            p.setBrush(QtGui.QColor(C["panel"]))
            p.drawEllipse(q, r, r)
            p.setBrush(qcolor(C["musculo"], 30 + 225 * act[k]))
            p.drawEllipse(q, r, r)
            p.setFont(f)
            p.setPen(QtGui.QColor(C["fg"]))
            p.drawText(QtCore.QRectF(q.x() - r, q.y() - 8, 2 * r, 16), Qt.AlignCenter, n)
        for c, ang in enumerate(ge["electrodos"]):
            nom, act_p = P(ge["electrodos_nominal"][c], R + 12), P(ang, R + 12)
            pen = QtGui.QPen(QtGui.QColor(colores[c]), 1.6, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(nom, 8, 8)
            p.setPen(QtGui.QPen(QtGui.QColor(colores[c]), 1.6))
            p.drawLine(nom, act_p)
            p.setBrush(QtGui.QColor(colores[c]))
            p.drawEllipse(act_p, 8, 8)
            et = P(ang, R + 40)
            p.setFont(f)
            p.drawText(QtCore.QRectF(et.x() - 50, et.y() - 8, 100, 16), Qt.AlignCenter, f"canal {c + 1}")
        p.end()


class Osciloscopio(pg.PlotWidget):
    def __init__(self, unidad, rieles=None, rango=None):
        super().__init__()
        self.setMinimumHeight(92)
        self.showGrid(y=True, alpha=0.12)
        self.setLabel("left", unidad)
        self.c1 = self.plot(pen=pg.mkPen(C["acento"], width=1))
        self.c2 = self.plot(pen=pg.mkPen(C["canal2"], width=1))
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(C["mal"], width=1.5))
        self.addItem(self.cursor)
        self.rieles = []
        self.rango = rango
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()

    def datos(self, serie, fs, rieles=None):
        s = np.asarray(serie, float)
        t = np.arange(len(s)) / fs
        self.c1.setData(t, s[:, 0])
        self.c2.setData(t, s[:, 1])
        for r in self.rieles:
            self.removeItem(r)
        self.rieles = []
        for v in rieles or []:
            ln = pg.InfiniteLine(pos=v, angle=0, pen=pg.mkPen(C["mal"], width=1, style=Qt.DashLine))
            self.addItem(ln)
            self.rieles.append(ln)
        if self.rango:
            self.setYRange(*self.rango, padding=0.02)
        else:
            self.enableAutoRange(axis="y")


class VistaGemelo(QtWidgets.QWidget):
    amplificador = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        col = QtWidgets.QVBoxLayout()
        f, l = panel("Corte del antebrazo · músculos y electrodos")
        self.brazo = BrazoWidget()
        l.addWidget(self.brazo, 1)
        nota = QtWidgets.QLabel("Rojo: músculo activo en este instante. Círculo punteado: dónde deberían ir los "
                                "electrodos; círculo lleno: dónde quedaron en esta sesión. Las líneas muestran cuánto "
                                "capta cada electrodo de cada músculo.")
        nota.setWordWrap(True)
        nota.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        l.addWidget(nota)
        col.addWidget(f, 3)
        f, l = panel("Persona y sesión de la demostración")
        self.ficha = QtWidgets.QFormLayout()
        self.ficha_lbl = {}
        for k in ("nivel de dificultad", "diferencias entre personas", "ruido de electrodos", "interferencia 60 Hz", "fatiga",
                  "electrodos movidos", "desfase continuo", "AD620", "saturación"):
            v = QtWidgets.QLabel("—")
            v.setWordWrap(True)
            v.setStyleSheet("font-family: Consolas; font-size: 12px;")
            self.ficha.addRow(QtWidgets.QLabel(k), v)
            self.ficha_lbl[k] = v
        l.addLayout(self.ficha)
        col.addWidget(f, 2)
        lay.addLayout(col, 2)
        f, l = panel("De los electrodos al ESP32, etapa por etapa · canal 1 (azul) y canal 2 (verde)")
        self.osc = {}
        for clave, nombre, unidad, rango in (("electrodo_uv", "Electrodos", "µV", None), ("ad620_v", "Amplificador AD620", "V", None),
                                              ("filtro_v", "Filtro 16–482 Hz × 10", "V", None), ("adc", "ADC del ESP32 (12 bits)", "cuentas", (0, 4095))):
            fila = QtWidgets.QHBoxLayout()
            etq = QtWidgets.QLabel(f"<b>{nombre}</b>")
            etq.setFixedWidth(190)
            etq.setWordWrap(True)
            fila.addWidget(etq)
            o = Osciloscopio(unidad, rango=rango)
            fila.addWidget(o, 1)
            l.addLayout(fila, 1)
            self.osc[clave] = (o, etq, nombre)
        fila = QtWidgets.QHBoxLayout()
        etq = QtWidgets.QLabel("<b>Calibradas</b><br><span style='color:#93a2ac'>log(1 + c) − calibración de 20 s</span>")
        etq.setFixedWidth(190)
        etq.setWordWrap(True)
        fila.addWidget(etq)
        self.carac = pg.PlotWidget()
        self.carac.setMinimumHeight(92)
        self.carac.setYRange(-3, 3)
        self.carac.getAxis("bottom").setTicks([[(k, f"c{1 + k // 4} {n}") for k, n in enumerate(["MAV", "RMS", "ZC", "WL"] * 2)]])
        self.barras = pg.BarGraphItem(x=np.arange(8), height=np.zeros(8), width=0.7,
                                      brushes=[pg.mkBrush(C["acento"])] * 4 + [pg.mkBrush(C["canal2"])] * 4)
        self.carac.addItem(self.barras)
        self.carac.setMouseEnabled(x=False, y=False)
        self.carac.hideButtons()
        fila.addWidget(self.carac, 1)
        l.addLayout(fila, 1)
        t = QtWidgets.QLabel("AMPLIFICADOR DEL GEMELO")
        t.setObjectName("titulo")
        l.addWidget(t)
        amp = QtWidgets.QGridLayout()
        self.rg = QtWidgets.QSlider(Qt.Horizontal)
        self.rg.setRange(0, 1000)
        self.rg_val = QtWidgets.QLabel()
        self.desfase = QtWidgets.QSlider(Qt.Horizontal)
        self.desfase.setRange(0, 80)
        self.desfase_val = QtWidgets.QLabel()
        amp.addWidget(QtWidgets.QLabel("Resistencia de ganancia RG"), 0, 0)
        amp.addWidget(self.rg, 0, 1)
        amp.addWidget(self.rg_val, 0, 2)
        amp.addWidget(QtWidgets.QLabel("Desfase continuo de los electrodos"), 1, 0)
        amp.addWidget(self.desfase, 1, 1)
        amp.addWidget(self.desfase_val, 1, 2)
        b = QtWidgets.QPushButton("Aplicar al gemelo")
        b.setObjectName("primario")
        b.clicked.connect(self._aplicar)
        amp.addWidget(b, 0, 3, 2, 1)
        l.addLayout(amp)
        self.nota_amp = QtWidgets.QLabel("Afecta a los datos nuevos y a la próxima demostración. Si el AD620 se satura, "
                                         "esa sesión no sirve: la calibración no lo puede arreglar.")
        self.nota_amp.setWordWrap(True)
        self.nota_amp.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        l.addWidget(self.nota_amp)
        lay.addWidget(f, 3)
        self.rg.valueChanged.connect(self._mostrar_amp)
        self.desfase.valueChanged.connect(self._mostrar_amp)
        self.tocado = False
        self.vista = None

    @staticmethod
    def _rg(s):
        return 100 * 100 ** (s / 1000)

    def _mostrar_amp(self, *_):
        self.tocado = True
        rg = self._rg(self.rg.value())
        self.rg_val.setText(f"{rg:,.0f} Ω → G = {1 + 49400 / rg:,.0f}")
        self.desfase_val.setText(f"±{self.desfase.value()} mV")

    def _aplicar(self):
        self.amplificador.emit({"rg_ohm": self._rg(self.rg.value()), "desfase_mv_sd": float(self.desfase.value())})
        self.tocado = False
        self.nota_amp.setText("Aplicado: los datos nuevos y la próxima demostración usan este amplificador.")

    def estado(self, e):
        a = e.get("amplificador")
        if a and not self.tocado:
            for w in (self.rg, self.desfase):
                w.blockSignals(True)
            self.rg.setValue(int(1000 * math.log(a["rg_ohm"] / 100) / math.log(100)))
            self.desfase.setValue(int(a["desfase_mv_sd"]))
            for w in (self.rg, self.desfase):
                w.blockSignals(False)
            self.rg_val.setText(f"{a['rg_ohm']:,.0f} Ω → G = {a['g_ad620']:,.0f}")
            self.desfase_val.setText(f"±{a['desfase_mv_sd']:.0f} mV")

    def nueva_demo(self, demo):
        v = self.vista = demo["vista"]
        fs = v["fs_senal"]
        a = v["amplificador"]
        for clave, (o, etq, nombre) in self.osc.items():
            o.datos(v["senal"][clave], fs, [a["riel_ad620_v"], -a["riel_ad620_v"]] if clave == "ad620_v" else None)
        self.osc["ad620_v"][1].setText(f"<b>Amplificador AD620</b><br><span style='color:#93a2ac'>G = {a['g_ad620']:.0f} "
                                       f"(1 + 49,4 kΩ / {a['rg_ohm']:.0f} Ω)</span>" + ("<br><span style='color:#ef6b63'>¡saturado!</span>" if a["saturo"] else ""))
        p, ge = v["persona"], v["geometria"]
        corr = [((x - n + 540) % 360) - 180 for x, n in zip(ge["electrodos"], ge["electrodos_nominal"])]
        vals = {"nivel de dificultad": f"{p['nivel']} de 5", "diferencias entre personas": f"{p['variabilidad']:.2f}",
                "ruido de electrodos": f"× {p['ruido']:.2f}", "interferencia 60 Hz": f"{p['interferencia_60hz']:.1f} µV",
                "fatiga": f"{p['fatiga']:.2f}",
                "electrodos movidos": " · ".join(f"canal {c + 1}: {d:+.0f}°, {ge['axial_cm'][c]:+.1f} cm" for c, d in enumerate(corr)),
                "desfase continuo": " · ".join(f"canal {c + 1}: {d:+.1f} mV" for c, d in enumerate(a["desfase_mv"])),
                "AD620": f"G = {a['g_ad620']:.0f} · total {a['g_total']:.0f}", "saturación": "SÍ: llegó al riel" if a["saturo"] else "no"}
        for k, val in vals.items():
            self.ficha_lbl[k].setText(val)

    def cuadro(self, reloj):
        r = reloj.ahora()
        if r is None or self.vista is None:
            return
        i, _, t, dur = r
        self.brazo.fijar(self.vista, i)
        for o, _, _ in self.osc.values():
            o.cursor.setValue(t)
        v = self.vista
        F = np.asarray(v["caracteristicas"][min(i, len(v["caracteristicas"]) - 1)], float)
        self.barras.setOpts(height=np.log1p(F) - np.asarray(v["calibracion"], float))


# ------------------------------------------------------------------
# Campeonas y datos
# ------------------------------------------------------------------
class VistaCampeonas(QtWidgets.QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.cont = QtWidgets.QWidget()
        self.lay = QtWidgets.QVBoxLayout(self.cont)
        self.lay.setSpacing(12)
        self.setWidget(self.cont)
        self.firma = None

    def estado(self, e):
        lista = e.get("campeonas")
        if lista is None:
            return
        firma = tuple((c["id"], len(c["fotos"])) for c in lista)
        if firma == self.firma:
            return
        self.firma = firma
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not lista:
            self.lay.addWidget(QtWidgets.QLabel("Todavía no hay campeonas: la primera aparece al terminar la primera generación."))
        for c in lista:
            f, l = panel()
            cab = QtWidgets.QLabel(f"<span style='font-size:16px'><b>Campeona #{c['id']}</b></span> &nbsp; generación {c['gen']} · "
                                   f"mano {c['mano'] + 1} · acierto <b>{pct(c['acierto'])}</b> · pseudo-R² <b>{c['pseudo_r2']:.3f}</b> · "
                                   f"R² pose <b>{c['r2_pose']:.3f}</b> · {time.strftime('%d/%m %H:%M', time.localtime(c['t']))}")
            l.addWidget(cab)
            fila = QtWidgets.QHBoxLayout()
            if c["fotos"]:
                for k, ruta in enumerate(c["fotos"]):
                    caja = QtWidgets.QVBoxLayout()
                    img = QtWidgets.QLabel()
                    img.setPixmap(QtGui.QPixmap(ruta).scaledToWidth(170, Qt.SmoothTransformation))
                    caja.addWidget(img)
                    caja.addWidget(QtWidgets.QLabel(NOMBRES[k] if k < 7 else Path(ruta).stem))
                    fila.addLayout(caja)
                fila.addStretch(1)
            else:
                fila.addWidget(QtWidgets.QLabel("Tomando las fotos de los 7 gestos…"))
            l.addLayout(fila)
            ruta = QtWidgets.QLabel(f"Guardada en {c['carpeta']} · modelo .keras y .h5, modelo_red_calibrado.h para el ESP32, ficha.json")
            ruta.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
            ruta.setTextInteractionFlags(Qt.TextSelectableByMouse)
            l.addWidget(ruta)
            self.lay.addWidget(f)
        self.lay.addStretch(1)


class VistaDatos(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        rejilla = QtWidgets.QGridLayout()
        rejilla.setSpacing(12)
        self.tiles = {}
        for k, nombre in enumerate(["Ventanas guardadas", "Personas simuladas", "Sesiones calibradas", "En disco",
                                    "En memoria de la GPU", "Lotes", "Sesiones con el AD620 saturado", "Generaciones"]):
            f, l = panel(nombre)
            v = QtWidgets.QLabel("—")
            v.setStyleSheet("font-size: 24px; font-weight: 600;")
            l.addWidget(v)
            rejilla.addWidget(f, k // 4, k % 4)
            self.tiles[nombre] = v
        lay.addLayout(rejilla)
        f, l = panel("Últimos lotes")
        self.tabla = QtWidgets.QTableWidget(0, 7)
        self.tabla.setHorizontalHeaderLabels(["Lote", "Archivo", "Ventanas", "Personas", "Sesiones", "Saturadas", "MB"])
        self.tabla.verticalHeader().hide()
        self.tabla.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tabla.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        l.addWidget(self.tabla)
        self.carpeta = QtWidgets.QLabel()
        self.carpeta.setWordWrap(True)
        self.carpeta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.carpeta.setStyleSheet(f"color: {C['muted']}; font-size: 12px;")
        l.addWidget(self.carpeta)
        lay.addWidget(f, 1)

    def estado(self, e):
        d = e.get("datos")
        if not d:
            return
        vals = {"Ventanas guardadas": f"{d['ventanas']:,}", "Personas simuladas": f"{d['personas']:,}", "Sesiones calibradas": f"{d['sesiones']:,}",
                "En disco": f"{d['gb_en_disco']:.2f} de {d['limite_gb']:.0f} GB", "En memoria de la GPU": f"{e.get('memoria', 0):,} ventanas",
                "Lotes": f"{d['lotes']:,}", "Sesiones con el AD620 saturado": f"{d['saturadas']:,}", "Generaciones": f"{e.get('gen', 0):,}"}
        for k, v in vals.items():
            self.tiles[k].setText(v.replace(",", "."))
        self.tabla.setRowCount(len(d["ultimos"]))
        for r, lt in enumerate(d["ultimos"]):
            for c, v in enumerate([lt["id"], lt["archivo"] or "(borrado por el límite de disco)", f"{lt['ventanas']:,}", lt["personas"],
                                   lt["sesiones"], lt["saturadas"], f"{lt['mb']:.1f}"]):
                self.tabla.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.carpeta.setText(f"Carpeta: {d['carpeta']}. Cada lote trae la señal cruda del ADC, las 8 características, la "
                             "calibración de cada sesión, el gesto y la pose. Al pasar el límite se borran los lotes más "
                             "viejos; las campeonas y el examen se conservan.")


# ------------------------------------------------------------------
# Ventana
# ------------------------------------------------------------------
class Ventana(QtWidgets.QMainWindow):
    def __init__(self, salida, entrada, proceso):
        super().__init__()
        self.salida, self.entrada, self.proceso = salida, entrada, proceso
        self.setWindowTitle("Gemelo de la mano — entrenamiento continuo")
        pantalla = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1600, pantalla.width() - 40), min(960, pantalla.height() - 60))
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(8)
        cab = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Gemelo de la mano")
        t.setStyleSheet("font-size: 18px; font-weight: 700;")
        cab.addWidget(t)
        self.chips = {}
        for k in ("fase", "gen", "campeona", "datos", "salud"):
            c = QtWidgets.QLabel("—")
            c.setObjectName("chip")
            cab.addWidget(c)
            self.chips[k] = c
        self.mensaje = QtWidgets.QLabel()
        self.mensaje.setStyleSheet(f"color: {C['muted']};")
        self.mensaje.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # no ensancha la ventana
        cab.addWidget(self.mensaje, 1)
        self.b_pausa = QtWidgets.QPushButton("⏸ Pausar")
        self.b_pausa.clicked.connect(self._pausa)
        cab.addWidget(self.b_pausa)
        v.addLayout(cab)
        self.tabs = QtWidgets.QTabWidget()
        self.competencia, self.vred, self.gemelo = VistaCompetencia(), VistaRed(), VistaGemelo()
        self.campeonas, self.datos = VistaCampeonas(), VistaDatos()
        for w, n in ((self.competencia, "Competencia de manos"), (self.vred, "Red neuronal"), (self.gemelo, "Gemelo y electrodos"),
                     (self.campeonas, "Campeonas"), (self.datos, "Datos guardados")):
            self.tabs.addTab(w, n)
        v.addWidget(self.tabs, 1)
        self.competencia.elegir_mano.connect(self._ver_red)
        self.gemelo.amplificador.connect(lambda d: self.entrada.put({"cmd": "amplificador", "valor": d}))
        self.reloj, self.estado, self.pausado, self.cerrando, self.error_visto = Reloj(), {}, False, False, False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tic)
        self.timer.start(33)
        self.chips["fase"].setText("arrancando el motor…")
        destino = __import__("os").environ.get("GEMELO_CAPTURAS")      # modo de prueba: recorre y fotografía las pestañas
        if destino:
            self._capturas = (Path(destino), 0)
            QtCore.QTimer.singleShot(45000, self._capturar)

    def _capturar(self):
        destino, k = self._capturas
        if k >= self.tabs.count():
            return
        self.tabs.setCurrentIndex(k)
        QtCore.QTimer.singleShot(5000, lambda: (self.grab().save(str(destino / f"pestana_{k}.png")),
                                                setattr(self, "_capturas", (destino, k + 1)), self._capturar()))

    def _ver_red(self, k):
        self.vred.seguir.setChecked(False)
        self.vred.combo.setCurrentIndex(k)
        self.vred.estado(self.estado)
        self.tabs.setCurrentWidget(self.vred)

    def _pausa(self):
        self.entrada.put({"cmd": "pausa", "valor": not self.pausado})

    def _recibir(self, m):
        tipo = m["tipo"]
        if tipo == "estado":
            if "manos" in m:
                self.estado = m
                for w in (self.competencia, self.vred, self.gemelo, self.campeonas, self.datos):
                    w.estado(m)
                self._cabecera(m)
            else:
                self.mensaje.setText(m.get("mensaje", ""))
                self._fase(m.get("fase", ""))
        elif tipo == "progreso":
            self.mensaje.setText(f"Generación {m['gen']}: entrena la mano {m['mano'] + 1} ({m['k'] + 1} de {m['total']})")
        elif tipo == "demo":
            self.reloj.recibir(m)
        elif tipo == "error" and not self.error_visto:
            self.error_visto = True
            self._fase("error")
            QtWidgets.QMessageBox.critical(self, "El motor se detuvo", m["texto"][-1500:])
        elif tipo == "detenido":
            self._fase("detenido")

    def _fase(self, f):
        self.pausado = f == "pausa"
        texto = {"entrenando": "● entrenando", "pausa": "❚❚ en pausa", "preparando": "preparando", "error": "error",
                 "deteniendo": "guardando…", "detenido": "detenido"}.get(f, f)
        color = C["ok"] if f == "entrenando" else C["mal"] if f == "error" else C["aviso"]
        self.chips["fase"].setText(texto)
        self.chips["fase"].setStyleSheet(f"color: {color}; border-color: {color};")
        self.b_pausa.setText("▶ Reanudar" if self.pausado else "⏸ Pausar")

    def _cabecera(self, e):
        self._fase(e["fase"])
        self.mensaje.setText(e.get("mensaje", ""))
        self.chips["gen"].setText(f"gen {e['gen']} · {e['segundos_gen']:.1f} s")
        c = e.get("campeona", {})
        self.chips["campeona"].setText(f"campeona #{c['id']} · {pct(c['metricas']['acierto'])}" if c.get("metricas") else "sin campeona aún")
        d = e["datos"]
        self.chips["datos"].setText(f"{d['ventanas'] / 1000:,.0f} mil ventanas · {d['gb_en_disco']:.2f} GB".replace(",", "."))
        s = e["salud"]
        gpu = e["dispositivo"].replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")
        self.chips["salud"].setText(f"{'⚡' if s['cargador'] else '🔋 batería'} · RAM {s['ram_libre_gb']:.1f} GB · {gpu}")
        self.chips["salud"].setToolTip(f"{'Cargador conectado' if s['cargador'] else 'En batería'} · RAM libre {s['ram_libre_gb']:.1f} GB · {e['dispositivo']}")
        self.chips["salud"].setStyleSheet("" if s["cargador"] and s["ram_libre_gb"] >= 2 else f"color: {C['aviso']}; border-color: {C['aviso']};")

    def _tic(self):
        for _ in range(20):
            try:
                self._recibir(self.salida.get_nowait())
            except queue.Empty:
                break
        if self.reloj.cambio:
            self.reloj.cambio = False
            self.gemelo.nueva_demo(self.reloj.demo)
        w = self.tabs.currentWidget()
        if hasattr(w, "cuadro"):
            w.cuadro(self.reloj)
        if not self.proceso.is_alive() and not self.cerrando and not self.error_visto:
            self.error_visto = True
            self._fase("error")
            QtWidgets.QMessageBox.critical(self, "El motor se detuvo", "El proceso de entrenamiento terminó inesperadamente.")

    def closeEvent(self, ev):
        if self.proceso.is_alive():
            self.cerrando = True
            self.entrada.put({"cmd": "detener"})
            self.mensaje.setText("Guardando la población antes de cerrar…")
            fin = time.time() + 40
            while self.proceso.is_alive() and time.time() < fin:
                QtWidgets.QApplication.processEvents()
                try:
                    if self.salida.get(timeout=0.2).get("tipo") == "detenido":
                        break
                except queue.Empty:
                    pass
            self.proceso.join(5)
            if self.proceso.is_alive():
                self.proceso.terminate()
        ev.accept()
