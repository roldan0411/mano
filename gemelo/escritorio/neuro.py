"""
Ventana "Neuroevolución": muchas manos aprenden desde cero a seguir a una persona
================================================================================

Como los carros que aprenden a manejar: 150 redes NEAT compiten en la misma
pista (una persona del gemelo haciendo gestos); se ven 30 que representan a
toda la población. Cada mano escucha los electrodos y mueve los dedos; si
hace el gesto equivocado medio segundo seguido, choca y se apaga. Las que
más aguantan tienen hijos; las redes crecen generación tras generación.

  ┌ la pista: lo que quiere la persona y su señal EMG ──────┬ red de la mejor ┐
  ├ 30 manos (color = especie, gris = chocó) ───────────────┤ curvas · R²     │
  │                                                         ├ especies        │
  │                                                         ├ lo que pasa     │
  └─────────────────────────────────────────────────────────┴─────────────────┘
"""

import multiprocessing as mp
import queue
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .manos3d import COLOR_MANO, GIRO, LUZ, Mano, cargar_piezas
from .ventana import C, NOMBRES, panel, pct, qcolor

N_ENT, SESGO = 24, 24
COLORES_GESTO = ["#8a96a0", "#5b9bf0", "#e25b52", "#e0a743", "#a77bf0", "#3fbf9a", "#ec8350"]
PALETA = ["#5b9bf0", "#3fbf9a", "#e0a743", "#a77bf0", "#ec8350", "#4fc3d9", "#d96fb0", "#9ccc65",
          "#f0d05b", "#7f8cff", "#ff8a80", "#4db6ac"]
GRIS = (0.30, 0.32, 0.36, 1.0)
VELOCIDADES = [("Cada generación ×1", "ver", 1.0), ("Cada generación ×2", "ver", 2.0),
               ("Cada generación ×4", "ver", 4.0), ("Rápida (ver la última)", "rapido", 4.0)]
NOMBRES_ENTRADAS = [f"{c} t{'' if k == 2 else f'-{2 - k}'}" for k in range(3)
                    for c in [f"{a}_{b}" for a in ("c1", "c2") for b in ("MAV", "RMS", "ZC", "WL")]]


def color_especie(sid):
    return PALETA[(sid or 0) % len(PALETA)]


def tinte(sid):
    c = QtGui.QColor(color_especie(sid))
    return tuple(0.55 * a + 0.45 * b for a, b in zip(COLOR_MANO[:3], (c.redF(), c.greenF(), c.blueF()))) + (1.0,)


class EscenaPoblacion(gl.GLViewWidget):
    """30 manos InMoov en 3 filas de 10."""

    def __init__(self, n=30, columnas=10):
        super().__init__()
        self.setBackgroundColor((22, 28, 34))
        self.setCameraPosition(distance=2350, elevation=4, azimuth=90)
        self.opts["center"] = QtGui.QVector3D(0, 0, 60)
        piezas, plantilla = cargar_piezas()
        self.manos, self.etiquetas = [], []
        filas = int(np.ceil(n / columnas))
        fuente = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.DemiBold)
        centro = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom
        for k in range(n):
            x = -((k % columnas) - (columnas - 1) / 2) * 250.0
            z = ((filas - 1) / 2 - k // columnas) * 470.0
            base = np.eye(4)
            base[:3, 3] = (x, 0, z)
            antebrazo = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=1, cols=20, radius=[32.0, 28.0], length=90.0),
                                      smooth=True, shader=LUZ, color=(0.42, 0.45, 0.5, 1.0))
            antebrazo.translate(x, 0, z - 92)
            self.addItem(antebrazo)
            m = Mano(self, piezas, plantilla, base @ GIRO, COLOR_MANO)
            m.poner([0.25, 0.25, 0.25, 0.25, 0.25, 0.0])
            self.manos.append(m)
            et = gl.GLTextItem(pos=(x, 90, z + 245), text="", color=(230, 236, 240, 255), font=fuente, alignment=centro)
            self.addItem(et)
            self.etiquetas.append(et)
        self._ultimo_texto = [None] * n

    def poner(self, k, pose, color, texto, color_texto):
        m = self.manos[k]
        m.poner(pose)
        m.colorear(color)
        if self._ultimo_texto[k] != (texto, color_texto):
            self.etiquetas[k].setData(text=texto, color=color_texto)
            self._ultimo_texto[k] = (texto, color_texto)


class Pista(QtWidgets.QWidget):
    """Lo que quiere la persona en cada momento, su señal EMG y dónde chocó cada mano."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(118)
        self.p, self.i, self.choques = None, 0.0, []

    def fijar(self, p, choques):
        self.p, self.choques = p, choques
        self.update()

    def paintEvent(self, _):
        if self.p is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        g, est = self.p["gesto"], self.p["estable"]
        T = len(g)
        x0, ancho = 8, W - 16
        X = lambda t: x0 + t / T * ancho
        # señal EMG de los 2 canales (lo que "escuchan" las manos)
        adc = self.p.get("adc")
        y_emg, h_emg = 26, H - 26 - 46
        if adc is not None and len(adc):
            s = adc - np.median(adc, 0)
            esc = np.percentile(np.abs(s), 99.5, axis=0) + 1e-6
            paso = max(1, len(s) // int(ancho))
            s = s[::paso]
            n = len(s)
            corte = int(self.i / T * n)
            for c, color in enumerate((C["acento"], C["canal2"])):
                yc = y_emg + h_emg * (0.27 + 0.46 * c)
                for parte, alfa in (((0, corte), 230), ((corte, n), 60)):
                    a, b = parte
                    if b - a < 2:
                        continue
                    poli = QtGui.QPolygonF([QtCore.QPointF(x0 + k / n * ancho, yc - np.clip(s[k, c] / esc[c], -1, 1) * h_emg * 0.24)
                                            for k in range(a, b)])
                    p.setPen(QtGui.QPen(qcolor(color, alfa), 1))
                    p.drawPolyline(poli)
            p.setPen(QtGui.QColor(C["muted"]))
            p.setFont(QtGui.QFont("Segoe UI", 8))
            p.drawText(QtCore.QRectF(x0, y_emg - 2, 200, 14), Qt.AlignLeft, "electrodo 1 (flexores) · electrodo 2 (extensores)")
        # gestos que quiere la persona
        y_bar, h_bar = H - 40, 22
        ini = 0
        f = QtGui.QFont("Segoe UI", 8)
        f.setBold(True)
        p.setFont(f)
        for t in range(1, T + 1):
            if t == T or g[t] != g[ini]:
                fin_trans = ini + (int(np.argmax(est[ini:t])) if est[ini:t].any() else t - ini)
                p.setPen(Qt.NoPen)
                p.setBrush(qcolor(COLORES_GESTO[g[ini]], 90))
                p.drawRect(QtCore.QRectF(X(ini), y_bar, X(fin_trans) - X(ini), h_bar))
                p.setBrush(qcolor(COLORES_GESTO[g[ini]], 210))
                p.drawRect(QtCore.QRectF(X(fin_trans), y_bar, X(t) - X(fin_trans), h_bar))
                if X(t) - X(ini) > 18:
                    nombre = p.fontMetrics().elidedText(NOMBRES[g[ini]].split()[0], Qt.ElideRight, int(X(t) - X(ini)) - 4)
                    p.setPen(QtGui.QColor("#10161b"))
                    p.drawText(QtCore.QRectF(X(ini), y_bar, X(t) - X(ini), h_bar), Qt.AlignCenter, nombre)
                ini = t
        # choques de las manos que se ven
        p.setPen(QtGui.QPen(qcolor(C["mal"], 200), 2))
        for c in self.choques:
            if c < T and c <= self.i:
                p.drawLine(QtCore.QPointF(X(c), y_bar + h_bar + 2), QtCore.QPointF(X(c), y_bar + h_bar + 9))
        # momento actual
        xa = X(min(self.i, T))
        p.setPen(QtGui.QPen(QtGui.QColor(C["fg"]), 2))
        p.drawLine(QtCore.QPointF(xa, y_emg - 4), QtCore.QPointF(xa, y_bar + h_bar + 2))
        k = min(int(self.i), T - 1)
        f = QtGui.QFont("Segoe UI", 12)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QtGui.QColor(C["fg"]))
        p.drawText(QtCore.QRectF(0, 0, W - 8, 22), Qt.AlignRight | Qt.AlignVCenter,
                   f"La persona quiere: {NOMBRES[g[k]]}   ·   {self.i * 0.1:4.1f} / {T * 0.1:.1f} s")
        p.end()


class RedNeat(QtWidgets.QWidget):
    """La red de la mejor mano: crece con la evolución; se iluminan las neuronas que se activan."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 260)
        self.r, self.i = None, 0

    def fijar(self, red):
        self.r = red
        self.update()

    def _posiciones(self, W, H):
        r = self.r
        nodos, prof, maxp = r["nodos"], r["prof"], r["maxp"]
        izq, der, arriba, abajo = 92, W - 74, 14, H - 10
        columnas = {}
        for n, d in zip(nodos, prof):
            columnas.setdefault(d if n >= N_ENT + 1 else 0, []).append(n)
        pos = {}
        for d, ns in columnas.items():
            x = izq + (der - izq) * (d / maxp if maxp else 0)
            for j, n in enumerate(ns):
                pos[n] = QtCore.QPointF(x, arriba + (j + 0.5) / len(ns) * (abajo - arriba))
        return pos

    def paintEvent(self, _):
        if self.r is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        pos = self._posiciones(W, H)
        r = self.r
        k = min(int(self.i), r["act"].shape[1] - 1)
        act = dict(zip(r["nodos"], r["act"][:, k].astype(float)))
        for a, b, w in r["conexiones"]:
            if a not in pos or b not in pos:
                continue
            fuerza = min(abs(w * act.get(a, 0.0)) / 2.0, 1.0)
            color = C["acento"] if w > 0 else C["neg"]
            p.setPen(QtGui.QPen(qcolor(color, 30 + 200 * fuerza), 0.6 + 2.2 * min(abs(w), 3) / 3))
            p.drawLine(pos[a], pos[b])
        salidas = [n for n in r["nodos"] if N_ENT + 1 <= n < N_ENT + 8]
        logits = np.array([act[n] for n in salidas])
        elegido = int(np.argmax(logits))
        p.setFont(QtGui.QFont("Segoe UI", 7))
        for n, q in pos.items():
            v = act.get(n, 0.0)
            if n < N_ENT + 1:
                rad, brillo = 3.2, min(abs(v) / 2, 1)
                p.setPen(Qt.NoPen)
                p.setBrush(qcolor(C["canal2"] if n < N_ENT and NOMBRES_ENTRADAS[n].startswith("c2") else C["acento"], 60 + 195 * brillo))
                p.drawEllipse(q, rad, rad)
                p.setPen(QtGui.QColor(C["muted"]))
                p.drawText(QtCore.QRectF(0, q.y() - 6, 86, 12), Qt.AlignRight | Qt.AlignVCenter,
                           "sesgo" if n == SESGO else NOMBRES_ENTRADAS[n])
            elif n in salidas:
                j = salidas.index(n)
                ok = j == elegido
                p.setPen(QtGui.QPen(QtGui.QColor(C["fg"] if ok else C["linea"]), 1.5))
                p.setBrush(qcolor(COLORES_GESTO[j], 255 if ok else 70))
                p.drawEllipse(q, 7, 7)
                f = QtGui.QFont("Segoe UI", 8)
                f.setBold(ok)
                p.setFont(f)
                p.setPen(QtGui.QColor(C["fg"] if ok else C["muted"]))
                p.drawText(QtCore.QRectF(q.x() + 10, q.y() - 7, 70, 14), Qt.AlignLeft | Qt.AlignVCenter, NOMBRES[j].split()[0])
                p.setFont(QtGui.QFont("Segoe UI", 7))
            else:
                p.setPen(QtGui.QPen(QtGui.QColor(C["fg"]), 1))
                p.setBrush(qcolor(C["acento"] if v > 0 else C["neg"], 40 + 215 * min(abs(v), 1)))
                p.drawEllipse(q, 5.5, 5.5)
        p.end()


class Especies(QtWidgets.QWidget):
    """Cuántas manos tiene cada especie, generación tras generación (nacen, crecen, se extinguen)."""

    def __init__(self):
        super().__init__()
        self.h = []

    def fijar(self, historial):
        self.h = historial
        self.update()

    def paintEvent(self, _):
        if not self.h:
            return
        p = QtGui.QPainter(self)
        W, H = self.width(), self.height()
        orden = []
        for f in self.h:
            for sid in f["especies"]:
                if sid not in orden:
                    orden.append(sid)
        ancho = W / len(self.h)
        for j, f in enumerate(self.h):
            total = sum(f["especies"].values()) or 1
            y = H
            for sid in orden:
                n = f["especies"].get(sid, 0)
                if n:
                    alto = n / total * H
                    p.fillRect(QtCore.QRectF(j * ancho, y - alto, ancho + 0.6, alto), QtGui.QColor(color_especie(sid)))
                    y -= alto
        p.setPen(QtGui.QColor(C["fg"]))
        p.drawText(QtCore.QRectF(6, 4, W - 12, 16), Qt.AlignLeft,
                   f"generación {self.h[0]['gen']} a {self.h[-1]['gen']} · cada color es una especie")
        p.end()


class VentanaNeuro(QtWidgets.QMainWindow):
    def __init__(self, solo=False):
        super().__init__()
        self.setWindowTitle("Neuroevolución de la mano — aprenden desde cero")
        pant = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1600, pant.width() - 40), min(960, pant.height() - 60))
        ctx = mp.get_context("spawn")
        self.salida, self.entrada = ctx.Queue(maxsize=3), ctx.Queue()
        from .lanzador import correr_neuro
        self.proceso = ctx.Process(target=correr_neuro, args=(self.salida, self.entrada), name="neuroevolucion")
        self.proceso.start()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(8)
        cab = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Neuroevolución")
        t.setStyleSheet("font-size: 18px; font-weight: 700;")
        cab.addWidget(t)
        self.chips = {}
        for k in ("gen", "especies", "aguante", "examen", "r2", "red"):
            c = QtWidgets.QLabel("—")
            c.setObjectName("chip")
            cab.addWidget(c)
            self.chips[k] = c
        cab.addStretch(1)
        self.vel = QtWidgets.QComboBox()
        self.vel.setToolTip("Ver cada generación completa (como el video) o dejarla evolucionar rápido y ver solo la última")
        self.vel.addItems([n for n, _, _ in VELOCIDADES])
        self.vel.currentIndexChanged.connect(self._velocidad)
        cab.addWidget(self.vel)
        self.b_pausa = QtWidgets.QPushButton("⏸ Pausar")
        self.b_pausa.clicked.connect(self._pausa)
        cab.addWidget(self.b_pausa)
        b = QtWidgets.QPushButton("Volver a empezar")
        b.clicked.connect(self._reiniciar)
        cab.addWidget(b)
        v.addLayout(cab)

        rejilla = QtWidgets.QGridLayout()
        rejilla.setSpacing(10)
        f, l = panel("La pista · una persona del gemelo hace 21 gestos; las manos solo escuchan sus 2 electrodos")
        self.pista = Pista()
        l.addWidget(self.pista)
        rejilla.addWidget(f, 0, 0)
        f, l = panel("30 de las 150 manos (de la mejor a la peor) · color = especie · gris = chocó (medio segundo con el gesto equivocado)")
        self.leyenda = QtWidgets.QLabel("Preparando las pistas: simulando personas con sus electrodos…")
        self.leyenda.setStyleSheet("font-size: 13px;")
        l.addWidget(self.leyenda)
        self.escena = EscenaPoblacion()
        self.escena.setMinimumHeight(380)
        l.addWidget(self.escena, 1)
        rejilla.addWidget(f, 1, 0, 3, 1)
        f, l = panel("La red de la mejor mano · crece con la evolución · se iluminan las neuronas activas")
        self.red = RedNeat()
        l.addWidget(self.red, 1)
        rejilla.addWidget(f, 0, 1, 2, 1)
        f, l = panel("Cómo evoluciona")
        tabs = QtWidgets.QTabWidget()
        self.g_aguante = pg.PlotWidget()
        self.g_aguante.setLabel("left", "segundos sin chocar")
        self.g_aguante.setLabel("bottom", "generación")
        self.g_aguante.showGrid(y=True, alpha=0.12)
        self.c_mejor = self.g_aguante.plot(pen=pg.mkPen(C["acento"], width=2), name="mejor")
        self.c_media = self.g_aguante.plot(pen=pg.mkPen(C["muted"], width=1.5), name="promedio")
        self.c_examen = self.g_aguante.plot(pen=pg.mkPen(C["ok"], width=1.5, style=Qt.DashLine), name="examen")
        tabs.addTab(self.g_aguante, "Aguante")
        self.g_acierto = pg.PlotWidget()
        self.g_acierto.setYRange(0, 1)
        self.g_acierto.setLabel("bottom", "generación")
        self.g_acierto.showGrid(y=True, alpha=0.12)
        self.c_acierto = self.g_acierto.plot(pen=pg.mkPen(C["acento"], width=2))
        self.c_pr2 = self.g_acierto.plot(pen=pg.mkPen(C["aviso"], width=1.5))
        self.c_r2 = self.g_acierto.plot(pen=pg.mkPen(C["ok"], width=1.5))
        tabs.addTab(self.g_acierto, "Acierto y R²")
        self.especies = Especies()
        tabs.addTab(self.especies, "Especies")
        l.addWidget(tabs, 1)
        self.nota = QtWidgets.QLabel("Aguante: mejor (azul), promedio (gris), mejor en el examen (verde). "
                                     "Acierto (azul), pseudo-R² (amarillo) y R² de la pose (verde) en 8 personas nuevas.")
        self.nota.setWordWrap(True)
        self.nota.setStyleSheet(f"color: {C['muted']}; font-size: 11px;")
        l.addWidget(self.nota)
        rejilla.addWidget(f, 2, 1)
        f, l = panel("Lo que pasa en la evolución")
        self.eventos = QtWidgets.QListWidget()
        self.eventos.setWordWrap(True)
        l.addWidget(self.eventos, 1)
        rejilla.addWidget(f, 3, 1)
        rejilla.setColumnStretch(0, 7)
        rejilla.setColumnStretch(1, 4)
        rejilla.setRowStretch(0, 0)
        rejilla.setRowStretch(1, 5)
        rejilla.setRowStretch(2, 4)
        rejilla.setRowStretch(3, 2)
        v.addLayout(rejilla, 1)

        self.g, self.pendiente, self.t0, self.pedido = None, None, time.perf_counter(), False
        self.modo, self.factor, self.pausado, self.n_eventos = "ver", 1.0, False, None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tic)
        self.timer.start(40)

    # ---------- controles ----------
    def _velocidad(self, i):
        _, modo, factor = VELOCIDADES[i]
        ahora = time.perf_counter()
        if self.g is not None:                                     # sigue en el mismo instante de la pista
            self.t0 = ahora - (ahora - self.t0) * self.factor / factor
        self.modo, self.factor = modo, factor
        self.entrada.put({"cmd": "modo", "valor": modo})

    def _pausa(self):
        self.pausado = not self.pausado
        self.entrada.put({"cmd": "pausa", "valor": self.pausado})
        self.b_pausa.setText("▶ Reanudar" if self.pausado else "⏸ Pausar")
        if self.pausado:
            self.t_pausa = time.perf_counter()
        else:
            self.t0 += time.perf_counter() - self.t_pausa

    def _reiniciar(self):
        r = QtWidgets.QMessageBox.question(self, "Volver a empezar",
                                           "Las 150 manos olvidan todo y nacen de nuevo con redes casi vacías. "
                                           "La población actual y sus campeones quedan guardados como respaldo. ¿Seguir?")
        if r == QtWidgets.QMessageBox.Yes:
            self.entrada.put({"cmd": "reiniciar"})
            self.pendiente = None

    # ---------- mensajes ----------
    def _recibir(self, m):
        t = m["tipo"]
        if t == "generacion":
            if self.g is None or self._terminada():
                self._empezar(m)
            else:
                self.pendiente = m
        elif t == "aviso":
            self.leyenda.setText(m["texto"])
        elif t == "error":
            QtWidgets.QMessageBox.critical(self, "La neuroevolución se detuvo", m["texto"][-1500:])

    def _empezar(self, m):
        self.g, self.pendiente, self.pedido = m, None, False
        self.t0 = time.perf_counter()
        self.T = len(m["pista"]["gesto"])
        self.choques = np.array([h["choque"] for h in m["manos"]])
        self.fin = min(self.T, int(self.choques.max()) + 8)
        self.pista.fijar(m["pista"], self.choques.tolist())
        self.red.fijar(m["red"])
        self.especies.fijar(m["historial"])
        h = m["historial"]
        gens = np.array([f["gen"] for f in h])
        self.c_mejor.setData(gens, [f["mejor"] for f in h])
        self.c_media.setData(gens, [f["media"] for f in h])
        self.c_examen.setData(gens, [f["aguante_examen"] for f in h])
        self.c_acierto.setData(gens, [f["acierto"] for f in h])
        self.c_pr2.setData(gens, np.clip([f["pseudo_r2"] for f in h], 0, 1))
        self.c_r2.setData(gens, np.clip([f["r2_pose"] for f in h], 0, 1))
        b = m["mejor"]
        self.chips["gen"].setText(f"gen {m['gen']}")
        self.chips["especies"].setText(f"{m['n_especies']} especies")
        self.chips["especies"].setToolTip(f"{m['n_poblacion']} manos agrupadas en {m['n_especies']} especies por el parecido de sus redes")
        self.chips["aguante"].setText(f"mejor {m['manos'][0]['aguante']:.1f}/{self.T * m['dt']:.0f} s")
        self.chips["aguante"].setToolTip("Segundos que aguantó la mejor mano en esta pista sin chocar")
        self.chips["examen"].setText(f"examen {pct(b['acierto'])}")
        self.chips["examen"].setToolTip(f"La mejor mano con 8 personas que nunca vio: acierto {pct(b['acierto'])}, "
                                        f"aguanta {b['aguante']:.1f} s de {b['largo']:.0f} en promedio")
        self.chips["r2"].setText(f"pR² {b['pseudo_r2']:.3f} · R² {b['r2_pose']:.3f}")
        self.chips["r2"].setToolTip("Pseudo-R² de McFadden del gesto y R² de la pose de los dedos, en las 8 personas del examen")
        mm = m["manos"][0]
        self.chips["red"].setText(f"red {mm['ocultas']}·{mm['conexiones']}")
        self.chips["red"].setToolTip(f"La red de la mejor mano: {mm['ocultas']} neuronas ocultas y {mm['conexiones']} conexiones")
        c = m.get("campeon")
        self.leyenda.setText(f"<b>Generación {m['gen']}</b> · persona de nivel {m['pista']['nivel']} que ninguna mano vio antes · "
                             + (f"campeón #{c['n']} (gen {c['gen']}): {pct(c['acierto'])} en el examen" if c else "todavía sin campeón"))
        ultimo = m["eventos"][-1]["texto"] if m["eventos"] else None
        if (len(m["eventos"]), ultimo) != self.n_eventos:
            self.n_eventos = (len(m["eventos"]), ultimo)
            self.eventos.clear()
            for ev in reversed(m["eventos"]):
                it = QtWidgets.QListWidgetItem(f"gen {ev['gen']} · {ev['texto']}")
                if ev["texto"].startswith("★"):
                    it.setForeground(QtGui.QBrush(QtGui.QColor(C["aviso"])))
                self.eventos.addItem(it)

    def _instante(self):
        if self.pausado:
            return (self.t_pausa - self.t0) * self.factor / 0.1
        return (time.perf_counter() - self.t0) * self.factor / 0.1

    def _terminada(self):
        return self.g is not None and self._instante() >= self.fin

    def _tic(self):
        for _ in range(3):
            try:
                self._recibir(self.salida.get_nowait())
            except queue.Empty:
                break
        if self.g is None:
            return
        if self._terminada() and not self.pausado:
            if self.pendiente is not None:
                self._empezar(self.pendiente)
            elif self.modo == "ver" and not self.pedido:
                self.entrada.put({"cmd": "siguiente"})
                self.pedido = True
        i = min(self._instante(), self.T - 1e-3)
        poses = self.g["poses"]
        for k, h in enumerate(self.g["manos"]):
            c = h["choque"]
            choco = c < self.T and i >= c
            j = min(i, c - 1) if choco else i
            a, frac = int(j), j - int(j)
            b = min(a + 1, self.T - 1)
            pose = poses[k, a] + (poses[k, b] - poses[k, a]) * frac
            if choco:
                texto, color_t, color = f"✗ #{h['rank']} · {c * 0.1:.1f} s", (240, 120, 110, 255), GRIS
            elif c >= self.T and i >= self.T - 1:
                texto, color_t, color = f"✓ #{h['rank']} completó", (120, 220, 160, 255), tinte(h["especie"])
            else:
                texto, color_t = f"#{h['rank']} · {i * 0.1:.1f} s", (140, 190, 255, 255) if h["rank"] == 1 else (230, 236, 240, 255)
                color = tinte(h["especie"])
            self.escena.poner(k, pose, color, texto, color_t)
        self.pista.i = i
        self.pista.update()
        self.red.i = i
        self.red.update()

    def closeEvent(self, ev):
        if self.proceso.is_alive():
            self.entrada.put({"cmd": "detener"})
            fin = time.time() + 15
            while self.proceso.is_alive() and time.time() < fin:
                QtWidgets.QApplication.processEvents()
                try:
                    if self.salida.get(timeout=0.2).get("tipo") == "detenido":
                        break
                except queue.Empty:
                    pass
            self.proceso.join(3)
            if self.proceso.is_alive():
                self.proceso.terminate()
        ev.accept()
