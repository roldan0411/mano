"""
Ventana "Entrenamiento libre": la mano bebé aprende de cero mirando internet
===========================================================================

  ┌ la mano InMoov ─────────┬ lo que mira en internet ──┬ qué tan bien imita ─────┐
  │ sólida = lo que logra   │ mano real (HaGRID v2)     │ los 33 gestos de        │
  │ azul = lo que imita     ├ mapa de lo que sabe hacer ┤ internet, uno por fila  │
  │ músculos que activa     │                           ├ curvas de aprendizaje ──┤
  ├─────────────────────────┴───────────────────────────┴─────────────────────────┤
  │ gestos de internet que aprendió (foto: la persona y la mano bebé)             │
  └────────────────────────────────────────────────────────────────────────────────┘
"""

import multiprocessing as mp
import queue
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from .manos3d import COLOR_MANO, COLOR_OBJETIVO, GIRO, Mano, cargar_piezas
from .ventana import C, panel, qcolor

ETAPAS = {"balbuceo": ("Balbuceo", "mueve sus músculos al azar para conocer su propio cuerpo"),
          "imita": ("Imita lo que ve en internet", "mira manos de personas reales e intenta hacer lo mismo"),
          "domina": ("Domina", "ya imita bien casi todo lo que su cuerpo permite; afina")}
ESTRELLA, ROJO, VERDE = "#e0a743", "#ef6b63", "#3fbf7f"
VELOCIDADES = [("Tiempo real", 1.0), ("× 5", 5.0), ("× 20", 20.0), ("× 100", 100.0), ("Máxima", 1e6)]
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def lienzo(titulo):
    f, l = panel(titulo)
    return f, l


class ManoGrande(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor((22, 28, 34))
        self.setCameraPosition(distance=640, elevation=12, azimuth=70)
        self.opts["center"] = QtGui.QVector3D(0, 0, 75)
        piezas, plantilla = cargar_piezas()
        antebrazo = gl.GLMeshItem(meshdata=gl.MeshData.cylinder(rows=1, cols=32, radius=[34.0, 29.0], length=130.0),
                                  smooth=True, color=(0.42, 0.45, 0.5, 1.0))
        antebrazo.translate(0, 0, -132)
        self.addItem(antebrazo)
        self.real = Mano(self, piezas, plantilla, GIRO.copy(), COLOR_MANO)
        self.meta = Mano(self, piezas, plantilla, GIRO.copy(), COLOR_OBJETIVO, "translucent")
        for m in (self.real, self.meta):
            m.poner([0.25, 0.25, 0.25, 0.25, 0.25, 0.0])


class Musculos(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        self.nombres, self.a = [], np.zeros(10)

    def fijar(self, nombres, a):
        self.nombres, self.a = nombres, np.asarray(a, float)
        self.update()

    def paintEvent(self, _):
        if not self.nombres:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        ancho = W / len(self.nombres)
        f = QtGui.QFont("Segoe UI", 9)
        f.setBold(True)
        p.setFont(f)
        for k, (nom, v) in enumerate(zip(self.nombres, self.a)):
            x = k * ancho + 5
            alto = (H - 26) * v
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(C["bg"]))
            p.drawRoundedRect(QtCore.QRectF(x, 4, ancho - 10, H - 30), 4, 4)
            p.setBrush(qcolor(C["musculo"], 90 + 165 * v))
            p.drawRoundedRect(QtCore.QRectF(x, 4 + (H - 30) - alto, ancho - 10, alto), 4, 4)
            p.setPen(QtGui.QColor(C["fg"] if v > 0.05 else C["muted"]))
            p.drawText(QtCore.QRectF(x - 4, H - 22, ancho - 2, 18), Qt.AlignCenter, nom)
        p.end()


class ManoInternet(QtWidgets.QWidget):
    """La mano de la persona real que está mirando (21 puntos de HaGRID)."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 200)
        self.L, self.nombre, self.fuente = None, "", ""

    def fijar(self, L, nombre):
        self.L, self.nombre = (None if L is None else np.asarray(L, float)), nombre or ""
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        if self.L is None:
            p.setPen(QtGui.QColor(C["muted"]))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                       "Todavía no mira internet:\nprimero conoce su propio cuerpo balbuceando")
            p.end()
            return
        L = self.L
        lo, hi = L.min(0), L.max(0)
        esc = min((W - 30) / max(hi[0] - lo[0], 1e-6), (H - 60) / max(hi[1] - lo[1], 1e-6))
        cx, cy = W / 2 - (lo[0] + hi[0]) / 2 * esc, (H - 24) / 2 + 18 + (lo[1] + hi[1]) / 2 * esc

        def P(q):
            return QtCore.QPointF(cx + q[0] * esc, cy - q[1] * esc)

        p.setPen(QtGui.QPen(QtGui.QColor(C["acento"]), 4, Qt.SolidLine, Qt.RoundCap))
        for a, b in CONEXIONES:
            p.drawLine(P(L[a]), P(L[b]))
        p.setPen(QtGui.QPen(QtGui.QColor(C["acento"]), 1.5))
        p.setBrush(QtGui.QColor(C["fg"]))
        for q in L:
            p.drawEllipse(P(q), 3.2, 3.2)
        f = QtGui.QFont("Segoe UI", 11)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QtGui.QColor(C["fg"]))
        p.drawText(QtCore.QRectF(0, 2, W, 22), Qt.AlignCenter, f"«{self.nombre}»")
        p.setFont(QtGui.QFont("Segoe UI", 8))
        p.setPen(QtGui.QColor(C["muted"]))
        p.drawText(QtCore.QRectF(0, H - 18, W, 16), Qt.AlignCenter, "persona real · HaGRID v2")
        p.end()


class Imitacion(QtWidgets.QWidget):
    """Una fila por gesto de internet: qué tan bien lo imita, si lo aprendió o si su anatomía no lo permite."""

    def __init__(self):
        super().__init__()
        self.clases, self.actual = [], None
        self.setMinimumHeight(300)

    def fijar(self, clases, actual=None):
        self.clases = clases
        if actual is not None:
            self.actual = actual
        self.update()

    def paintEvent(self, _):
        if not self.clases:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        orden = {"aprendido": 0, "aprendiendo": 1, "imposible": 2}
        filas = sorted(self.clases, key=lambda c: (orden[c["estado"]], c["error"] if c["error"] == c["error"] else 9))
        alto = H / len(filas)
        p.setFont(QtGui.QFont("Segoe UI", max(7, min(10, int(alto * 0.55)))))
        for k, c in enumerate(filas):
            y = k * alto
            calidad = 0.0 if c["error"] != c["error"] else float(np.clip(1 - c["error"] / 0.5, 0, 1))
            color = VERDE if c["estado"] == "aprendido" else ROJO if c["estado"] == "imposible" else C["acento"]
            if c["nombre"] == self.actual:
                p.fillRect(QtCore.QRectF(0, y, W, alto), qcolor(C["acento"], 35))
            p.setPen(QtGui.QColor(C["fg"] if c["intentos"] else C["muted"]))
            p.drawText(QtCore.QRectF(4, y, 150, alto), Qt.AlignVCenter | Qt.AlignLeft, c["nombre"])
            x0, ancho = 158, W - 158 - 70
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(C["bg"]))
            p.drawRoundedRect(QtCore.QRectF(x0, y + alto * 0.2, ancho, alto * 0.6), 3, 3)
            p.setBrush(qcolor(color, 220))
            p.drawRoundedRect(QtCore.QRectF(x0, y + alto * 0.2, ancho * calidad, alto * 0.6), 3, 3)
            p.setPen(QtGui.QColor(color if c["estado"] != "aprendiendo" else C["muted"]))
            texto = {"aprendido": "✓ aprendido", "imposible": "✗ imposible", "aprendiendo": f"{c['intentos']} int."}[c["estado"]]
            p.drawText(QtCore.QRectF(W - 68, y, 66, alto), Qt.AlignVCenter | Qt.AlignRight, texto)
        p.end()


class MapaWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 200)
        self.m, self.puntos, self.hitos, self.actual = None, np.zeros((0, 2)), [], None

    def paintEvent(self, _):
        if self.m is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        lo, hi = np.asarray(self.m["lo"]), np.asarray(self.m["hi"])

        def P(z):
            return QtCore.QPointF(14 + (z[0] - lo[0]) / (hi[0] - lo[0]) * (W - 28), H - 14 - (z[1] - lo[1]) / (hi[1] - lo[1]) * (H - 40))

        p.setPen(Qt.NoPen)
        p.setBrush(qcolor(C["muted"], 45))
        for z in self.m["nube"][::3]:
            p.drawEllipse(P(z), 1.5, 1.5)
        p.setBrush(qcolor(C["acento"], 110))
        for z in self.puntos[::3]:
            p.drawEllipse(P(z), 1.9, 1.9)
        aprendidos = {h["nombre"] for h in self.hitos}
        p.setFont(QtGui.QFont("Segoe UI", 8))
        for z, nombre in zip(self.m["gestos"], self.m.get("nombres_gestos", [])):
            q = P(z)
            ok = nombre in aprendidos
            p.setPen(QtGui.QPen(QtGui.QColor(VERDE if ok else C["muted"]), 1.2))
            p.setBrush(QtGui.QColor(VERDE) if ok else Qt.NoBrush)
            p.drawEllipse(q, 3.5, 3.5)
        if self.actual:
            z, zm = self.actual
            if zm is not None:
                p.setPen(QtGui.QPen(QtGui.QColor(C["fg"]), 1.2, Qt.DashLine))
                p.drawLine(P(zm), P(z))
                p.setBrush(Qt.NoBrush)
                p.setPen(QtGui.QPen(QtGui.QColor(C["acento"]), 2))
                p.drawEllipse(P(zm), 7, 7)
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(C["fg"]))
            p.drawEllipse(P(z), 4.5, 4.5)
        p.setPen(QtGui.QColor(C["muted"]))
        p.drawText(QtCore.QRectF(6, 2, W - 12, 16), Qt.AlignLeft,
                   "gris: lo que su cuerpo permite · azul: lo que probó · ● gesto de internet (verde: aprendido)")
        p.end()


class VentanaLibre(QtWidgets.QMainWindow):
    def __init__(self, solo=False):
        super().__init__()
        self.setWindowTitle("Mano bebé — aprende de internet")
        pant = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1520, pant.width() - 50), min(900, pant.height() - 70))
        ctx = mp.get_context("spawn")
        self.salida, self.entrada = ctx.Queue(maxsize=20), ctx.Queue()
        from .lanzador import correr_bebe
        self.proceso = ctx.Process(target=correr_bebe, args=(self.salida, self.entrada), name="mano-bebe")
        self.proceso.start()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(10)
        cab = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Mano bebé")
        t.setStyleSheet("font-size: 18px; font-weight: 700;")
        cab.addWidget(t)
        self.chips = {}
        for k in ("etapa", "intentos", "aprendidos", "imposibles", "r2"):
            c = QtWidgets.QLabel("—")
            c.setObjectName("chip")
            cab.addWidget(c)
            self.chips[k] = c
        cab.addStretch(1)
        cab.addWidget(QtWidgets.QLabel("Velocidad"))
        self.vel = QtWidgets.QComboBox()
        self.vel.addItems([n for n, _ in VELOCIDADES])
        self.vel.currentIndexChanged.connect(lambda i: self.entrada.put({"cmd": "velocidad", "valor": VELOCIDADES[i][1]}))
        cab.addWidget(self.vel)
        self.b_pausa = QtWidgets.QPushButton("⏸ Pausar")
        self.b_pausa.clicked.connect(self._pausa)
        cab.addWidget(self.b_pausa)
        b = QtWidgets.QPushButton("Volver a nacer")
        b.clicked.connect(self._renacer)
        cab.addWidget(b)
        v.addLayout(cab)

        rejilla = QtWidgets.QGridLayout()
        rejilla.setSpacing(12)
        f, l = panel("La mano bebé · sólida = lo que logra · azul transparente = lo que intenta imitar")
        self.leyenda = QtWidgets.QLabel("Naciendo…")
        self.leyenda.setStyleSheet("font-size: 14px;")
        self.leyenda.setWordWrap(True)
        l.addWidget(self.leyenda)
        self.mano = ManoGrande()
        self.mano.setMinimumHeight(240)
        l.addWidget(self.mano, 1)
        t = QtWidgets.QLabel("MÚSCULOS QUE ACTIVA EN ESTE INTENTO")
        t.setObjectName("titulo")
        l.addWidget(t)
        self.musculos = Musculos()
        l.addWidget(self.musculos)
        rejilla.addWidget(f, 0, 0, 2, 1)
        f, l = panel("Lo que está mirando en internet")
        self.internet = ManoInternet()
        l.addWidget(self.internet, 1)
        rejilla.addWidget(f, 0, 1)
        f, l = panel("Mapa de lo que sabe hacer")
        self.mapa = MapaWidget()
        l.addWidget(self.mapa, 1)
        rejilla.addWidget(f, 1, 1)
        f, l = panel("Qué tan bien imita cada gesto de internet")
        self.imitacion = Imitacion()
        l.addWidget(self.imitacion, 1)
        rejilla.addWidget(f, 0, 2)
        f, l = panel("Cómo aprende · error imitando (azul) · R² de la imitación (gris)")
        self.g = pg.PlotWidget()
        self.g.setYRange(0, 1)
        self.g.setLabel("bottom", "intentos")
        self.g.showGrid(y=True, alpha=0.12)
        self.c_error = self.g.plot(pen=pg.mkPen(C["acento"], width=2))
        self.c_r2 = self.g.plot(pen=pg.mkPen(C["muted"], width=1.5))
        self.g.setMinimumHeight(120)
        l.addWidget(self.g)
        rejilla.addWidget(f, 1, 2)
        rejilla.setColumnStretch(0, 5)
        rejilla.setColumnStretch(1, 3)
        rejilla.setColumnStretch(2, 4)
        rejilla.setRowStretch(0, 3)
        rejilla.setRowStretch(1, 2)
        v.addLayout(rejilla, 1)
        f, l = panel("Gestos de internet que aprendió · foto: la mano de la persona y la de la mano bebé")
        self.galeria = QtWidgets.QListWidget()
        self.galeria.setViewMode(QtWidgets.QListView.IconMode)
        self.galeria.setFlow(QtWidgets.QListView.LeftToRight)
        self.galeria.setWrapping(False)
        self.galeria.setIconSize(QtCore.QSize(150, 92))
        self.galeria.setGridSize(QtCore.QSize(180, 140))
        self.galeria.setWordWrap(True)
        self.galeria.setFixedHeight(150)
        self.galeria.setMovement(QtWidgets.QListView.Static)
        l.addWidget(self.galeria)
        f.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        v.addWidget(f, 0)

        self.items_hito, self.pausado, self.nombres_musculos = {}, False, []
        self.desde = self.hacia = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.0])
        self.meta, self.t_anim, self.dur = None, time.perf_counter(), 0.5
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tic)
        self.timer.start(30)

    # ---------- controles ----------
    def _pausa(self):
        self.pausado = not self.pausado
        self.entrada.put({"cmd": "pausa", "valor": self.pausado})
        self.b_pausa.setText("▶ Reanudar" if self.pausado else "⏸ Pausar")

    def _renacer(self):
        r = QtWidgets.QMessageBox.question(self, "Volver a nacer",
                                           "La mano olvida todo lo que aprendió y empieza de cero, balbuceando. "
                                           "Las fotos de lo que aprendió quedan guardadas. ¿Seguir?")
        if r == QtWidgets.QMessageBox.Yes:
            self.entrada.put({"cmd": "renacer"})

    # ---------- mensajes ----------
    def _recibir(self, m):
        t = m["tipo"]
        if t == "mapa":
            self.mapa.m = m
            self.nombres_musculos = m["musculos"]
        elif t == "intento":
            ahora = time.perf_counter()
            self.desde, self.hacia, self.t_anim = self._pose_actual(ahora), np.asarray(m["alcanzado"]), ahora
            self.dur = float(np.clip(0.5 / np.sqrt(m.get("velocidad", 1.0)), 0.08, 0.5))
            self.meta = np.asarray(m["meta"]) if m["meta"] is not None else None
            self.mano.meta.colorear(COLOR_OBJETIVO if self.meta is not None else (0, 0, 0, 0))
            self.mapa.actual = (m["z"], m["zmeta"])
            self.musculos.fijar(self.nombres_musculos, m["activacion"])
            if m["obs"] is not None:
                self.internet.fijar(m["obs"], m["nombre"])
                self.imitacion.actual = m["nombre"]
            elif m["etapa"] == "balbuceo":
                self.internet.fijar(None, "")
            nombre, desc = ETAPAS[m["etapa"]]
            texto = f"<b>Intento {m['n']:,}</b> · <b>{nombre}</b> — {desc}".replace(",", ".")
            if m["nombre"]:
                texto += f"<br>Mira a una persona hacer <b>«{m['nombre']}»</b> e intenta imitarla · error <b>{m['error']:.3f}</b>"
            elif m["meta"] is not None:
                texto += f"<br>Juega sola con una pose que se inventó · error {m['error']:.3f}"
            self.leyenda.setText(texto)
            self.mapa.update()
        elif t == "estado":
            self._estado(m)
        elif t == "descubrimiento":
            self._hito(m)
        elif t == "foto":
            it = self.items_hito.get(m["n"])
            if it:
                it.setIcon(QtGui.QIcon(QtGui.QPixmap(m["foto"])))
        elif t == "renacio":
            self.galeria.clear()
            self.items_hito = {}
        elif t == "error":
            QtWidgets.QMessageBox.critical(self, "La mano bebé se detuvo", m["texto"][-1500:])

    def _hito(self, h):
        clave = h["n"] if h.get("tipo") == "aprendido" else f"imposible-{h['nombre']}"
        if clave in self.items_hito:
            return
        if h.get("tipo") == "imposible":
            it = QtWidgets.QListWidgetItem(f"✗ «{h['nombre']}»\nsu anatomía no lo permite\n(intento {h['intento']:,})".replace(",", "."))
            it.setForeground(QtGui.QBrush(QtGui.QColor(ROJO)))
        else:
            it = QtWidgets.QListWidgetItem(f"✓ «{h['nombre']}»\nintento {h['intento']:,}".replace(",", "."))
            it.setForeground(QtGui.QBrush(QtGui.QColor(VERDE)))
            if h.get("foto"):
                it.setIcon(QtGui.QIcon(QtGui.QPixmap(h["foto"])))
        self.galeria.insertItem(0, it)
        self.items_hito[clave] = it

    def _estado(self, e):
        nombre, _ = ETAPAS[e["etapa"]]
        self.chips["etapa"].setText(nombre)
        self.chips["intentos"].setText(f"{e['intentos']:,} intentos".replace(",", "."))
        n = len(e["clases"]) or 33
        self.chips["aprendidos"].setText(f"✓ {e['aprendidos']}/{n} aprendidos")
        self.chips["aprendidos"].setToolTip("Gestos de internet que ya imita bien")
        self.chips["imposibles"].setText(f"✗ {e['imposibles']} imposibles")
        self.chips["imposibles"].setToolTip("Gestos que su anatomía no permite (necesitan mover dedos sueltos)")
        self.chips["r2"].setText("R² —" if e["r2_imitacion"] != e["r2_imitacion"] else f"R² {e['r2_imitacion']:.3f}")
        self.chips["r2"].setToolTip("R² de la imitación: poses logradas frente a las manos de internet del examen")
        self.mapa.puntos, self.mapa.hitos = e["puntos"], e["hitos"]
        self.imitacion.fijar(e["clases"])
        for h in e["hitos"]:
            self._hito(h)
        c = e["curva"]
        if len(c):
            ok = ~np.isnan(c[:, 1])
            self.c_error.setData(c[ok, 0], c[ok, 1])
            self.c_r2.setData(c[ok, 0], np.clip(c[ok, 2], 0, 1))

    def _pose_actual(self, ahora):
        s = min(1.0, (ahora - self.t_anim) / self.dur)
        s = s * s * (3 - 2 * s)
        return self.desde + (self.hacia - self.desde) * s

    def _tic(self):
        for _ in range(40):
            try:
                self._recibir(self.salida.get_nowait())
            except queue.Empty:
                break
        self.mano.real.poner(self._pose_actual(time.perf_counter()))
        if self.meta is not None:
            self.mano.meta.poner(self.meta)

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
