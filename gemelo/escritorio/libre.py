"""
Ventana "Entrenamiento libre": la mano bebé aprende sola qué puede hacer
=======================================================================

  ┌ la mano InMoov ─────────┬ qué decide hacer ─────────┬ su repertorio ─────────┐
  │ sólida = lo que logra   │ estrategias + diario      │ habilidades que         │
  │ azul = lo que se propuso├ espejo de internet ───────┤ descubrió (sin tope)    │
  │ músculos que activa     │ mapa de lo que sabe hacer ├ curvas y R² ────────────┤
  ├─────────────────────────┴───────────────────────────┴─────────────────────────┤
  │ galería: cada habilidad con su foto (y la persona de internet que se parece)   │
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
          "explora": ("Explora libre", "se pone sus propias metas y decide sola cómo aprender")}
ESTRATEGIAS = {"inventar": ("Inventar", "se imagina una pose cualquiera"),
               "frontera": ("Ir más allá", "lleva más lejos algo que ya sabe"),
               "practicar": ("Practicar", "repite lo nuevo hasta dominarlo"),
               "balbucear": ("Balbucear", "músculos al azar")}
ESTRELLA, VERDE = "#e0a743", "#3fbf7f"
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


class Espejo(QtWidgets.QWidget):
    """La persona real de internet cuya mano más se parece a lo que la mano bebé está haciendo."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(170, 150)
        self.L, self.nombre, self.d = None, "", None

    def fijar(self, L, nombre, d=None):
        self.L, self.nombre, self.d = (None if L is None else np.asarray(L, float)), nombre or "", d
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        if self.L is None:
            p.setPen(QtGui.QColor(C["muted"]))
            p.drawText(self.rect().adjusted(8, 0, -8, 0), Qt.AlignCenter | Qt.TextWordWrap,
                       "Ninguna persona de internet hace algo parecido a esto")
            p.end()
            return
        L = self.L
        lo, hi = L.min(0), L.max(0)
        esc = min((W - 30) / max(hi[0] - lo[0], 1e-6), (H - 74) / max(hi[1] - lo[1], 1e-6))
        cx, cy = W / 2 - (lo[0] + hi[0]) / 2 * esc, (H - 24) / 2 + 26 + (lo[1] + hi[1]) / 2 * esc

        def P(q):
            return QtCore.QPointF(cx + q[0] * esc, cy - q[1] * esc)

        p.setPen(QtGui.QPen(QtGui.QColor(C["acento"]), 4, Qt.SolidLine, Qt.RoundCap))
        for a, b in CONEXIONES:
            p.drawLine(P(L[a]), P(L[b]))
        p.setPen(QtGui.QPen(QtGui.QColor(C["acento"]), 1.5))
        p.setBrush(QtGui.QColor(C["fg"]))
        for q in L:
            p.drawEllipse(P(q), 3.2, 3.2)
        f = QtGui.QFont("Segoe UI", 9)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QtGui.QColor(C["fg"]))
        p.drawText(QtCore.QRectF(2, 0, W - 4, 34), Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
                   f"se parece a «{self.nombre}»")
        p.setFont(QtGui.QFont("Segoe UI", 8))
        p.setPen(QtGui.QColor(C["muted"]))
        p.drawText(QtCore.QRectF(0, H - 18, W, 16), Qt.AlignCenter,
                   "HaGRID v2" + (f" · dist. {self.d:.2f}" if self.d is not None else ""))
        p.end()


class Decision(QtWidgets.QWidget):
    """Qué estrategia prefiere ahora (lo decide ella según cuánto le está enseñando cada una)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(118)
        self.pref, self.actual = {}, None

    def fijar(self, pref=None, actual=None):
        if pref is not None:
            self.pref = dict(zip(ESTRATEGIAS, pref))
        if actual is not None:
            self.actual = actual
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        alto = H / len(ESTRATEGIAS)
        for k, (clave, (nombre, desc)) in enumerate(ESTRATEGIAS.items()):
            y = k * alto
            v = self.pref.get(clave, 0.0)
            activa = clave == self.actual
            if activa:
                p.fillRect(QtCore.QRectF(0, y, W, alto), qcolor(C["acento"], 40))
            f = QtGui.QFont("Segoe UI", 9)
            f.setBold(activa)
            p.setFont(f)
            p.setPen(QtGui.QColor(C["fg"] if activa else C["muted"]))
            p.drawText(QtCore.QRectF(6, y, 92, alto), Qt.AlignVCenter | Qt.AlignLeft, nombre)
            x0, ancho = 100, max(W - 100 - 54, 10)
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(C["bg"]))
            p.drawRoundedRect(QtCore.QRectF(x0, y + alto * 0.25, ancho, alto * 0.5), 3, 3)
            p.setBrush(qcolor(C["acento"], 230 if activa else 150))
            p.drawRoundedRect(QtCore.QRectF(x0, y + alto * 0.25, ancho * v, alto * 0.5), 3, 3)
            p.setPen(QtGui.QColor(C["muted"]))
            p.setFont(QtGui.QFont("Segoe UI", 8))
            p.drawText(QtCore.QRectF(W - 52, y, 50, alto), Qt.AlignVCenter | Qt.AlignRight, f"{v * 100:.0f} %")
        p.end()


class MapaWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 200)
        self.m, self.puntos, self.actual = None, np.zeros((0, 2)), None
        self.z_hab, self.z_cand = np.zeros((0, 2)), np.zeros((0, 2))

    def paintEvent(self, _):
        if self.m is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        lo, hi = np.asarray(self.m["lo"]), np.asarray(self.m["hi"])

        def P(z):
            return QtCore.QPointF(14 + (z[0] - lo[0]) / (hi[0] - lo[0]) * (W - 28), H - 14 - (z[1] - lo[1]) / (hi[1] - lo[1]) * (H - 56))

        p.setPen(Qt.NoPen)
        p.setBrush(qcolor(C["muted"], 45))
        for z in self.m["nube"][::3]:
            p.drawEllipse(P(z), 1.5, 1.5)
        p.setBrush(qcolor(C["acento"], 110))
        for z in self.puntos[::3]:
            p.drawEllipse(P(z), 1.9, 1.9)
        p.setPen(QtGui.QPen(QtGui.QColor(C["bg"]), 0.8))
        p.setBrush(QtGui.QColor(VERDE))
        for z in self.z_hab:
            p.drawEllipse(P(z), 3.2, 3.2)
        p.setPen(QtGui.QPen(QtGui.QColor(ESTRELLA), 1.6))
        p.setBrush(Qt.NoBrush)
        for z in self.z_cand:
            p.drawEllipse(P(z), 5, 5)
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
        p.drawText(QtCore.QRectF(6, 2, W - 12, 34), Qt.AlignLeft | Qt.TextWordWrap,
                   "gris: su cuerpo · azul: probó · verde: sabe · amarillo: practica")
        p.end()


class VentanaLibre(QtWidgets.QMainWindow):
    def __init__(self, solo=False):
        super().__init__()
        self.setWindowTitle("Mano bebé — aprende sola qué puede hacer")
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
        for k in ("etapa", "intentos", "habilidades", "finura", "r2", "espejo"):
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
        f, l = panel("La mano bebé · sólida = lo que logra · azul transparente = la meta que ella misma se propuso")
        self.leyenda = QtWidgets.QLabel("Despertando…")
        self.leyenda.setStyleSheet("font-size: 14px;")
        self.leyenda.setWordWrap(True)
        self.leyenda.setMinimumHeight(44)
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
        f, l = panel("Qué decide hacer · prefiere lo que más le está enseñando")
        self.decision = Decision()
        l.addWidget(self.decision)
        self.diario = QtWidgets.QListWidget()
        self.diario.setWordWrap(True)
        self.diario.setMinimumHeight(70)
        l.addWidget(self.diario, 1)
        rejilla.addWidget(f, 0, 1)
        f, l = panel("Espejo de internet · mapa de lo que sabe hacer")
        h = QtWidgets.QHBoxLayout()
        self.espejo = Espejo()
        h.addWidget(self.espejo, 2)
        self.mapa = MapaWidget()
        h.addWidget(self.mapa, 3)
        l.addLayout(h, 1)
        rejilla.addWidget(f, 1, 1)
        f, l = panel("Su repertorio · lo que descubrió sola, sin tope")
        self.tabla = QtWidgets.QTableWidget(0, 4)
        self.tabla.setHorizontalHeaderLabels(["#", "Qué hace", "Dominio", "En internet se parece a"])
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabla.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tabla.setWordWrap(False)
        cab_t = self.tabla.horizontalHeader()
        cab_t.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for c in (0, 2, 3):
            cab_t.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        l.addWidget(self.tabla, 1)
        rejilla.addWidget(f, 0, 2)
        f, l = panel("Cómo aprende · R² del control (azul) · R² de su cuerpo (gris) · cobertura (verde)")
        self.g = pg.PlotWidget()
        self.g.setYRange(0, 1)
        self.g.setLabel("bottom", "intentos")
        self.g.showGrid(y=True, alpha=0.12)
        self.c_control = self.g.plot(pen=pg.mkPen(C["acento"], width=2))
        self.c_cuerpo = self.g.plot(pen=pg.mkPen(C["muted"], width=1.5))
        self.c_cob = self.g.plot(pen=pg.mkPen(VERDE, width=1.5, style=Qt.DashLine))
        self.g.setMinimumHeight(120)
        l.addWidget(self.g)
        rejilla.addWidget(f, 1, 2)
        rejilla.setColumnStretch(0, 5)
        rejilla.setColumnStretch(1, 4)
        rejilla.setColumnStretch(2, 4)
        rejilla.setRowStretch(0, 3)
        rejilla.setRowStretch(1, 2)
        v.addLayout(rejilla, 1)
        f, l = panel("Habilidades que descubrió · foto: la mano bebé y, si existe, la persona de internet que hace algo parecido")
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
        f.setMaximumHeight(205)                         # la galería no se come el espacio de la mano
        v.addWidget(f, 0)

        self.items_hito, self.pausado, self.nombres_musculos, self.n_diario = {}, False, [], None
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
                                           "Las fotos de sus habilidades quedan guardadas. ¿Seguir?")
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
            self.espejo.fijar(m["espejo"], m["espejo_nombre"], m["espejo_d"] if m["espejo"] is not None else None)
            self.decision.fijar(actual=m["estrategia"])
            nombre, _ = ESTRATEGIAS[m["estrategia"]]
            texto = f"<b>Intento {m['n']:,}</b> · decide <b>{nombre.lower()}</b>: {m['texto']}".replace(",", ".")
            if m["meta"] is not None:
                texto += f" · error {m['error']:.3f}"
            self.leyenda.setText(texto)
            self.mapa.update()
        elif t == "estado":
            self._estado(m)
        elif t == "habilidad":
            self._hito(m)
        elif t == "foto":
            it = self.items_hito.get(m["n"])
            if it:
                it.setIcon(QtGui.QIcon(QtGui.QPixmap(m["foto"])))
        elif t == "renacio":
            self.galeria.clear()
            self.items_hito = {}
        elif t == "aviso":
            self.leyenda.setText(m["texto"])
        elif t == "error":
            QtWidgets.QMessageBox.critical(self, "La mano bebé se detuvo", m["texto"][-1500:])

    def _hito(self, h):
        if h["n"] in self.items_hito:
            return
        texto = f"#{h['n']} · intento {h['intento']:,}".replace(",", ".")
        if h.get("parecido"):
            texto += f"\n≈ «{h['parecido']}»"
        it = QtWidgets.QListWidgetItem(texto)
        it.setToolTip(h["nombre"])
        it.setForeground(QtGui.QBrush(QtGui.QColor(VERDE)))
        if h.get("foto"):
            it.setIcon(QtGui.QIcon(QtGui.QPixmap(h["foto"])))
        self.galeria.insertItem(0, it)
        self.items_hito[h["n"]] = it
        while self.galeria.count() > 80:                    # la galería muestra las últimas; las fotos quedan en disco
            viejo = self.galeria.takeItem(self.galeria.count() - 1)
            self.items_hito = {k: v for k, v in self.items_hito.items() if v is not viejo}

    def _estado(self, e):
        nombre, _ = ETAPAS[e["etapa"]]
        self.chips["etapa"].setText(nombre)
        self.chips["intentos"].setText(f"{e['intentos']:,} intentos".replace(",", "."))
        self.chips["habilidades"].setText(f"★ {e['n_habilidades']:,} habilidades".replace(",", "."))
        self.chips["habilidades"].setToolTip(f"Poses que descubrió sola y ya sabe repetir. Practicando ahora: {e['candidatas']}. "
                                             f"Cosas que le salieron por casualidad y no logró repetir: {e['casualidades']}")
        self.chips["finura"].setText(f"finura {e['nivel']}")
        self.chips["finura"].setToolTip(f"Cuando ya no encuentra nada nuevo, distingue detalles más finos: "
                                        f"hoy dos poses son distintas si difieren en {e['radio']:.2f}")
        self.chips["r2"].setText("R² —" if e["r2_control"] != e["r2_control"] else f"R² {e['r2_control']:.3f}")
        self.chips["r2"].setToolTip(f"R² del control: la pose que logra frente a la que se propuso (300 poses de examen). "
                                    f"R² de su cuerpo (modelo directo): {e['r2_cuerpo']:.3f}")
        self.chips["espejo"].setText(f"≈ internet {e['parecidos']}")
        self.chips["espejo"].setToolTip("Habilidades suyas que también hacen personas reales en internet (HaGRID v2)")
        self.mapa.puntos, self.mapa.z_hab, self.mapa.z_cand = e["puntos"], e["z_hab"], e["z_cand"]
        self.decision.fijar(e["preferencias"])
        habs = e["habilidades"]
        self.tabla.setRowCount(len(habs))
        for r, h in enumerate(reversed(habs)):
            vals = [str(h["n"]), h["nombre"], f"{max(0.0, 1 - h['error'] / 0.3) * 100:.0f} %", h["parecido"] or "—"]
            for c, v in enumerate(vals):
                it = self.tabla.item(r, c)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    self.tabla.setItem(r, c, it)
                if it.text() != v:
                    it.setText(v)
                    it.setToolTip(v if c == 1 else "")
                    if c == 3:
                        it.setForeground(QtGui.QBrush(QtGui.QColor(C["acento"] if h["parecido"] else C["muted"])))
        for h in habs[-80:]:
            self._hito(h)
        ultimo = e["diario"][-1]["intento"] if e["diario"] else None
        if (len(e["diario"]), ultimo) != self.n_diario:
            self.n_diario = (len(e["diario"]), ultimo)
            self.diario.clear()
            for d in reversed(e["diario"]):
                it = QtWidgets.QListWidgetItem(f"{d['intento']:,} · {d['texto']}".replace(",", "."))
                if d["texto"].startswith("★"):
                    it.setForeground(QtGui.QBrush(QtGui.QColor(VERDE)))
                self.diario.addItem(it)
        c = e["curva"]
        if len(c):
            ok = ~np.isnan(c[:, 1])
            self.c_control.setData(c[ok, 0], np.clip(c[ok, 1], 0, 1))
            self.c_cuerpo.setData(c[ok, 0], np.clip(c[ok, 2], 0, 1))
            self.c_cob.setData(c[:, 0], c[:, 3])

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
