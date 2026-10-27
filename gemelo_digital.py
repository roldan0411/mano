"""
============================================================
 PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 GEMELO DIGITAL DE LA MANO — aplicación de escritorio
------------------------------------------------------------
 Roldán Rodríguez — Ing. CCIA
============================================================

Qué es:
  Una copia virtual de todo el sistema (antebrazo, electrodos,
  circuito, ESP32, servos, tendones y mano) donde la IA aprende
  sola, sin persona ni hardware. Lo que aprende aquí es la BASE
  que después se pasa al modelo físico.

Pestañas:
  1. Gemelo en vivo: la persona virtual hace gestos y ves la
     señal EMG, lo que detecta la red y cómo se mueve la mano 3D.
  2. Aprendizaje autónomo: la red neuronal (.h5) practica, se
     evalúa, refuerza lo que confunde y sube la dificultad sola.
  3. Aprendizaje por refuerzo: un agente aprende qué comando dar
     a cada servo para que la mano logre cada pose.

Instalar (una vez):
  pip install tensorflow numpy scipy matplotlib h5py PySide6

Usar:
  python gemelo_digital.py                       abre la app
  python gemelo_digital.py --h5 mi_red.h5        abre la app con esa red
  python gemelo_digital.py --sin-ventana --iteraciones 60
        aprende sin abrir ventana (sirve para dejarlo corriendo)
  python gemelo_digital.py --capturas carpeta    genera imágenes de las
        pestañas para el informe

Salidas (carpeta gemelo_salida/):
  red_base_gemelo.h5    red neuronal base (la mejor encontrada)
  modelo_red.h          esa red en C para el ESP32
  politica_servos.h5    política aprendida por refuerzo
  poses_aprendidas.h    comandos de servos para el ESP32
  historial_*.csv       registro del aprendizaje
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import ia_mano as ia
import simulador as sm

CARPETA = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(CARPETA, "gemelo_salida")
CREDITO = "Roldán Rodríguez — Ing. CCIA"
COLORES_DEDOS = ["#D85A30", "#534AB7", "#1D9E75", "#BA7517", "#D4537E"]


# ==================================================================
# Cinemática de la mano 3D
# ==================================================================
BASES = np.array([[-3.2, 2.2, -0.4], [-2.4, 8.0, 0], [-0.8, 8.3, 0], [0.8, 8.0, 0], [2.3, 7.4, 0]])
FALANGES = [[3.0, 2.4, 1.9], [4.0, 2.4, 1.8], [4.4, 2.8, 1.9], [4.1, 2.6, 1.8], [3.2, 2.0, 1.6]]
FLEX_MAX = np.radians([80, 100, 70])


def _rot_muneca(p, muneca):
    th = np.radians(55) * muneca  # +1 extensión (hacia el dorso), -1 flexión (hacia la palma)
    c, s = np.cos(th), np.sin(th)
    p = np.atleast_2d(p)
    return np.c_[p[:, 0], p[:, 1] * c - p[:, 2] * s, p[:, 1] * s + p[:, 2] * c]


def puntos_mano(dedos, muneca):
    """Devuelve (palma 4x3, lista de 5 polilíneas de dedos) en coordenadas 3D."""
    palma = np.array([[-3.4, 0.6, 0], [3.0, 0.6, 0], [3.0, 7.6, 0], [-3.4, 8.2, 0]])
    lineas = []
    for i in range(5):
        f = float(np.clip(dedos[i], 0, 1))
        p = [BASES[i]]
        if i == 0:  # pulgar: se opone hacia el centro de la palma y hacia abajo
            d0, d1 = np.array([-0.55, 0.83, 0.0]), np.array([0.75, 0.35, -0.55])
            acum = 0.0
            for k, L in enumerate(FALANGES[0]):
                acum = min(1.0, acum + f * 0.45)
                d = (1 - acum) * d0 + acum * d1
                p.append(p[-1] + L * d / np.linalg.norm(d))
        else:
            ang = 0.0
            for k, L in enumerate(FALANGES[i]):
                ang += f * FLEX_MAX[k]
                p.append(p[-1] + L * np.array([0, np.cos(ang), -np.sin(ang)]))
        lineas.append(_rot_muneca(np.array(p), muneca))
    return _rot_muneca(palma, muneca), lineas


class Mano3D:
    """Dibuja y actualiza la mano en un eje 3D de matplotlib (rápido: solo mueve datos)."""

    def __init__(self, ax, alpha=1.0, gris=False, titulo=None):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.ax = ax
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5, 17)
        ax.set_zlim(-12, 9)
        ax.set_box_aspect((12, 22, 21), zoom=1.35)
        ax.view_init(elev=16, azim=-32)
        ax.set_axis_off()
        if titulo:
            ax.set_title(titulo, fontsize=10)
        if not gris:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection as P3
            ax.add_collection3d(P3([np.array([[-2.8, -5, 0], [2.4, -5, 0], [2.4, -0.4, 0], [-2.8, -0.4, 0]])],
                                   facecolor="#D3D1C7", edgecolor="#888780", lw=1))
        color_palma = "#D3D1C7" if gris else "#F1EFE8"
        self.palma = Poly3DCollection([np.zeros((4, 3))], facecolor=color_palma,
                                      edgecolor="#5F5E5A" if not gris else "#B4B2A9",
                                      alpha=0.9 * alpha, lw=1)
        ax.add_collection3d(self.palma)
        self.lineas = []
        for i in range(5):
            c = "#B4B2A9" if gris else COLORES_DEDOS[i]
            (ln,) = ax.plot([], [], [], color=c, lw=6 if not gris else 4, alpha=alpha,
                            solid_capstyle="round", marker="o", markersize=4 if not gris else 0,
                            markerfacecolor="white")
            self.lineas.append(ln)

    def actualizar(self, dedos, muneca):
        palma, lineas = puntos_mano(dedos, muneca)
        self.palma.set_verts([palma])
        for ln, p in zip(self.lineas, lineas):
            ln.set_data_3d(p[:, 0], p[:, 1], p[:, 2])


# ==================================================================
# Modo sin ventana
# ==================================================================
def guardar_historial(ruta, filas):
    if not filas:
        return
    claves = [k for k in filas[0] if k not in ("recall", "eventos")]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(claves + [f"recall_{g}" for g in ia.GESTOS] * ("recall" in filas[0]))
        for r in filas:
            fila = [r[k] for k in claves]
            if "recall" in r:
                fila += [round(float(v), 4) for v in r["recall"]]
            w.writerow(fila)


def modo_sin_ventana(args):
    os.makedirs(SALIDA, exist_ok=True)
    modelo = ia.cargar_red(args.h5) if args.h5 else None
    if modelo is not None:
        print(f"Red cargada: {args.h5} ({ia.contar_parametros(modelo)} parámetros)")
    ap = sm.AprendizajeAutonomo(modelo, semilla=args.semilla,
                                ruta_mejor=os.path.join(SALIDA, "red_base_gemelo.h5"),
                                ruta_c=os.path.join(SALIDA, "modelo_red.h"))
    print("\n== Aprendizaje autónomo de la red neuronal ==")
    t0 = time.time()
    for _ in range(args.iteraciones):
        r = ap.iteracion_completa()
        print(f"  it {r['iteracion']:3d} | nivel {r['nivel']} | precisión {r['precision']*100:5.1f} % "
              f"| robustez {r['robustez']*100:5.1f} % | más débil: {r['peor_gesto']}")
        for e in r["eventos"]:
            print(f"        -> {e}")
        if r["terminado"]:
            break
    guardar_historial(os.path.join(SALIDA, "historial_red.csv"), ap.historial)
    print(f"  Listo en {time.time() - t0:.0f} s. Mejor robustez: {ap.mejor_robustez*100:.1f} %")

    print("\n== Aprendizaje por refuerzo de los servos ==")
    ag = sm.AgenteServos(sm.ManoFisica(), semilla=args.semilla)
    for i in range(args.iteraciones_refuerzo):
        r = ag.iteracion_completa()
        if i % 100 == 0 or i == args.iteraciones_refuerzo - 1:
            print(f"  it {r['iteracion']:4d} | recompensa {r['recompensa']:.4f} | error de pose {r['error_medio']:.4f}")
    ag.guardar(os.path.join(SALIDA, "politica_servos.h5"))
    ia.exportar_poses_c(ag.poses(), os.path.join(SALIDA, "poses_aprendidas.h"))
    guardar_historial(os.path.join(SALIDA, "historial_refuerzo.csv"), ag.historial)
    print(f"\nArchivos en {SALIDA}:")
    for f in sorted(os.listdir(SALIDA)):
        print("  ", f)


# ==================================================================
# Aplicación de escritorio
# ==================================================================
def lanzar_app(args):
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Lienzo
    from matplotlib.figure import Figure
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt

    matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.spines.top": False,
                                "axes.spines.right": False})

    # ---------------- hilos de trabajo ----------------
    class HiloAutonomo(QtCore.QThread):
        progreso = QtCore.Signal(object)
        fin = QtCore.Signal(str)

        def __init__(self, aprendiz, maximo, rondas, epocas):
            super().__init__()
            self.ap, self.maximo, self.rondas, self.epocas = aprendiz, maximo, rondas, epocas
            self.detener = False

        def run(self):
            try:
                hechas = 0
                while not self.detener and hechas < self.maximo and not self.ap.terminado:
                    r = self.ap.iteracion_completa(rondas=self.rondas, epocas=self.epocas)
                    hechas += 1
                    self.progreso.emit((r, ia.RedNumpy(self.ap.modelo), self.ap.confusion.copy()))
                self.fin.emit("terminado" if self.ap.terminado else "pausado")
            except Exception as e:  # se muestra en la ventana
                self.fin.emit(f"error: {e}")

    class HiloRefuerzo(QtCore.QThread):
        progreso = QtCore.Signal(object)
        fin = QtCore.Signal(str)

        def __init__(self, agente, maximo):
            super().__init__()
            self.ag, self.maximo = agente, maximo
            self.detener = False

        def run(self):
            try:
                for i in range(self.maximo):
                    if self.detener:
                        break
                    r = self.ag.iteracion_completa()
                    if i % 10 == 0 or i == self.maximo - 1:
                        self.progreso.emit((r, self.ag.poses()))
                self.fin.emit("pausado")
            except Exception as e:
                self.fin.emit(f"error: {e}")

    def deslizador(minimo, maximo, valor, escala=100):
        s = QtWidgets.QSlider(Qt.Horizontal)
        s.setRange(int(minimo * escala), int(maximo * escala))
        s.setValue(int(valor * escala))
        s.escala = escala
        return s

    def grupo(titulo):
        g = QtWidgets.QGroupBox(titulo)
        return g, QtWidgets.QVBoxLayout(g)

    # ---------------- ventana principal ----------------
    class Ventana(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Gemelo digital — Prótesis mioeléctrica · {CREDITO}")
            self.resize(1380, 860)
            os.makedirs(SALIDA, exist_ok=True)
            self.modelo = None
            self.ruta_modelo = None
            self.info_modelo = {}
            self.aprendiz = None
            self.agente = None
            self.hilo_auto = None
            self.hilo_ref = None
            self.gemelo = sm.GemeloEnVivo(sm.Condiciones(), semilla=1)
            self.gesto_usuario = "reposo"
            self.t_secuencia = 0.0
            self.poses_rl = None

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            v = QtWidgets.QVBoxLayout(central)
            v.addLayout(self._barra_superior())
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.addTab(self._tab_vivo(), "1 · Gemelo en vivo")
            self.tabs.addTab(self._tab_autonomo(), "2 · Aprendizaje autónomo (red neuronal)")
            self.tabs.addTab(self._tab_refuerzo(), "3 · Aprendizaje por refuerzo (servos)")
            v.addWidget(self.tabs)
            pie = QtWidgets.QLabel(f"{CREDITO}  ·  Ingeniería en Ciencias Computacionales e IA")
            pie.setStyleSheet("color:#5F5E5A; font-size:11px;")
            v.addWidget(pie)
            self.statusBar().showMessage("Listo. Carga una red .h5 o inicia el aprendizaje autónomo.")

            self.reloj = QtCore.QTimer(self)
            self.reloj.timeout.connect(self._tic)
            self.reloj.start(100)
            if args.h5:
                self._cargar_h5(args.h5)

        # ---------- barra superior ----------
        def _barra_superior(self):
            h = QtWidgets.QHBoxLayout()
            self.b_cargar = QtWidgets.QPushButton("Cargar red .h5…")
            self.b_nueva = QtWidgets.QPushButton("Red nueva")
            self.b_guardar = QtWidgets.QPushButton("Guardar red .h5…")
            self.b_exportar = QtWidgets.QPushButton("Exportar al ESP32…")
            self.b_cargar.clicked.connect(lambda: self._cargar_h5())
            self.b_nueva.clicked.connect(self._red_nueva)
            self.b_guardar.clicked.connect(self._guardar_h5)
            self.b_exportar.clicked.connect(self._exportar)
            for b in (self.b_cargar, self.b_nueva, self.b_guardar, self.b_exportar):
                h.addWidget(b)
            self.lbl_modelo = QtWidgets.QLabel()
            self.lbl_modelo.setStyleSheet("padding-left:12px;")
            h.addWidget(self.lbl_modelo, 1)
            self._actualizar_etiqueta_modelo()
            return h

        def _actualizar_etiqueta_modelo(self):
            if self.modelo is None:
                self.lbl_modelo.setText("<b>Red:</b> ninguna — carga un .h5 o inicia el aprendizaje autónomo")
                return
            nombre = os.path.basename(self.ruta_modelo) if self.ruta_modelo else "red del gemelo (sin guardar)"
            extra = ""
            if self.aprendiz and self.aprendiz.historial:
                r = self.aprendiz.historial[-1]
                extra = f" · nivel {r['nivel']} · robustez {r['robustez']*100:.1f} %"
            elif self.info_modelo.get("precision_robustez"):
                extra = f" · robustez {self.info_modelo['precision_robustez']*100:.1f} %"
            self.lbl_modelo.setText(f"<b>Red:</b> {nombre} · 8-32-16-7 · "
                                    f"{ia.contar_parametros(self.modelo)} parámetros{extra}")

        def _ocupado(self):
            if self.hilo_auto and self.hilo_auto.isRunning():
                QtWidgets.QMessageBox.information(self, "Gemelo digital",
                                                  "Pausa el aprendizaje autónomo primero.")
                return True
            return False

        def _cargar_h5(self, ruta=None):
            if self._ocupado():
                return
            if ruta is None:
                ruta, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Cargar red neuronal", CARPETA,
                                                                "Red Keras (*.h5)")
                if not ruta:
                    return
            try:
                modelo = ia.cargar_red(ruta)
                red = ia.RedNumpy(modelo)
            except ia.ErrorModelo as e:
                QtWidgets.QMessageBox.warning(self, "No se pudo cargar la red", str(e))
                return
            self.modelo, self.ruta_modelo = modelo, ruta
            self.info_modelo = ia.leer_metadatos(ruta)
            self.aprendiz = None
            self.gemelo.fijar_red(red)
            self._limpiar_graficas_autonomo()
            self._actualizar_etiqueta_modelo()
            self.statusBar().showMessage(f"Red cargada: {ruta}. El aprendizaje autónomo continúa desde ella.")

        def _red_nueva(self):
            if self._ocupado():
                return
            self.modelo, self.ruta_modelo, self.info_modelo, self.aprendiz = None, None, {}, None
            self.gemelo.fijar_red(None)
            self._limpiar_graficas_autonomo()
            self._actualizar_etiqueta_modelo()
            self.statusBar().showMessage("Red nueva: se crea en la primera iteración del aprendizaje autónomo.")

        def _guardar_h5(self):
            if self._ocupado():
                return
            if self.modelo is None:
                QtWidgets.QMessageBox.information(self, "Gemelo digital", "Todavía no hay red.")
                return
            ruta, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Guardar red neuronal", os.path.join(SALIDA, "red_base_gemelo.h5"), "Red Keras (*.h5)")
            if not ruta:
                return
            if not ruta.endswith(".h5"):
                ruta += ".h5"
            meta = {"origen": "gemelo digital"}
            if self.aprendiz and self.aprendiz.historial:
                r = self.aprendiz.historial[-1]
                meta.update({"iteracion": r["iteracion"], "nivel": r["nivel"],
                             "precision_robustez": round(r["robustez"], 4)})
            ia.guardar_red(self.modelo, ruta, meta)
            self.ruta_modelo = ruta
            self._actualizar_etiqueta_modelo()
            self.statusBar().showMessage(f"Red guardada en {ruta}")

        def _exportar(self):
            if self._ocupado():
                return
            carpeta = QtWidgets.QFileDialog.getExistingDirectory(self, "Carpeta del sketch protesis_tiempo_real",
                                                                 CARPETA)
            if not carpeta:
                return
            hechos = []
            if self.modelo is not None:
                ia.exportar_red_c(self.modelo, os.path.join(carpeta, "modelo_red.h"))
                hechos.append("modelo_red.h (red neuronal)")
            if self.agente is not None and self.agente.iteracion > 0:
                ia.exportar_poses_c(self.agente.poses(), os.path.join(carpeta, "poses_aprendidas.h"))
                hechos.append("poses_aprendidas.h (comandos aprendidos por refuerzo)")
            if not hechos:
                QtWidgets.QMessageBox.information(self, "Exportar", "No hay nada entrenado todavía.")
                return
            QtWidgets.QMessageBox.information(self, "Exportado al ESP32",
                                              "Se escribieron en " + carpeta + ":\n\n• " + "\n• ".join(hechos))

        # ---------- pestaña 1: gemelo en vivo ----------
        def _tab_vivo(self):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            izq = QtWidgets.QVBoxLayout()
            self.fig_mano = Figure(figsize=(5.2, 5.6))
            self.fig_mano.subplots_adjust(0, 0, 1, 1)
            ax = self.fig_mano.add_subplot(111, projection="3d")
            self.mano3d = Mano3D(ax)
            self.mano3d.actualizar(np.full(5, 0.2), 0.0)
            self.lienzo_mano = Lienzo(self.fig_mano)
            izq.addWidget(self.lienzo_mano, 1)
            self.lbl_usuario = QtWidgets.QLabel()
            self.lbl_red = QtWidgets.QLabel()
            self.lbl_prec = QtWidgets.QLabel()
            for l in (self.lbl_usuario, self.lbl_red, self.lbl_prec):
                l.setStyleSheet("font-size:15px; padding:2px;")
                izq.addWidget(l)
            h.addLayout(izq, 5)

            centro = QtWidgets.QVBoxLayout()
            self.fig_emg = Figure(figsize=(5.4, 5.6))
            self.ax_c1, self.ax_c2, self.ax_p = self.fig_emg.subplots(
                3, 1, gridspec_kw={"height_ratios": [1, 1, 1.2]})
            self.fig_emg.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.12, hspace=0.6)
            t = np.arange(-2000, 0) / 1000
            (self.l1,) = self.ax_c1.plot(t, np.full(2000, 2048), lw=0.6, color="#185FA5")
            (self.l2,) = self.ax_c2.plot(t, np.full(2000, 2048), lw=0.6, color="#0F6E56")
            for a, tit in ((self.ax_c1, "Electrodo F3 · centro de los flexores"), (self.ax_c2, "Electrodo E3 · centro de los extensores")):
                a.set_ylim(0, 4095)
                a.set_xlim(-2, 0)
                a.set_title(tit, loc="left")
                a.set_ylabel("ADC")
            self.ax_c2.set_xlabel("tiempo (s)")
            self.barras = self.ax_p.bar(range(ia.NUM_GESTOS), np.zeros(ia.NUM_GESTOS), color="#B4B2A9")
            self.ax_p.axhline(ia.UMBRAL_CONFIANZA, color="#A32D2D", lw=0.8, ls="--")
            self.ax_p.set_ylim(0, 1)
            self.ax_p.set_xticks(range(ia.NUM_GESTOS),
                                 [ia.NOMBRES[g].replace(" muñeca", "\nmuñeca") for g in ia.GESTOS])
            self.ax_p.set_title("Salida de la red (probabilidad por gesto) · umbral 60 %", loc="left")
            self.lienzo_emg = Lienzo(self.fig_emg)
            centro.addWidget(self.lienzo_emg)
            h.addLayout(centro, 5)

            der = QtWidgets.QVBoxLayout()
            g, gl = grupo("Gesto que hace la persona virtual")
            self.botones_gesto = QtWidgets.QButtonGroup(self)
            rejilla = QtWidgets.QGridLayout()
            for i, gesto in enumerate(ia.GESTOS):
                b = QtWidgets.QPushButton(ia.NOMBRES[gesto])
                b.setCheckable(True)
                b.clicked.connect(lambda _=False, x=gesto: self._fijar_gesto(x))
                self.botones_gesto.addButton(b, i)
                rejilla.addWidget(b, i // 2, i % 2)
            self.botones_gesto.button(0).setChecked(True)
            gl.addLayout(rejilla)
            self.chk_auto = QtWidgets.QCheckBox("Secuencia automática (cambia cada 2.5 s)")
            self.chk_auto.setChecked(True)
            gl.addWidget(self.chk_auto)
            der.addWidget(g)

            g, gl = grupo("Condiciones del antebrazo y los electrodos")
            self.s_ruido = deslizador(0.5, 5, 1.0)
            self.s_60 = deslizador(0, 30, 3, escala=1)
            self.s_desp = deslizador(0, 1, 0.1)
            self.s_fatiga = deslizador(0, 1, 0.0)
            self.s_var = deslizador(0, 1, 0.2)
            for nombre, s in (("Ruido de electrodos (×)", self.s_ruido), ("Interferencia 60 Hz (µV)", self.s_60),
                              ("Desplazamiento de electrodos", self.s_desp), ("Fatiga muscular", self.s_fatiga),
                              ("Diferencia entre personas", self.s_var)):
                gl.addWidget(QtWidgets.QLabel(nombre))
                gl.addWidget(s)
                s.valueChanged.connect(self._condiciones)
            fila = QtWidgets.QHBoxLayout()
            b1 = QtWidgets.QPushButton("Nueva persona")
            b2 = QtWidgets.QPushButton("Recolocar electrodos")
            b1.clicked.connect(lambda: (self.gemelo.antebrazo.nuevo_usuario(), self._reset_precision()))
            b2.clicked.connect(lambda: (self.gemelo.antebrazo.recolocar_electrodos(), self._reset_precision()))
            fila.addWidget(b1)
            fila.addWidget(b2)
            gl.addLayout(fila)
            der.addWidget(g)

            g, gl = grupo("Comandos de los servos")
            self.cmb_poses = QtWidgets.QComboBox()
            self.cmb_poses.addItems(["Tabla fija (la del ESP32)", "Aprendidos por refuerzo (pestaña 3)"])
            self.cmb_poses.currentIndexChanged.connect(self._fuente_poses)
            gl.addWidget(self.cmb_poses)
            self.lbl_error_pose = QtWidgets.QLabel()
            gl.addWidget(self.lbl_error_pose)
            der.addWidget(g)
            self.b_pausa_vivo = QtWidgets.QPushButton("Pausar simulación")
            self.b_pausa_vivo.setCheckable(True)
            der.addWidget(self.b_pausa_vivo)
            der.addStretch(1)
            h.addLayout(der, 3)
            return w

        def _fijar_gesto(self, gesto):
            self.gesto_usuario = gesto
            self.botones_gesto.button(ia.GESTOS.index(gesto)).setChecked(True)
            self.t_secuencia = 0.0

        def _condiciones(self):
            c = sm.Condiciones(self.s_ruido.value() / 100, float(self.s_60.value()), self.s_desp.value() / 100,
                               self.s_fatiga.value() / 100, self.s_var.value() / 100)
            ant = self.gemelo.antebrazo.cond
            self.gemelo.antebrazo.cambiar_condiciones(replace(c, variabilidad=ant.variabilidad))
            self.gemelo.antebrazo.cond.variabilidad = c.variabilidad  # se aplica con "Nueva persona"
            self._reset_precision()

        def _reset_precision(self):
            self.gemelo.aciertos = []

        def _fuente_poses(self):
            if self.cmb_poses.currentIndex() == 1:
                if self.poses_rl is None:
                    QtWidgets.QMessageBox.information(self, "Gemelo digital",
                                                      "Primero entrena el agente en la pestaña 3.")
                    self.cmb_poses.setCurrentIndex(0)
                    return
                self.gemelo.poses = dict(self.poses_rl)
            else:
                self.gemelo.poses = dict(ia.POSES_BASE)

        def _tic(self):
            if self.tabs.currentIndex() != 0 or self.b_pausa_vivo.isChecked():
                return
            self.t_secuencia += 0.1
            if self.chk_auto.isChecked() and self.t_secuencia >= 2.5:
                otros = [g for g in ia.GESTOS if g != self.gesto_usuario]
                self._fijar_gesto(str(np.random.choice(otros)))
            e = self.gemelo.paso(self.gesto_usuario, 0.1)
            self._dibujar_vivo(e)

        def _dibujar_vivo(self, e):
            self.mano3d.actualizar(e["dedos"], e["muneca"])
            self.lienzo_mano.draw_idle()
            s = self.gemelo.senal
            self.l1.set_ydata(s[:, ia.CANALES.index("F3")])
            self.l2.set_ydata(s[:, ia.CANALES.index("E3")])
            for i, b in enumerate(self.barras):
                b.set_height(e["probs"][i])
                b.set_color("#534AB7" if ia.GESTOS[i] == e["gesto_red"] else "#B4B2A9")
            self.lienzo_emg.draw_idle()
            ok = e["gesto_red"] == self.gesto_usuario
            self.lbl_usuario.setText(f"La persona hace: <b>{ia.NOMBRES[self.gesto_usuario]}</b>")
            if self.gemelo.red is None:
                self.lbl_red.setText("La red detecta: <i>sin red — carga un .h5 o inicia el aprendizaje</i>")
            else:
                color = "#0F6E56" if ok else "#A32D2D"
                self.lbl_red.setText(f"La red detecta: <b style='color:{color}'>{ia.NOMBRES[e['gesto_red']]}</b>"
                                     f" · confianza {e['confianza']*100:.0f} %")
            p = e["precision_vivo"]
            self.lbl_prec.setText(f"Acierto en vivo (últimos 10 s): <b>{p*100:.0f} %</b>" if p is not None else "")
            err = sm.error_pose(e["dedos"], e["muneca"], e["gesto_red"])
            self.lbl_error_pose.setText(f"Error de la pose de la mano: {err:.3f}")

        # ---------- pestaña 2: aprendizaje autónomo ----------
        def _tab_autonomo(self):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            izq = QtWidgets.QVBoxLayout()
            g, gl = grupo("Control")
            self.b_auto = QtWidgets.QPushButton("▶  Iniciar aprendizaje autónomo")
            self.b_auto.setStyleSheet("font-weight:bold; padding:8px;")
            self.b_auto.clicked.connect(self._alternar_autonomo)
            gl.addWidget(self.b_auto)
            form = QtWidgets.QFormLayout()
            self.sp_max = QtWidgets.QSpinBox()
            self.sp_max.setRange(1, 1000)
            self.sp_max.setValue(60)
            self.sp_meta = QtWidgets.QSpinBox()
            self.sp_meta.setRange(50, 100)
            self.sp_meta.setValue(95)
            self.sp_meta.setSuffix(" %")
            self.sp_rondas = QtWidgets.QSpinBox()
            self.sp_rondas.setRange(1, 10)
            self.sp_rondas.setValue(3)
            self.sp_epocas = QtWidgets.QSpinBox()
            self.sp_epocas.setRange(1, 20)
            self.sp_epocas.setValue(3)
            form.addRow("Iteraciones por tanda", self.sp_max)
            form.addRow("Meta para subir de nivel", self.sp_meta)
            form.addRow("Rondas simuladas / iteración", self.sp_rondas)
            form.addRow("Épocas / iteración", self.sp_epocas)
            gl.addLayout(form)
            nota = QtWidgets.QLabel("El mejor modelo se guarda solo en\ngemelo_salida/red_base_gemelo.h5\n"
                                    "y gemelo_salida/modelo_red.h")
            nota.setStyleSheet("color:#5F5E5A;")
            gl.addWidget(nota)
            izq.addWidget(g)
            g, gl = grupo("Cómo decide solo")
            exp = QtWidgets.QLabel(
                "1. Practica: simula 2 personas nuevas haciendo los 7 gestos,\n"
                "   con más repeticiones de los que más confunde.\n"
                "2. Entrena: unas épocas sobre lo nuevo + lo anterior.\n"
                "3. Se evalúa con personas que nunca vio.\n"
                "4. Si domina el nivel o se estanca, sube la dificultad\n"
                "   (más ruido, fatiga y electrodos movidos).\n"
                "5. Si mejora la robustez, guarda el mejor .h5.\n"
                "Termina al estancarse en el nivel 5.")
            exp.setStyleSheet("font-size:11px;")
            gl.addWidget(exp)
            izq.addWidget(g)
            self.log_auto = QtWidgets.QPlainTextEdit()
            self.log_auto.setReadOnly(True)
            self.log_auto.setStyleSheet("font-family:monospace; font-size:11px;")
            izq.addWidget(self.log_auto, 1)
            h.addLayout(izq, 3)

            self.fig_auto = Figure(figsize=(8, 6))
            gs = self.fig_auto.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.45, wspace=0.3,
                                            left=0.07, right=0.98, top=0.94, bottom=0.12)
            self.ax_curva = self.fig_auto.add_subplot(gs[0, :])
            self.ax_recall = self.fig_auto.add_subplot(gs[1, 0])
            self.ax_conf = self.fig_auto.add_subplot(gs[1, 1])
            self.lienzo_auto = Lienzo(self.fig_auto)
            h.addWidget(self.lienzo_auto, 7)
            self._limpiar_graficas_autonomo()
            return w

        def _limpiar_graficas_autonomo(self):
            if not hasattr(self, "ax_curva"):
                return
            for a in (self.ax_curva, self.ax_recall, self.ax_conf):
                a.clear()
            self.ax_curva.set_title("Precisión con personas nuevas, por iteración", loc="left")
            self.ax_curva.set_ylim(0, 100)
            self.ax_curva.set_xlabel("iteración")
            self.ax_curva.set_ylabel("%")
            self.ax_recall.set_title("Acierto por gesto (nivel actual)", loc="left")
            self.ax_conf.set_title("Matriz de confusión (nivel actual)", loc="left")
            self.ax_recall.set_axis_off()
            self.ax_conf.set_axis_off()
            if hasattr(self, "lienzo_auto"):
                self.lienzo_auto.draw_idle()

        def _alternar_autonomo(self):
            if self.hilo_auto and self.hilo_auto.isRunning():
                self.hilo_auto.detener = True
                self.b_auto.setEnabled(False)
                self.b_auto.setText("Terminando la iteración en curso…")
                return
            if self.aprendiz is None or self.aprendiz.terminado:
                if self.aprendiz is not None and self.aprendiz.terminado:
                    self.aprendiz.terminado = False
                    self.aprendiz.sin_mejora = 0
                else:
                    self.aprendiz = sm.AprendizajeAutonomo(
                        self.modelo, semilla=int(time.time()) % 10000,
                        ruta_mejor=os.path.join(SALIDA, "red_base_gemelo.h5"),
                        ruta_c=os.path.join(SALIDA, "modelo_red.h"))
            self.aprendiz.meta = self.sp_meta.value() / 100
            self.hilo_auto = HiloAutonomo(self.aprendiz, self.sp_max.value(), self.sp_rondas.value(),
                                          self.sp_epocas.value())
            self.hilo_auto.progreso.connect(self._progreso_autonomo)
            self.hilo_auto.fin.connect(self._fin_autonomo)
            for b in (self.b_cargar, self.b_nueva, self.b_guardar, self.b_exportar):
                b.setEnabled(False)
            self.b_auto.setText("⏸  Pausar")
            self._log(f"Inicio de tanda ({self.sp_max.value()} iteraciones máx.)"
                      + (" desde la red cargada" if self.modelo is not None and not self.aprendiz.historial else ""))
            self.hilo_auto.start()

        def _log(self, texto):
            self.log_auto.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {texto}")

        def _progreso_autonomo(self, datos):
            r, red, cm = datos
            self.modelo = self.aprendiz.modelo
            self.gemelo.fijar_red(red)
            self._log(f"it {r['iteracion']:3d} · nivel {r['nivel']} · precisión {r['precision']*100:5.1f} % · "
                      f"robustez {r['robustez']*100:5.1f} % · más débil: {ia.NOMBRES[r['peor_gesto']]}")
            for e in r["eventos"]:
                self._log("   → " + e)
            self._actualizar_etiqueta_modelo()
            hist = self.aprendiz.historial
            it = [x["iteracion"] for x in hist]
            a = self.ax_curva
            a.clear()
            niveles = [x["nivel"] for x in hist]
            colores = ["#E1F5EE", "#EEEDFE", "#FAEEDA", "#FBEAF0", "#F1EFE8"]
            ini = 0
            for k in range(1, len(niveles) + 1):
                if k == len(niveles) or niveles[k] != niveles[ini]:
                    a.axvspan(it[ini] - 0.5, it[k - 1] + 0.5, color=colores[niveles[ini] - 1], lw=0)
                    a.text((it[ini] + it[k - 1]) / 2, 3, f"N{niveles[ini]}", ha="center", fontsize=8, color="#444441")
                    ini = k
            a.plot(it, [x["precision"] * 100 for x in hist], "-o", ms=3, color="#534AB7", label="nivel actual")
            a.plot(it, [x["robustez"] * 100 for x in hist], "-o", ms=3, color="#D85A30",
                   label="robustez (nivel 5, el más difícil)")
            a.axhline(self.aprendiz.meta * 100, color="#5F5E5A", ls=":", lw=0.8)
            a.set_ylim(0, 100)
            a.set_xlabel("iteración")
            a.set_ylabel("%")
            a.set_title("Precisión con personas nuevas, por iteración (fondo = nivel de dificultad)", loc="left")
            a.legend(loc="lower right", fontsize=8)
            b = self.ax_recall
            b.clear()
            b.barh(range(ia.NUM_GESTOS), r["recall"] * 100,
                   color=["#A32D2D" if g == r["peor_gesto"] else "#1D9E75" for g in ia.GESTOS])
            b.set_yticks(range(ia.NUM_GESTOS), [ia.NOMBRES[g].replace(" muñeca", "") for g in ia.GESTOS])
            b.invert_yaxis()
            b.set_xlim(0, 100)
            b.set_title(f"Acierto por gesto (nivel {r['nivel']})", loc="left")
            c = self.ax_conf
            c.clear()
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            c.imshow(cmn, cmap="Purples", vmin=0, vmax=1)
            etiquetas = ["Rep", "Abi", "Puñ", "Pin", "Señ", "Flx", "Ext"]
            c.set_xticks(range(7), etiquetas)
            c.set_yticks(range(7), etiquetas)
            c.set_xlabel("detectado")
            c.set_ylabel("real")
            for i in range(7):
                for j in range(7):
                    if cmn[i, j] >= 0.05:
                        c.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center", fontsize=7,
                               color="white" if cmn[i, j] > 0.5 else "#26215C")
            c.set_title(f"Matriz de confusión (nivel {r['nivel']}, %)", loc="left")
            self.lienzo_auto.draw_idle()

        def _fin_autonomo(self, estado):
            guardar_historial(os.path.join(SALIDA, "historial_red.csv"), self.aprendiz.historial)
            for b in (self.b_cargar, self.b_nueva, self.b_guardar, self.b_exportar, self.b_auto):
                b.setEnabled(True)
            if estado.startswith("error"):
                self._log("ERROR: " + estado)
                QtWidgets.QMessageBox.warning(self, "Aprendizaje autónomo", estado)
            elif estado == "terminado":
                self._log(f"Aprendizaje terminado. Mejor robustez: {self.aprendiz.mejor_robustez*100:.1f} %. "
                          "Archivos en gemelo_salida/.")
            else:
                self._log("Pausado.")
            self.b_auto.setText("▶  Continuar aprendizaje autónomo" if estado != "terminado"
                                else "▶  Seguir afinando")
            self.ruta_modelo = None
            self._actualizar_etiqueta_modelo()

        # ---------- pestaña 3: refuerzo ----------
        def _tab_refuerzo(self):
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            izq = QtWidgets.QVBoxLayout()
            g, gl = grupo("Mano física virtual (lo que el agente NO conoce)")
            self.s_acople = deslizador(0, 0.5, 0.22)
            self.s_desfase = deslizador(-0.2, 0.2, 0.08)
            self.s_holgura = deslizador(0, 0.2, 0.09)
            for nombre, s in (("Acople muñeca → dedos (tendones)", self.s_acople),
                              ("Desfase del disco del servo de muñeca", self.s_desfase),
                              ("Holgura de los tendones", self.s_holgura)):
                gl.addWidget(QtWidgets.QLabel(nombre))
                gl.addWidget(s)
            fila = QtWidgets.QHBoxLayout()
            b_apl = QtWidgets.QPushButton("Aplicar")
            b_ale = QtWidgets.QPushButton("Mano aleatoria")
            b_apl.clicked.connect(self._aplicar_mano)
            b_ale.clicked.connect(self._mano_aleatoria)
            fila.addWidget(b_apl)
            fila.addWidget(b_ale)
            gl.addLayout(fila)
            izq.addWidget(g)
            g, gl = grupo("Agente")
            self.b_ref = QtWidgets.QPushButton("▶  Iniciar aprendizaje por refuerzo")
            self.b_ref.setStyleSheet("font-weight:bold; padding:8px;")
            self.b_ref.clicked.connect(self._alternar_refuerzo)
            gl.addWidget(self.b_ref)
            form = QtWidgets.QFormLayout()
            self.sp_ref = QtWidgets.QSpinBox()
            self.sp_ref.setRange(10, 20000)
            self.sp_ref.setValue(800)
            form.addRow("Iteraciones por tanda", self.sp_ref)
            gl.addLayout(form)
            for texto, fn in (("Guardar política .h5…", self._guardar_politica),
                              ("Cargar política .h5…", self._cargar_politica),
                              ("Exportar poses_aprendidas.h…", self._exportar_poses)):
                b = QtWidgets.QPushButton(texto)
                b.clicked.connect(fn)
                gl.addWidget(b)
            izq.addWidget(g)
            g, gl = grupo("Ver gesto")
            self.cmb_ver = QtWidgets.QComboBox()
            self.cmb_ver.addItems([ia.NOMBRES[x] for x in ia.GESTOS])
            self.cmb_ver.currentIndexChanged.connect(self._dibujar_refuerzo_mano)
            gl.addWidget(self.cmb_ver)
            izq.addWidget(g)
            self.tabla = QtWidgets.QTableWidget(ia.NUM_GESTOS, 6)
            self.tabla.setHorizontalHeaderLabels(["Pulgar", "Índice", "Medio", "Anular", "Meñique", "Muñeca"])
            self.tabla.setVerticalHeaderLabels([ia.NOMBRES[x] for x in ia.GESTOS])
            self.tabla.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            self.tabla.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            izq.addWidget(QtWidgets.QLabel("Comandos aprendidos (los que irían al ESP32):"))
            izq.addWidget(self.tabla, 1)
            h.addLayout(izq, 4)

            self.fig_ref = Figure(figsize=(8, 6))
            gs = self.fig_ref.add_gridspec(2, 2, hspace=0.4, wspace=0.15, left=0.07, right=0.98,
                                           top=0.94, bottom=0.1)
            self.ax_rec = self.fig_ref.add_subplot(gs[0, 0])
            self.ax_err = self.fig_ref.add_subplot(gs[1, 0])
            self.ax_mano_ref = self.fig_ref.add_subplot(gs[:, 1], projection="3d")
            self.mano_obj = Mano3D(self.ax_mano_ref, alpha=0.35, gris=True)
            self.mano_ref = Mano3D(self.ax_mano_ref, titulo="Gris: pose objetivo · Color: lo que logra")
            self.lienzo_ref = Lienzo(self.fig_ref)
            h.addWidget(self.lienzo_ref, 7)
            self._aplicar_mano()
            return w

        def _params_mano(self):
            p = sm.ParametrosMano()
            k = self.s_acople.value() / 100
            hol = self.s_holgura.value() / 100
            return replace(p, acople_muneca=tuple(np.array(p.acople_muneca) / 0.22 * k),
                           desfase_muneca=self.s_desfase.value() / 100,
                           holgura=tuple(np.clip(np.array(p.holgura) / 0.09 * hol, 0, 0.3)))

        def _aplicar_mano(self, params=None):
            if self.hilo_ref and self.hilo_ref.isRunning():
                QtWidgets.QMessageBox.information(self, "Gemelo digital", "Pausa el agente primero.")
                return
            self.mano_params = params or self._params_mano()
            self.gemelo.mano.p = self.mano_params
            if self.agente is not None:
                self.agente.mano = sm.ManoFisica(self.mano_params)
            self._dibujar_refuerzo()

        def _mano_aleatoria(self):
            self._aplicar_mano(sm.ParametrosMano.aleatoria(np.random.default_rng()))

        def _alternar_refuerzo(self):
            if self.hilo_ref and self.hilo_ref.isRunning():
                self.hilo_ref.detener = True
                return
            if self.agente is None:
                self.agente = sm.AgenteServos(sm.ManoFisica(self.mano_params), semilla=0)
            self.hilo_ref = HiloRefuerzo(self.agente, self.sp_ref.value())
            self.hilo_ref.progreso.connect(self._progreso_refuerzo)
            self.hilo_ref.fin.connect(self._fin_refuerzo)
            self.b_ref.setText("⏸  Pausar")
            self.hilo_ref.start()

        def _progreso_refuerzo(self, datos):
            r, poses = datos
            self.poses_rl = poses
            if self.cmb_poses.currentIndex() == 1:
                self.gemelo.poses = dict(poses)
            self._dibujar_refuerzo()

        def _fin_refuerzo(self, estado):
            self.b_ref.setText("▶  Continuar aprendizaje por refuerzo")
            if estado.startswith("error"):
                QtWidgets.QMessageBox.warning(self, "Aprendizaje por refuerzo", estado)
            elif self.agente is not None:
                guardar_historial(os.path.join(SALIDA, "historial_refuerzo.csv"), self.agente.historial)
                self.statusBar().showMessage("Agente en pausa. Puedes usar sus comandos en la pestaña 1.")

        def _dibujar_refuerzo(self):
            mano = sm.ManoFisica(self.mano_params)
            a = self.ax_rec
            a.clear()
            a.set_title("Recompensa media por iteración", loc="left")
            b = self.ax_err
            b.clear()
            err_tabla = []
            for gesto in ia.GESTOS:
                d, m = ia.POSES_BASE[gesto]
                dd, mm = mano.respuesta(np.array(d), m)
                err_tabla.append(sm.error_pose(dd[0], mm[0], gesto))
            x = np.arange(ia.NUM_GESTOS)
            b.bar(x - 0.2, err_tabla, 0.4, color="#B4B2A9", label="tabla fija")
            if self.agente is not None and self.agente.historial:
                hist = self.agente.historial
                a.plot([h["iteracion"] for h in hist], [h["recompensa"] for h in hist], color="#534AB7", lw=1)
                poses = self.agente.poses()
                err_rl = [sm.error_pose(*[v[0] for v in mano.respuesta(np.array(poses[g][0]), poses[g][1])], g)
                          for g in ia.GESTOS]
                b.bar(x + 0.2, err_rl, 0.4, color="#1D9E75", label="aprendido por refuerzo")
                for i, g in enumerate(ia.GESTOS):
                    d, m = poses[g]
                    for j, val in enumerate(list(d) + [m]):
                        self.tabla.setItem(i, j, QtWidgets.QTableWidgetItem(f"{val:+.2f}" if j == 5 else f"{val:.2f}"))
                a.set_xlabel("iteración")
            else:
                a.text(0.5, 0.5, "Inicia el agente", ha="center", transform=a.transAxes, color="#5F5E5A")
            b.set_xticks(x, ["Rep", "Abi", "Puñ", "Pin", "Señ", "Flx", "Ext"])
            b.set_title("Error de la pose por gesto (menos es mejor)", loc="left")
            b.legend(fontsize=8)
            self._dibujar_refuerzo_mano()

        def _dibujar_refuerzo_mano(self):
            gesto = ia.GESTOS[self.cmb_ver.currentIndex()]
            od, om = sm.POSES_OBJETIVO[gesto]
            self.mano_obj.actualizar(od, om)
            poses = self.agente.poses() if self.agente is not None and self.agente.historial else ia.POSES_BASE
            d, m = poses[gesto]
            dd, mm = sm.ManoFisica(self.mano_params).respuesta(np.array(d), m)
            self.mano_ref.actualizar(dd[0], mm[0])
            self.lienzo_ref.draw_idle()

        def _guardar_politica(self):
            if self.agente is None or not self.agente.historial:
                QtWidgets.QMessageBox.information(self, "Gemelo digital", "El agente aún no ha aprendido.")
                return
            ruta, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Guardar política", os.path.join(SALIDA, "politica_servos.h5"), "Política Keras (*.h5)")
            if ruta:
                self.agente.guardar(ruta if ruta.endswith(".h5") else ruta + ".h5")

        def _cargar_politica(self):
            ruta, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Cargar política", SALIDA, "Política Keras (*.h5)")
            if not ruta:
                return
            if self.agente is None:
                self.agente = sm.AgenteServos(sm.ManoFisica(self.mano_params), semilla=0)
            try:
                self.agente.cargar(ruta)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "No se pudo cargar", str(e))
                return
            self.agente.historial.append({"iteracion": 0, "recompensa": float("nan"), "error_medio": 0, "sigma": 0})
            self.poses_rl = self.agente.poses()
            self._dibujar_refuerzo()

        def _exportar_poses(self):
            if self.agente is None or not self.agente.historial:
                QtWidgets.QMessageBox.information(self, "Gemelo digital", "El agente aún no ha aprendido.")
                return
            carpeta = QtWidgets.QFileDialog.getExistingDirectory(self, "Carpeta del sketch", CARPETA)
            if carpeta:
                ia.exportar_poses_c(self.agente.poses(), os.path.join(carpeta, "poses_aprendidas.h"))

        def closeEvent(self, ev):
            for hilo in (self.hilo_auto, self.hilo_ref):
                if hilo and hilo.isRunning():
                    hilo.detener = True
                    hilo.wait(60000)
            ev.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = Ventana()
    if args.capturas:
        return capturas(app, ventana, args.capturas)
    ventana.show()
    return app.exec()


def capturas(app, v, carpeta):
    """Corre una demo corta y guarda imágenes de cada pestaña (para el informe)."""
    from PySide6 import QtCore
    os.makedirs(carpeta, exist_ok=True)
    v.resize(1500, 900)
    v.show()
    app.processEvents()

    def esperar(hilo):
        while hilo.isRunning():
            app.processEvents()
            time.sleep(0.05)
        app.processEvents()

    v.tabs.setCurrentIndex(1)
    v.sp_max.setValue(int(os.environ.get("GEMELO_ITER", "40")))
    v._alternar_autonomo()
    esperar(v.hilo_auto)
    app.processEvents()
    v.lienzo_auto.draw()
    v.tabs.widget(1).grab().save(os.path.join(carpeta, "gemelo_autonomo.png"))

    v.tabs.setCurrentIndex(2)
    v.sp_ref.setValue(600)
    v._alternar_refuerzo()
    esperar(v.hilo_ref)
    v.cmb_ver.setCurrentIndex(ia.GESTOS.index("extension"))
    v._dibujar_refuerzo()
    v.lienzo_ref.draw()
    v.tabs.widget(2).grab().save(os.path.join(carpeta, "gemelo_refuerzo.png"))

    v.tabs.setCurrentIndex(0)
    v.cmb_poses.setCurrentIndex(1)
    v.chk_auto.setChecked(False)
    v._fijar_gesto("pinza")
    for _ in range(25):
        v._tic()
        app.processEvents()
    v.lienzo_mano.draw()
    v.lienzo_emg.draw()
    v.tabs.widget(0).grab().save(os.path.join(carpeta, "gemelo_vivo.png"))
    v.grab().save(os.path.join(carpeta, "gemelo_ventana.png"))
    print("Capturas guardadas en", carpeta)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Gemelo digital de la prótesis mioeléctrica")
    ap.add_argument("--h5", help="red neuronal .h5 para empezar (opcional)")
    ap.add_argument("--sin-ventana", action="store_true", help="aprender sin abrir la app")
    ap.add_argument("--iteraciones", type=int, default=60, help="máximo de iteraciones de la red")
    ap.add_argument("--iteraciones-refuerzo", type=int, default=800)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--capturas", help="carpeta donde guardar imágenes de la app")
    args = ap.parse_args()
    if args.sin_ventana:
        modo_sin_ventana(args)
    else:
        sys.exit(lanzar_app(args))


if __name__ == "__main__":
    main()
