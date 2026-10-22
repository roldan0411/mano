"""
============================================================
 PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 Gemelo digital — Motor de simulación y aprendizaje
------------------------------------------------------------
 Roldán Rodríguez — Ing. CCIA
============================================================

Este módulo NO tiene ventana: es el "motor" del gemelo. Lo usa
gemelo_digital.py (la app) y también se puede usar solo.

Partes:
  1. Antebrazo virtual: 10 músculos del antebrazo generan EMG
     según el gesto; dos pares de electrodos lo captan.
  2. Circuito virtual: la misma cadena del circuito real
     (AD620 ×106, pasa-banda 16–482 Hz ×10, acople, 1.65 V)
     y el ADC de 12 bits del ESP32 a 1 kHz.
  3. Mano física virtual: servos, tendones con holgura y el
     acople de la muñeca (al doblarla, los dedos se mueven).
  4. Aprendizaje autónomo de la red: genera sus propios datos,
     se evalúa, practica más los gestos que confunde y sube la
     dificultad sola (currículo).
  5. Aprendizaje por refuerzo: un agente aprende qué comando dar
     a cada servo para que la mano logre cada pose.
"""

import os
import threading
from dataclasses import dataclass, replace

import numpy as np
from scipy import signal

import ia_mano as ia

FS_INTERNA = 4000             # el antebrazo y el circuito se simulan a 4 kHz
DECIMACION = FS_INTERNA // ia.FS
A_MVC = 0.4e-3                # amplitud RMS (V) de un músculo superficial al 100 %

# ------------------------------------------------------------------
# 1. Antebrazo virtual
# ------------------------------------------------------------------
#            nombre: (descripción, banda Hz, peso canal 1, peso canal 2)
# Los músculos de la muñeca tienen fibras más grandes y conducción más rápida,
# por eso su EMG tiene más contenido de alta frecuencia que el de los dedos.
# Los músculos profundos llegan más atenuados y con menos alta frecuencia.
MUSCULOS = {
    "FDS": ("Flexor superficial de los dedos", (25, 190), 1.00, 0.10),
    "FDP": ("Flexor profundo de los dedos",    (20, 130), 0.50, 0.06),
    "FPL": ("Flexor largo del pulgar",         (20, 150), 0.35, 0.05),
    "FCR": ("Flexor radial del carpo",         (45, 300), 0.85, 0.08),
    "FCU": ("Flexor cubital del carpo",        (45, 300), 0.60, 0.10),
    "EDC": ("Extensor común de los dedos",     (25, 190), 0.10, 1.00),
    "EIP": ("Extensor propio del índice",      (20, 150), 0.05, 0.35),
    "EPL": ("Extensor largo del pulgar",       (20, 150), 0.05, 0.30),
    "ECR": ("Extensores radiales del carpo",   (45, 300), 0.08, 0.80),
    "ECU": ("Extensor cubital del carpo",      (45, 300), 0.10, 0.60),
}
NOMBRES_MUSCULOS = list(MUSCULOS)
ACOPLE_BASE = np.array([[MUSCULOS[m][2] for m in MUSCULOS],
                        [MUSCULOS[m][3] for m in MUSCULOS]])

# Activación de cada músculo (0 a 1) en cada gesto.
#            FDS   FDP   FPL   FCR   FCU   EDC   EIP   EPL   ECR   ECU
ACTIVACION = {
    "reposo":    [0.03, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02],
    "abierta":   [0.05, 0.04, 0.03, 0.05, 0.05, 0.70, 0.50, 0.50, 0.25, 0.20],
    "puno":      [0.70, 0.80, 0.50, 0.20, 0.25, 0.10, 0.05, 0.05, 0.35, 0.20],
    "pinza":     [0.45, 0.35, 0.60, 0.10, 0.08, 0.15, 0.05, 0.10, 0.20, 0.10],
    "senalar":   [0.55, 0.60, 0.40, 0.10, 0.10, 0.20, 0.60, 0.05, 0.25, 0.10],
    "flexion":   [0.25, 0.15, 0.08, 0.75, 0.70, 0.05, 0.03, 0.03, 0.05, 0.05],
    "extension": [0.05, 0.05, 0.03, 0.05, 0.05, 0.30, 0.10, 0.10, 0.75, 0.65],
}
ACTIVACION = {g: np.array(v) for g, v in ACTIVACION.items()}


@dataclass
class Condiciones:
    ruido: float = 1.0           # multiplica el ruido de electrodos (base 3 µV RMS)
    interferencia_60hz: float = 3.0   # µV de 60 Hz que se cuelan
    desplazamiento: float = 0.1  # 0 a 1: qué tanto se movieron los electrodos
    fatiga: float = 0.0          # 0 a 1: baja la frecuencia y sube la amplitud
    variabilidad: float = 0.2    # 0 a 1: diferencias entre personas


NIVELES = [  # dificultad del currículo del aprendizaje autónomo
    Condiciones(1.0, 2.0, 0.05, 0.00, 0.10),
    Condiciones(1.5, 5.0, 0.15, 0.10, 0.25),
    Condiciones(2.0, 10.0, 0.30, 0.25, 0.40),
    Condiciones(3.0, 15.0, 0.45, 0.40, 0.60),
    Condiciones(4.0, 20.0, 0.60, 0.60, 0.80),
]


def _cadena_analogica_sos():
    """Función de transferencia del circuito real, discretizada a 4 kHz."""
    G1 = 1 + 49.4e3 / 470
    R1, C1, Rf, Cf = 10e3, 1e-6, 100e3, 3.3e-9
    Rth, C2 = 51e3, 1e-6
    num = G1 * np.polymul([-Rf * C1, 0], [Rth * C2, 0])
    den = np.polymul(np.polymul([Rf * Cf, 1], [R1 * C1, 1]), [Rth * C2, 1])
    b, a = signal.bilinear(num, den, fs=FS_INTERNA)
    return signal.tf2sos(b, a)


SOS_CADENA = _cadena_analogica_sos()


def ganancia_cadena(f):
    w, h = signal.sosfreqz(SOS_CADENA, worN=[f], fs=FS_INTERNA)
    return abs(h[0])


class Antebrazo:
    """Antebrazo + electrodos + circuito + ADC. Entrega cuentas del ADC (0–4095)
    igual que el ESP32, una muestra por milisegundo."""

    def __init__(self, condiciones=None, semilla=None):
        self.rng = np.random.default_rng(semilla)
        self.cond = condiciones or Condiciones()
        self._fatiga_filtros = None
        self.nuevo_usuario()

    # --- personas y electrodos ---
    def nuevo_usuario(self):
        v = self.cond.variabilidad
        M = len(MUSCULOS)
        self.ganancia_musculo = np.exp(self.rng.normal(0, 0.35 * v, M))
        self.patron = {g: np.clip(a * np.exp(self.rng.normal(0, 0.25 * v, M))
                                  + self.rng.normal(0, 0.03 * v, M), 0.0, 1.0)
                       for g, a in ACTIVACION.items()}
        self.piel = np.exp(self.rng.normal(0, 0.2 * v, 2))
        self.recolocar_electrodos()
        self._reiniciar()

    def recolocar_electrodos(self):
        d = self.cond.desplazamiento
        self.W = ACOPLE_BASE * np.exp(self.rng.normal(0, 0.6 * d, ACOPLE_BASE.shape))
        self.W += d * 0.15 * self.rng.random(ACOPLE_BASE.shape)  # más cruce entre canales

    def cambiar_condiciones(self, cond):
        cambio_persona = cond.variabilidad != self.cond.variabilidad
        cambio_elec = cond.desplazamiento != self.cond.desplazamiento
        self.cond = cond
        if cambio_persona:
            self.nuevo_usuario()
        elif cambio_elec:
            self.recolocar_electrodos()

    # --- estado interno ---
    def _filtros_musculo(self):
        f = self.cond.fatiga
        if self._fatiga_filtros == f:
            return
        self._fatiga_filtros = f
        self.sos_m, self.norma_m = [], []
        for nombre, (_, (lo, hi), _, _) in MUSCULOS.items():
            k = 1 - 0.3 * f  # la fatiga baja la frecuencia
            sos = signal.butter(2, [lo * k, hi * k], btype="band", fs=FS_INTERNA, output="sos")
            imp = signal.sosfilt(sos, np.r_[1.0, np.zeros(4000)])
            self.sos_m.append(sos)
            self.norma_m.append(np.sqrt(np.sum(imp ** 2)))
        if not hasattr(self, "zi_m") or len(self.zi_m) != len(self.sos_m):
            self.zi_m = [np.zeros((s.shape[0], 2)) for s in self.sos_m]

    def _reiniciar(self):
        self._fatiga_filtros = None
        if hasattr(self, "zi_m"):
            del self.zi_m
        self._filtros_musculo()
        self.zi_cadena = [np.zeros((SOS_CADENA.shape[0], 2)) for _ in range(2)]
        self.t = 0.0
        self.fase60 = self.rng.uniform(0, 2 * np.pi)
        self.fases_mod = self.rng.uniform(0, 2 * np.pi, (len(MUSCULOS), 2))
        self.gesto = "reposo"
        self.activ = self.patron["reposo"].copy()
        self.objetivo = self.activ.copy()
        self.artefacto = 0.0
        self.t_artefacto = 0.0
        self.muestras(300)  # deja asentar los filtros

    def fijar_gesto(self, gesto):
        if gesto == self.gesto:
            return
        self.gesto = gesto
        fuerza = self.rng.uniform(0.75, 1.25) if gesto != "reposo" else 1.0
        self.objetivo = np.clip(self.patron[gesto] * fuerza, 0, 1)
        self.artefacto = self.rng.uniform(0.05, 0.3) * 1e-3  # movimiento de cables y piel
        self.t_artefacto = 0.0

    def muestras(self, n):
        """Genera n muestras a 1 kHz -> arreglo (n, 2) de cuentas del ADC."""
        self._filtros_musculo()
        N = n * DECIMACION
        dt = 1.0 / FS_INTERNA
        t = self.t + dt * np.arange(1, N + 1)
        tau = 0.06
        tt = (t - self.t)[:, None]
        activ = self.objetivo + (self.activ - self.objetivo) * np.exp(-tt / tau)
        self.activ = activ[-1].copy()
        mod = (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t[:, None] + self.fases_mod[:, 0])
                 + 0.03 * np.sin(2 * np.pi * 10 * t[:, None] + self.fases_mod[:, 1]))
        amp = 1 + 0.3 * self.cond.fatiga
        fuentes = np.empty((N, len(MUSCULOS)))
        ruido_blanco = self.rng.normal(0, 1, (N, len(MUSCULOS)))
        for m in range(len(MUSCULOS)):
            y, self.zi_m[m] = signal.sosfilt(self.sos_m[m], ruido_blanco[:, m], zi=self.zi_m[m])
            fuentes[:, m] = y / self.norma_m[m]
        s = A_MVC * amp * self.ganancia_musculo * activ * mod * fuentes   # (N, M) voltios
        v = (s @ self.W.T) * self.piel                                    # (N, 2)
        v += self.rng.normal(0, 3e-6 * self.cond.ruido, (N, 2))
        v += self.cond.interferencia_60hz * 1e-6 * np.sin(2 * np.pi * 60 * t + self.fase60)[:, None] \
            * np.array([1.0, 0.8])
        if self.artefacto > 0:
            ta = self.t_artefacto + dt * np.arange(1, N + 1)
            art = self.artefacto * np.exp(-ta / 0.15) * np.sin(2 * np.pi * 4 * ta)
            v += art[:, None] * np.array([1.0, 0.7])
            self.t_artefacto = ta[-1]
            if ta[-1] > 1.0:
                self.artefacto = 0.0
        salida = np.empty((n, 2))
        for c in range(2):
            y, self.zi_cadena[c] = signal.sosfilt(SOS_CADENA, v[:, c], zi=self.zi_cadena[c])
            y = np.clip(y, -7.5, 7.5)                     # rieles del TL072
            vadc = np.clip(1.65 + y[DECIMACION - 1::DECIMACION], 0.0, 3.3)  # diodos + ADC
            salida[:, c] = vadc
        self.t = t[-1]
        cuentas = np.round(salida / 3.3 * 4095 + self.rng.normal(0, 1.5, salida.shape))
        return np.clip(cuentas, 0, 4095).astype(np.int16)


def generar_dataset(cond, n_usuarios=2, rondas=3, pesos=None, extra_por_ronda=3, semilla=None):
    """Simula sesiones de captura iguales a capturar_datos.py. Devuelve X (n, 8), y (n,)."""
    rng = np.random.default_rng(semilla)
    X, y = [], []
    for _ in range(n_usuarios):
        ab = Antebrazo(cond, semilla=int(rng.integers(1 << 31)))
        for _ in range(rondas):
            orden = list(ia.GESTOS)
            if pesos is not None and extra_por_ronda > 0:
                p = np.asarray(pesos, float)
                orden += list(rng.choice(ia.GESTOS, size=extra_por_ronda, p=p / p.sum()))
            rng.shuffle(orden)
            for g in orden:
                ab.fijar_gesto("reposo")
                ab.muestras(400)
                ab.fijar_gesto(g)
                s = ab.muestras(2500)
                v = ia.ventanear(s, descarte=300)
                X.append(ia.caracteristicas_lote(v))
                y += [ia.GESTOS.index(g)] * len(v)
    return np.vstack(X), np.array(y)


# ------------------------------------------------------------------
# 3. Mano física virtual (servos + tendones + muñeca)
# ------------------------------------------------------------------
# Pose anatómica que se quiere lograr en cada gesto (lo que se VE en la mano).
POSES_OBJETIVO = {
    "reposo":    ([0.25, 0.25, 0.25, 0.25, 0.25], 0.0),
    "abierta":   ([0.00, 0.00, 0.00, 0.00, 0.00], 0.0),
    "puno":      ([0.80, 0.90, 0.90, 0.90, 0.90], 0.0),
    "pinza":     ([0.65, 0.65, 0.05, 0.05, 0.05], 0.0),
    "senalar":   ([0.75, 0.00, 0.90, 0.90, 0.90], 0.0),
    "flexion":   ([0.25, 0.25, 0.25, 0.25, 0.25], -0.9),
    "extension": ([0.25, 0.25, 0.25, 0.25, 0.25], 0.9),
}


@dataclass
class ParametrosMano:
    holgura: tuple = (0.10, 0.08, 0.08, 0.09, 0.12)   # tendón flojo antes de mover el dedo
    curvatura: tuple = (1.15, 0.95, 1.0, 1.05, 1.2)   # no linealidad del tendón
    alcance: tuple = (0.90, 0.97, 0.97, 0.95, 0.92)   # flexión máxima que se logra
    acople_muneca: tuple = (0.15, 0.25, 0.25, 0.25, 0.20)  # extensión de muñeca cierra los dedos
    desfase_muneca: float = 0.08                      # el disco del servo quedó girado
    alcance_muneca: float = 0.92

    @staticmethod
    def aleatoria(rng):
        return ParametrosMano(
            holgura=tuple(rng.uniform(0.03, 0.18, 5)),
            curvatura=tuple(rng.uniform(0.8, 1.3, 5)),
            alcance=tuple(rng.uniform(0.85, 1.0, 5)),
            acople_muneca=tuple(rng.uniform(0.05, 0.40, 5)),
            desfase_muneca=float(rng.uniform(-0.15, 0.15)),
            alcance_muneca=float(rng.uniform(0.85, 1.0)))


class ManoFisica:
    """Modelo de la mano impresa: de comandos de servo a posición real de dedos y muñeca."""
    VEL_DEDO = 1 / 0.3            # SG90: recorrido completo en ~0.3 s
    VEL_MUNECA = 0.05 / 0.02      # firmware: 2° cada 20 ms con rango de 40° -> 2.5 /s

    def __init__(self, params=None):
        self.p = params or ParametrosMano()
        self.servo_dedos = np.zeros(5)
        self.servo_muneca = 0.0

    def respuesta(self, cmd_dedos, cmd_muneca):
        """Posición en reposo (estado estable) para unos comandos. Vectorizado."""
        p = self.p
        cd = np.atleast_2d(np.clip(cmd_dedos, 0, 1))
        cm = np.clip(np.atleast_1d(cmd_muneca), -1, 1)
        h, g, a, k = (np.array(p.holgura), np.array(p.curvatura), np.array(p.alcance),
                      np.array(p.acople_muneca))
        muneca = np.clip(cm * p.alcance_muneca + p.desfase_muneca, -1, 1)
        base = np.clip((cd - h) / (1 - h), 0, 1) ** g * a
        dedos = np.clip(base + k * muneca[:, None], 0, 1)
        return dedos, muneca

    def paso(self, dt, cmd_dedos, cmd_muneca):
        d = np.clip(cmd_dedos, 0, 1) - self.servo_dedos
        self.servo_dedos += np.clip(d, -self.VEL_DEDO * dt, self.VEL_DEDO * dt)
        dm = np.clip(cmd_muneca, -1, 1) - self.servo_muneca
        self.servo_muneca += float(np.clip(dm, -self.VEL_MUNECA * dt, self.VEL_MUNECA * dt))
        dedos, muneca = self.respuesta(self.servo_dedos, self.servo_muneca)
        return dedos[0], float(muneca[0])


def error_pose(dedos, muneca, gesto):
    od, om = POSES_OBJETIVO[gesto]
    return float(np.mean(np.abs(np.asarray(dedos) - od)) * 0.8 + abs(muneca - om) * 0.2)


# ------------------------------------------------------------------
# 4. Aprendizaje autónomo de la red neuronal
# ------------------------------------------------------------------
class AprendizajeAutonomo:
    """Ciclo: practicar -> entrenar -> evaluar -> decidir, sin intervención.

    - Practicar: simula personas nuevas haciendo los 7 gestos, con más
      repeticiones de los gestos que la red está confundiendo.
    - Entrenar: unas épocas sobre una memoria de repetición (lo nuevo +
      lo anterior), para no olvidar lo aprendido.
    - Evaluar: con personas que la red nunca vio, en el nivel actual y
      en el nivel más difícil (prueba de robustez).
    - Decidir: si domina el nivel (meta) o ya no mejora en él (se
      estancó), sube la dificultad; si mejora la robustez, guarda el
      mejor .h5 y su modelo_red.h. En el último nivel, si se estanca,
      da el aprendizaje por terminado.
    """

    def __init__(self, modelo=None, semilla=0, memoria_max=40000, meta=0.95, paciencia=6,
                 ruta_mejor=None, ruta_c=None):
        self.modelo = modelo
        self.rng = np.random.default_rng(semilla)
        self.memoria_max = memoria_max
        self.meta = meta
        self.paciencia = paciencia
        self.mejor_nivel = -1.0
        self.sin_mejora = 0
        self.terminado = False
        self.ruta_mejor = ruta_mejor
        self.ruta_c = ruta_c
        self.nivel = 0
        self.racha = 0
        self.iteracion = 0
        self.X = np.empty((0, ia.NUM_CARACTERISTICAS))
        self.y = np.empty((0,), int)
        self.recall = np.full(ia.NUM_GESTOS, 0.5)
        self.historial = []
        self.mejor_robustez = -1.0
        self.confusion = np.zeros((ia.NUM_GESTOS, ia.NUM_GESTOS), int)
        self._bancos = {}

    def _banco(self, nivel):
        if nivel not in self._bancos:  # personas fijas de evaluación por nivel
            self._bancos[nivel] = generar_dataset(NIVELES[nivel], n_usuarios=3, rondas=2,
                                                  semilla=10_000 + nivel)
        return self._bancos[nivel]

    def _evaluar(self, X, y):
        p = self.modelo(X.astype(np.float32), training=False).numpy().argmax(1)
        cm = np.zeros((ia.NUM_GESTOS, ia.NUM_GESTOS), int)
        np.add.at(cm, (y, p), 1)
        recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
        return float(np.mean(p == y)), recall, cm

    def _linea_base(self):
        """Antes de la primera iteración: cuál es el mejor modelo que ya existe.
        Así una corrida nueva nunca reemplaza un .h5 guardado por uno peor."""
        base, avisos = -1.0, []
        if self.modelo is not None:
            base, _, _ = self._evaluar(*self._banco(len(NIVELES) - 1))
            avisos.append(f"Red de partida: robustez {base * 100:.1f} %")
        if self.ruta_mejor and os.path.exists(self.ruta_mejor):
            guardada = ia.leer_metadatos(self.ruta_mejor).get("precision_robustez")
            if guardada is not None and guardada > base:
                base = float(guardada)
                avisos.append(f"Ya hay un .h5 guardado con robustez {base * 100:.1f} %: "
                              "solo se reemplaza si esta corrida lo supera")
        self.mejor_robustez = base
        return avisos

    def iteracion_completa(self, rondas=3, epocas=3):
        avisos = self._linea_base() if self.iteracion == 0 else []
        self.iteracion += 1
        # 1) practicar: más datos de los gestos débiles + repaso de niveles anteriores
        pesos = (1 - self.recall) + 0.15
        Xn, yn = generar_dataset(NIVELES[self.nivel], n_usuarios=2, rondas=rondas,
                                 pesos=pesos, semilla=int(self.rng.integers(1 << 31)))
        if self.nivel > 0:
            nivel_repaso = int(self.rng.integers(0, self.nivel))
            Xr, yr = generar_dataset(NIVELES[nivel_repaso], n_usuarios=1, rondas=1,
                                     semilla=int(self.rng.integers(1 << 31)))
            Xn, yn = np.vstack([Xn, Xr]), np.r_[yn, yr]
        self.X = np.vstack([self.X, Xn])[-self.memoria_max:]
        self.y = np.r_[self.y, yn][-self.memoria_max:]
        nuevo = self.modelo is None
        if nuevo:
            self.modelo = ia.crear_red(X_adaptar=self.X)
        # 2) entrenar
        h = self.modelo.fit(self.X.astype(np.float32), self.y, epochs=epocas, batch_size=64,
                            shuffle=True, verbose=0)
        perdida = float(h.history["loss"][-1])
        # 3) evaluar
        acc, self.recall, self.confusion = self._evaluar(*self._banco(self.nivel))
        robustez, _, _ = self._evaluar(*self._banco(len(NIVELES) - 1))
        # 4) decidir
        eventos = list(avisos)
        self.racha = self.racha + 1 if acc >= self.meta else 0
        if acc > self.mejor_nivel + 0.005:
            self.mejor_nivel, self.sin_mejora = acc, 0
        else:
            self.sin_mejora += 1
        motivo = None
        if self.racha >= 2:
            motivo = "domina el nivel"
        elif self.sin_mejora >= self.paciencia:
            motivo = "se estancó"
        if motivo and self.nivel < len(NIVELES) - 1:
            self.nivel += 1
            self.racha, self.sin_mejora, self.mejor_nivel = 0, 0, -1.0
            eventos.append(f"La red {motivo}; sube a nivel {self.nivel + 1}")
            _, self.recall, self.confusion = self._evaluar(*self._banco(self.nivel))
        elif motivo and not self.terminado:
            self.terminado = True
            eventos.append(f"La red {motivo} en el nivel máximo: aprendizaje terminado")
        if robustez > self.mejor_robustez:
            self.mejor_robustez = robustez
            eventos.append(f"Nuevo mejor modelo (robustez {robustez * 100:.1f} %)")
            if self.ruta_mejor:
                ia.guardar_red(self.modelo, self.ruta_mejor,
                               {"origen": "gemelo digital — aprendizaje autónomo",
                                "iteracion": self.iteracion, "nivel": self.nivel + 1,
                                "precision_robustez": round(robustez, 4)})
            if self.ruta_c:
                ia.exportar_red_c(self.modelo, self.ruta_c)
        peor = int(np.argmin(self.recall))
        registro = {"iteracion": self.iteracion, "nivel": self.nivel + 1, "precision": acc,
                    "robustez": robustez, "perdida": perdida, "memoria": len(self.X),
                    "recall": self.recall.copy(), "peor_gesto": ia.GESTOS[peor], "eventos": eventos,
                    "terminado": self.terminado}
        self.historial.append(registro)
        return registro


# ------------------------------------------------------------------
# 5. Aprendizaje por refuerzo para los servos
# ------------------------------------------------------------------
class AgenteServos:
    """Política gaussiana entrenada con REINFORCE (gradiente de política).

    Estado:   el gesto que hay que hacer (vector one-hot de 7).
    Acción:   6 comandos (5 dedos de 0 a 1 y la muñeca de -1 a 1).
    Recompensa: menos el error entre la pose lograda por la mano física
                virtual y la pose objetivo, menos un pequeño costo de
                esfuerzo (no tensar de más los servos).
    La red de la política se guarda en .h5 y sus comandos medios se
    exportan a poses_aprendidas.h para el ESP32.
    """

    def __init__(self, mano, semilla=0, tasa=3e-3, sigma_inicial=0.25):
        import keras
        import tensorflow as tf
        self.tf = tf
        self.keras = keras
        self.mano = mano
        self.rng = np.random.default_rng(semilla)
        keras.utils.set_random_seed(semilla)
        ent = keras.Input(shape=(ia.NUM_GESTOS,), name="gesto")
        x = keras.layers.Dense(32, activation="tanh", name="oculta")(ent)
        x = keras.layers.Dense(6, name="comandos_crudos")(x)
        self.politica = keras.Model(ent, x, name="politica_servos")
        self.log_sigma = tf.Variable(np.full(6, np.log(sigma_inicial), np.float32))
        self.opt = keras.optimizers.Adam(tasa)
        self.base = np.zeros(ia.NUM_GESTOS)
        self.historial = []
        self.iteracion = 0
        self.objetivos = np.array([POSES_OBJETIVO[g][0] + [POSES_OBJETIVO[g][1]] for g in ia.GESTOS])

    def _media(self, onehot):
        tf = self.tf
        z = self.politica(onehot)
        return tf.concat([tf.sigmoid(z[:, :5]), tf.tanh(z[:, 5:])], axis=1)

    def recompensa(self, acciones, idx):
        dedos, muneca = self.mano.respuesta(acciones[:, :5], acciones[:, 5])
        obj = self.objetivos[idx]
        err = np.mean((dedos - obj[:, :5]) ** 2, axis=1) + 2 * (muneca - obj[:, 5]) ** 2
        esfuerzo = 0.01 * np.mean(np.clip(acciones[:, :5], 0, 1) ** 2, axis=1)
        return -(err + esfuerzo)

    def iteracion_completa(self, lote=128):
        tf = self.tf
        self.iteracion += 1
        idx = self.rng.integers(0, ia.NUM_GESTOS, lote)
        onehot = np.eye(ia.NUM_GESTOS, dtype=np.float32)[idx]
        with tf.GradientTape() as cinta:
            mu = self._media(onehot)
            sigma = tf.exp(self.log_sigma)
            eps = tf.constant(self.rng.normal(0, 1, (lote, 6)).astype(np.float32))
            accion = tf.stop_gradient(mu + sigma * eps)
            a = accion.numpy()
            a_aplicada = np.c_[np.clip(a[:, :5], 0, 1), np.clip(a[:, 5], -1, 1)]
            r = self.recompensa(a_aplicada, idx)
            ventaja = r - self.base[idx]
            ventaja = (ventaja - ventaja.mean()) / (ventaja.std() + 1e-8)
            logp = tf.reduce_sum(-0.5 * ((accion - mu) / sigma) ** 2 - self.log_sigma, axis=1)
            perdida = -tf.reduce_mean(tf.constant(ventaja, tf.float32) * logp)
        variables = self.politica.trainable_variables + [self.log_sigma]
        grads = cinta.gradient(perdida, variables)
        self.opt.apply_gradients(zip(grads, variables))
        self.log_sigma.assign(tf.maximum(self.log_sigma, np.log(0.02)))
        for g in range(ia.NUM_GESTOS):  # línea base por gesto (promedio móvil)
            m = idx == g
            if m.any():
                self.base[g] = 0.9 * self.base[g] + 0.1 * r[m].mean()
        poses = self.poses()
        err = np.mean([self.error_gesto(g, poses) for g in ia.GESTOS])
        reg = {"iteracion": self.iteracion, "recompensa": float(r.mean()),
               "error_medio": err, "sigma": float(np.exp(self.log_sigma.numpy()).mean())}
        self.historial.append(reg)
        return reg

    def poses(self):
        """Comandos medios (sin exploración) que la política da a cada gesto."""
        mu = self._media(np.eye(ia.NUM_GESTOS, dtype=np.float32)).numpy()
        return {g: (list(mu[i, :5]), float(mu[i, 5])) for i, g in enumerate(ia.GESTOS)}

    def error_gesto(self, gesto, poses):
        d, m = poses[gesto]
        dedos, muneca = self.mano.respuesta(np.array(d), m)
        return error_pose(dedos[0], muneca[0], gesto)

    def guardar(self, ruta):
        self.politica.save(ruta)
        import h5py, json
        with h5py.File(ruta, "a") as f:
            f.attrs["metadatos_mano"] = json.dumps({
                "tipo": "politica de servos (REINFORCE)", "gestos": ia.GESTOS,
                "salidas": ["pulgar", "indice", "medio", "anular", "menique", "muneca"],
                "activacion_salida": "sigmoid (dedos), tanh (muñeca)",
                "sigma": np.exp(self.log_sigma.numpy()).tolist(),
                "autor": "Roldán Rodríguez — Ing. CCIA"}, ensure_ascii=False)

    def cargar(self, ruta):
        modelo = self.keras.models.load_model(ruta, compile=False)
        if modelo.input_shape[-1] != ia.NUM_GESTOS or modelo.output_shape[-1] != 6:
            raise ia.ErrorModelo("Esa política no es de 7 gestos y 6 servos.")
        self.politica.set_weights(modelo.get_weights())


# ------------------------------------------------------------------
# Gemelo en vivo: antebrazo -> circuito -> red -> firmware -> mano
# ------------------------------------------------------------------
class GemeloEnVivo:
    """Repite en el PC lo que hace protesis_tiempo_real.ino, paso a paso."""

    def __init__(self, condiciones=None, semilla=None):
        self.antebrazo = Antebrazo(condiciones, semilla)
        self.mano = ManoFisica()
        self.red = None
        self.poses = dict(ia.POSES_BASE)
        self.senal = np.full((2000, 2), 2048, np.int16)
        self.historial = [-1, -1, -1]
        self.ultima_clase = 0
        self.probs = np.full(ia.NUM_GESTOS, 1 / ia.NUM_GESTOS)
        self.confianza = 0.0
        self.clase = 0
        self.aciertos = []
        self.lock = threading.Lock()

    def fijar_red(self, red_numpy):
        with self.lock:
            self.red = red_numpy

    def _mayoria(self, nueva):
        h = self.historial
        h[0], h[1], h[2] = h[1], h[2], nueva
        if h[0] >= 0 and (h[0] == h[1] or h[0] == h[2]):
            return h[0]
        if h[1] >= 0 and h[1] == h[2]:
            return h[1]
        return nueva

    def paso(self, gesto_usuario, dt=0.1):
        n = int(round(dt * ia.FS))
        self.antebrazo.fijar_gesto(gesto_usuario)
        nuevas = self.antebrazo.muestras(n)
        self.senal = np.vstack([self.senal, nuevas])[-2000:]
        with self.lock:
            red = self.red
        if red is not None:
            x = ia.caracteristicas_lote(self.senal[None, -ia.VENTANA:])
            self.probs = red.probabilidades(x)[0]
            pred = int(self.probs.argmax())
            self.confianza = float(self.probs[pred])
            if self.confianza >= ia.UMBRAL_CONFIANZA:
                self.ultima_clase = pred
            self.clase = self._mayoria(self.ultima_clase)
        else:
            self.clase, self.confianza = 0, 0.0
        gesto_red = ia.GESTOS[self.clase]
        cmd_d, cmd_m = self.poses[gesto_red]
        dedos, muneca = self.mano.paso(dt, np.array(cmd_d, float), cmd_m)
        if red is not None and self.antebrazo.activ.max() > 0.5 * self.antebrazo.objetivo.max():
            self.aciertos = (self.aciertos + [gesto_red == gesto_usuario])[-100:]
        return {"gesto_red": gesto_red, "dedos": dedos, "muneca": muneca,
                "probs": self.probs, "confianza": self.confianza,
                "precision_vivo": float(np.mean(self.aciertos)) if self.aciertos else None}
