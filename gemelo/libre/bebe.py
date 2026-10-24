"""
La mano bebé: empieza de cero y aprende mirando manos de personas en internet
=============================================================================

Nadie le enseña gestos. Como un bebé:

  1. BALBUCEO   activa sus 10 músculos al azar y mira dónde quedan sus dedos:
                así conoce su propio cuerpo.
  2. IMITA      mira la mano de una persona real de internet (HaGRID v2: 33
                gestos cotidianos como "llamar", "ok", "rock", "me gusta") e
                intenta hacer lo mismo. Elige con curiosidad: practica más los
                gestos en los que está mejorando.
  3. DOMINA     cuando ya imita bien casi todo lo que puede, afina.

Su cerebro son dos redes que aprenden en línea, intento a intento:
  modelo inverso  pose deseada → qué músculos activar
  modelo directo  músculos → qué pose va a salir

Un DESCUBRIMIENTO es la primera vez que imita bien un gesto de internet
(foto: la mano de la persona y la suya). Si tras muchos intentos no lo logra,
lo reconoce como imposible para su anatomía (por ejemplo, "rock" necesita
mover el meñique solo, y sus tendones no lo permiten).

Corre en la CPU (NumPy + scikit-learn): no le quita la GPU a la competencia.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import collections
import pickle
import queue
import threading
import time
import traceback
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from .. import config
from . import cuerpo, internet

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RAIZ = config.RAIZ_DATOS / "motor" / "bebe"
VERSION = 2
FIN_BALBUCEO = 300          # ~3 minutos en tiempo real: luego empieza a mirar internet
MEMORIA = 30_000
APRENDIDO, IMPOSIBLE, MIN_INTENTOS = 0.13, 0.26, 80


class Bebe:
    def __init__(self, red=None, semilla=None):
        self.version = VERSION
        self.semilla = int(time.time()) if semilla is None else semilla
        self.rng = np.random.default_rng(self.semilla)
        self.inverso = MLPRegressor(hidden_layer_sizes=(96, 96), activation="tanh", learning_rate_init=3e-3, random_state=1)
        self.directo = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", learning_rate_init=3e-3, random_state=2)
        self._A = np.zeros((MEMORIA, 10), np.float32)             # búfer circular de intentos
        self._P = np.zeros((MEMORIA, 6), np.float32)
        self.n_mem, self.pos = 0, 0
        self.intentos, self.listo = 0, False
        self.celdas_vistas = set()
        self.hitos = []
        self.curva = []                                            # (intento, error imitando, R² imitación, R² directo, cobertura)
        self.nacimiento = time.time()
        self.n_clases = 0 if red is None else len(red["claves"])
        self.errores = [collections.deque(maxlen=40) for _ in range(self.n_clases)]
        self.intentos_clase = np.zeros(self.n_clases, int)
        self.estado_clase = ["aprendiendo"] * self.n_clases
        if red is not None:                                        # examen fijo: 300 manos de internet
            r = np.random.default_rng(99)
            self.i_prueba = r.choice(len(red["pose"]), min(300, len(red["pose"])), replace=False)

    @property
    def etapa(self):
        if self.intentos < FIN_BALBUCEO or not self.listo:
            return "balbuceo"
        error = self.curva[-1][1] if self.curva else 1.0
        return "domina" if error < 0.12 else "imita"

    # ---------- elegir qué imitar (curiosidad) ----------
    def _elegir_clase(self):
        peso = np.ones(self.n_clases)
        for c in range(self.n_clases):
            e = list(self.errores[c])
            if self.estado_clase[c] == "imposible":
                peso[c] = 0.15                                     # lo sigue intentando de vez en cuando
            elif len(e) >= 10:
                progreso = max(np.mean(e[: len(e) // 2]) - np.mean(e[len(e) // 2:]), 0)   # ¿está mejorando?
                peso[c] = 0.3 + 20 * progreso + (0.3 if self.estado_clase[c] == "aprendiendo" else 0)
        return int(self.rng.choice(self.n_clases, p=peso / peso.sum()))

    # ---------- un intento ----------
    def intento(self, mapa, red):
        etapa = self.etapa
        meta, obs, clase = None, None, None
        if etapa == "balbuceo" or red is None:
            a = cuerpo.balbuceo(self.rng)[0]
        else:
            if self.rng.random() < 0.15:                            # a veces juega sola para conocer mejor su cuerpo
                base = self._P[int(self.rng.integers(self.n_mem))]
                meta = base + self.rng.normal(0, 0.18, 6)
                meta = np.r_[np.clip(meta[:5], 0, 1), np.clip(meta[5], -1, 1)]
            else:
                clase = self._elegir_clase()
                obs = int(self.rng.choice(np.flatnonzero(red["clase"] == clase)))
                meta = red["pose"][obs].astype(float)
            ruido = 0.25 if etapa == "imita" else 0.08
            a = np.clip(self.inverso.predict(meta[None])[0] + self.rng.normal(0, ruido * self.rng.random(), 10), 0, 1)
        p = cuerpo.pose(a, self.rng)[0]
        self.intentos += 1
        self._A[self.pos], self._P[self.pos] = a, p
        self.pos = (self.pos + 1) % MEMORIA
        self.n_mem = min(self.n_mem + 1, MEMORIA)
        z = mapa.proyectar(p)[0]
        self.celdas_vistas.add(int(mapa.celda(z[None])[0]))
        error = float(cuerpo.distancia(p, meta)) if meta is not None else np.nan
        nuevo = None
        if clase is not None:
            self.errores[clase].append(error)
            self.intentos_clase[clase] += 1
            nuevo = self._revisar(clase, p, red, obs)
        self._aprender()
        if self.intentos % 50 == 0 and self.listo:
            self._evaluar(mapa, red)
        return {"n": self.intentos, "etapa": etapa, "meta": None if meta is None else meta.tolist(), "alcanzado": p.tolist(),
                "activacion": a.tolist(), "error": error, "z": z.tolist(),
                "zmeta": None if meta is None else mapa.proyectar(meta)[0].tolist(), "descubrimiento": nuevo,
                "clase": clase, "nombre": None if clase is None else str(red["nombres"][clase]),
                "obs": None if obs is None else red["landmarks_vista"][obs].tolist()}

    def _revisar(self, c, p, red, obs):
        e = list(self.errores[c])
        if self.estado_clase[c] != "aprendido" and len(e) >= 8 and np.median(e[-8:]) < APRENDIDO:
            self.estado_clase[c] = "aprendido"
            h = {"n": len(self.hitos) + 1, "nombre": str(red["nombres"][c]), "clave": str(red["claves"][c]),
                 "intento": self.intentos, "dof": p.tolist(), "obs": red["landmarks_vista"][obs].tolist(),
                 "error": float(np.median(e[-8:])), "foto": None, "t": time.time(), "tipo": "aprendido"}
            self.hitos.append(h)
            return h
        if self.estado_clase[c] == "aprendiendo" and self.intentos_clase[c] >= MIN_INTENTOS and np.median(e) > IMPOSIBLE:
            self.estado_clase[c] = "imposible"
            return {"n": None, "nombre": str(red["nombres"][c]), "tipo": "imposible", "intento": self.intentos,
                    "error": float(np.median(e))}
        return None

    def _aprender(self, lote=64):
        if self.n_mem < lote:
            return
        i = self.rng.integers(0, self.n_mem, lote)
        self.inverso.partial_fit(self._P[i], self._A[i])
        self.directo.partial_fit(self._A[i], self._P[i])
        self.listo = True

    def _evaluar(self, mapa, red):
        cob = len(self.celdas_vistas & set(mapa.alcanzables.tolist())) / len(mapa.alcanzables)
        A_al = cuerpo.balbuceo(np.random.default_rng(5), 300)
        r2_dir = float(r2_score(cuerpo.pose(A_al), self.directo.predict(A_al)))
        if red is None:
            self.curva.append((self.intentos, np.nan, np.nan, r2_dir, cob))
            return
        metas = red["pose"][self.i_prueba]
        logrado = cuerpo.pose(np.clip(self.inverso.predict(metas), 0, 1))
        error = float(np.mean(cuerpo.distancia(logrado, metas)))
        r2_imit = float(r2_score(metas[:, :5], logrado[:, :5]))   # la muñeca no se ve en las fotos
        self.curva.append((self.intentos, error, r2_imit, r2_dir, cob))

    def resumen(self, mapa, red):
        u = self.curva[-1] if self.curva else (0, np.nan, np.nan, np.nan, 0.0)
        ult = (self.pos - 1 - np.arange(min(self.n_mem, 4000))) % MEMORIA
        clases = []
        for c in range(self.n_clases):
            e = list(self.errores[c])
            clases.append({"nombre": str(red["nombres"][c]), "estado": self.estado_clase[c], "intentos": int(self.intentos_clase[c]),
                           "error": float(np.median(e[-10:])) if e else np.nan})
        return {"intentos": self.intentos, "etapa": self.etapa, "error_metas": u[1], "r2_imitacion": u[2], "r2_directo": u[3],
                "cobertura": u[4], "curva": np.array(self.curva[-600:], float) if self.curva else np.zeros((0, 5)),
                "puntos": mapa.proyectar(self._P[ult]).astype(np.float32) if self.n_mem else np.zeros((0, 2)),
                "hitos": self.hitos[-60:], "n_hitos": len(self.hitos), "clases": clases,
                "aprendidos": sum(s == "aprendido" for s in self.estado_clase),
                "imposibles": sum(s == "imposible" for s in self.estado_clase), "edad_s": time.time() - self.nacimiento}


# ------------------------------------------------------------------
# Proceso que lanza la ventana "Entrenamiento libre"
# ------------------------------------------------------------------
def _cargar(red):
    ruta = RAIZ / "bebe.pkl"
    if ruta.exists():
        try:
            with open(ruta, "rb") as f:
                b = pickle.load(f)
            if getattr(b, "version", 1) == VERSION and b.n_clases == (0 if red is None else len(red["claves"])):
                return b
        except Exception:
            pass
    return Bebe(red)


def _guardar(b):
    RAIZ.mkdir(parents=True, exist_ok=True)
    tmp = RAIZ / "bebe.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(b, f)
    os.replace(tmp, RAIZ / "bebe.pkl")


def _foto(h, salida):
    """Foto del descubrimiento: la mano de la persona (internet) y la de la mano bebé."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from ..motor import fotos
        carpeta = RAIZ / "descubrimientos"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / f"aprendio_{h['n']:03d}_{h['clave']}.png"
        fotos.foto_gesto(ruta, np.array(h["dof"]), 0, f"Aprendió «{h['nombre']}»",
                         f"intento {h['intento']:,} · error {h['error']:.3f}".replace(",", "."))
        fig = Figure(figsize=(7.6, 4.6), dpi=110)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 2, 1)
        L = np.asarray(h["obs"])
        for a, b in internet.CONEXIONES:
            ax.plot(L[[a, b], 0], L[[a, b], 1], color="#2563c9", lw=3, solid_capstyle="round")
        ax.scatter(L[:, 0], L[:, 1], s=18, color="white", edgecolors="#2563c9", zorder=3)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Lo que vio en internet", fontsize=10, color="#52514e")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(mpimg.imread(ruta))
        ax2.set_axis_off()
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.92, 0.05)
        fig.savefig(ruta)
        h["foto"] = str(ruta)
        salida.put({"tipo": "foto", "n": h["n"], "foto": str(ruta)})
    except Exception as e:
        salida.put({"tipo": "aviso", "texto": f"No se pudo tomar la foto: {e}"})


def proceso(salida, entrada):
    def enviar(m):
        try:
            salida.put_nowait(m)
        except queue.Full:
            pass

    RAIZ.mkdir(parents=True, exist_ok=True)
    candado = open(RAIZ / "en_uso.lock", "w")                  # una sola mano bebé a la vez
    try:
        import msvcrt
        msvcrt.locking(candado.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:
        pass
    except OSError:
        enviar({"tipo": "error", "texto": "La mano bebé ya está abierta en otra ventana. Ciérrala antes de abrir esta."})
        return
    try:
        red = internet.cargar()
        if red is None:
            enviar({"tipo": "aviso", "texto": "Todavía no están las manos de internet (HaGRID). Aprende solo con su cuerpo."})
        mapa = cuerpo.Mapa()
        b = _cargar(red)
        centros = (np.stack([mapa.proyectar(np.median(red["pose"][red["clase"] == c], 0))[0] for c in range(len(red["claves"]))])
                   if red is not None else np.zeros((0, 2)))
        enviar({"tipo": "mapa", "nube": mapa.nube.astype(np.float32), "gestos": centros,
                "nombres_gestos": [] if red is None else [str(n) for n in red["nombres"]], "lo": mapa.lo, "hi": mapa.hi,
                "musculos": cuerpo.MUSCULOS, "fuente": "" if red is None else str(red["fuente"]),
                "n_manos": 0 if red is None else int(len(red["pose"]))})
        velocidad, pausa, detener = 1.0, False, False
        ultimo_envio = ultimo_estado = ultimo_guardado = 0.0
        while not detener:
            while True:
                try:
                    c = entrada.get_nowait()
                except queue.Empty:
                    break
                if c["cmd"] == "velocidad":
                    velocidad = float(c["valor"])
                elif c["cmd"] == "pausa":
                    pausa = bool(c["valor"])
                elif c["cmd"] == "renacer":
                    b = Bebe(red)
                    _guardar(b)
                    enviar({"tipo": "renacio"})
                elif c["cmd"] == "detener":
                    detener = True
            ahora = time.time()
            if pausa:
                if ahora - ultimo_estado > 0.5:
                    enviar({"tipo": "estado", "pausa": True, **b.resumen(mapa, red)})
                    ultimo_estado = ahora
                time.sleep(0.1)
                continue
            r = b.intento(mapa, red)
            if r["descubrimiento"]:
                h = r["descubrimiento"]
                enviar({"tipo": "descubrimiento", **h})
                if h["tipo"] == "aprendido":
                    threading.Thread(target=_foto, args=(h, salida), daemon=True).start()
            intervalo = 0.6 / velocidad if velocidad < 1000 else 0.0
            if ahora - ultimo_envio >= max(intervalo, 0.08):
                enviar({"tipo": "intento", **r, "velocidad": velocidad})
                ultimo_envio = ahora
            if ahora - ultimo_estado > 0.5:
                enviar({"tipo": "estado", "pausa": False, "velocidad": velocidad, **b.resumen(mapa, red)})
                ultimo_estado = ahora
            if ahora - ultimo_guardado > 60:
                _guardar(b)
                ultimo_guardado = ahora
            if intervalo:
                time.sleep(intervalo)
        _guardar(b)
        enviar({"tipo": "detenido"})
    except Exception:
        enviar({"tipo": "error", "texto": traceback.format_exc()[-2500:]})
