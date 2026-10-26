"""
La mano bebé: aprende sola qué es capaz de hacer su cuerpo
==========================================================

Nadie le da gestos, listas ni metas. Como un bebé:

  1. BALBUCEO   activa sus 10 músculos al azar y mira dónde quedan sus dedos:
                así conoce su propio cuerpo.
  2. EXPLORA    se pone sus propias metas y decide sola cómo aprender:
                  inventar    se imagina una pose cualquiera e intenta hacerla
                  ir más allá parte de algo que ya sabe y lo lleva más lejos
                  practicar   repite algo nuevo que le salió, hasta dominarlo
                  balbucear   vuelve a mover músculos al azar
                Prefiere la estrategia que más le está enseñando en ese momento
                (curiosidad: novedad + progreso), y eso cambia con el tiempo.

Una HABILIDAD es una pose que descubrió ella misma y que ya sabe repetir a
voluntad (3 veces seguidas). El repertorio no tiene tope: cuando ya no
encuentra nada nuevo, empieza a distinguir detalles más finos (sube de
"finura") y sigue descubriendo.

Internet es un espejo, no una lista de tareas: de cada cosa que hace mira si
alguna persona real hace algo parecido (HaGRID v2, 19.800 manos) y lo anota.

Su cerebro son dos redes que aprenden en línea, intento a intento:
  modelo inverso  pose que se propone → qué músculos activar
  modelo directo  músculos → qué pose va a salir
R² del control: pose lograda frente a la que se propuso (300 poses de examen
que su cuerpo puede hacer). R² del cuerpo: lo que el modelo directo predice
frente a lo que pasa.

Corre en la CPU (NumPy + scikit-learn): no le quita la GPU a la competencia.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
VERSION = 3
FIN_BALBUCEO = 300          # ~3 minutos en tiempo real
MEMORIA = 30_000
ESTRATEGIAS = ["inventar", "frontera", "practicar", "balbucear"]
RADIO_INICIAL, RADIO_MIN, FINURA = 0.35, 0.03, 0.85     # qué tan distinta tiene que ser una pose para ser "nueva"
SATURACION = 1500           # intentos sin nada nuevo antes de afinar la mirada
REPETIR, MAX_PRACTICA, MAX_CANDIDATAS = 3, 20, 24
FOTOS_MAX = 400             # después, solo fotos de lo que también hacen personas en internet


class Bebe:
    def __init__(self, semilla=None):
        self.version = VERSION
        self.semilla = int(time.time()) if semilla is None else semilla
        self.rng = np.random.default_rng(self.semilla)
        self.inverso = MLPRegressor(hidden_layer_sizes=(96, 96), activation="tanh", learning_rate_init=3e-3, random_state=1)
        self.directo = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", learning_rate_init=3e-3, random_state=2)
        self._A = np.zeros((MEMORIA, 10), np.float32)             # búfer circular de intentos
        self._P = np.zeros((MEMORIA, 6), np.float32)
        self.n_mem, self.pos = 0, 0
        self.intentos, self.listo = 0, False
        self.nacimiento = time.time()
        self._nuevo_explorador()

    def _nuevo_explorador(self):
        self.celdas_vistas = getattr(self, "celdas_vistas", set())
        self.habilidades, self.H = [], np.zeros((0, 6))
        self.candidatas, self.casualidades = [], 0
        self.radio, self.nivel, self.ultimo_hallazgo = RADIO_INICIAL, 1, self.intentos
        self.valor = np.full(len(ESTRATEGIAS), 0.3)                # cuánto le enseña cada estrategia (lo aprende ella)
        self.uso = np.zeros(len(ESTRATEGIAS), int)
        self.diario, self.curva = [], []                           # curva: (intento, R² control, R² cuerpo, cobertura, habilidades, nivel)
        self.prueba = cuerpo.pose(cuerpo.balbuceo(np.random.default_rng(11), 300))

    def anotar(self, texto):
        self.diario = (self.diario + [{"intento": self.intentos, "texto": texto}])[-200:]

    # ---------- decidir (curiosidad) ----------
    def preferencias(self):
        disp = np.ones(len(ESTRATEGIAS), bool)
        disp[2] = bool(self.candidatas or self.habilidades)
        p = np.exp((self.valor - self.valor.max()) / 0.08) * disp
        return 0.85 * p / p.sum() + 0.15 * disp / disp.sum()

    def _novedad(self, p):
        return float(np.min(cuerpo.distancia(self.H, p))) if len(self.H) else np.inf

    def _umbral(self):
        return max(0.45 * self.radio, 0.045)

    @staticmethod
    def _recortar(p):
        return np.r_[np.clip(p[:5], 0, 1), np.clip(p[5], -1, 1)]

    @property
    def etapa(self):
        return "balbuceo" if self.intentos < FIN_BALBUCEO or not self.listo else "explora"

    # ---------- un intento ----------
    def intento(self, mapa, red):
        k = 3 if self.etapa == "balbuceo" else int(self.rng.choice(len(ESTRATEGIAS), p=self.preferencias()))
        est = ESTRATEGIAS[k]
        meta, cand, hab = None, None, None
        if est == "balbucear":
            a = cuerpo.balbuceo(self.rng)[0]
            texto = "mueve músculos al azar para ver qué pasa"
        else:
            if est == "inventar":
                meta, ruido = np.r_[self.rng.random(5), self.rng.uniform(-1, 1)], 0.15
                texto = "se imagina una pose cualquiera e intenta hacerla"
            elif est == "frontera":
                base = self.H[self.rng.integers(len(self.H))] if len(self.H) else self._P[self.rng.integers(self.n_mem)]
                d = self.rng.normal(size=6)
                meta, ruido = self._recortar(base + d / np.linalg.norm(d) * self.radio * self.rng.uniform(1, 2)), 0.12
                texto = "parte de algo que ya sabe y lo lleva más lejos"
            elif self.candidatas and (not self.habilidades or self.rng.random() < 0.75):
                cand = self.candidatas[0]
                meta, ruido = cand["pose"], 0.04
                texto = (f"repite algo nuevo que le salió en el intento {cand['intento']:,} "
                         f"(práctica {cand['intentos'] + 1})").replace(",", ".")
            else:
                peores = sorted(self.habilidades, key=lambda h: -h["error"])[:5]
                hab = peores[int(self.rng.integers(len(peores)))]
                meta, ruido = np.asarray(hab["pose"]), 0.04
                texto = f"repasa su habilidad #{hab['n']} para hacerla mejor"
            a = np.clip(self.inverso.predict(meta[None])[0] + self.rng.normal(0, ruido * self.rng.random(), 10), 0, 1)
        p = cuerpo.pose(a, self.rng)[0]
        self.intentos += 1
        self._A[self.pos], self._P[self.pos] = a, p
        self.pos = (self.pos + 1) % MEMORIA
        self.n_mem = min(self.n_mem + 1, MEMORIA)
        z = mapa.proyectar(p)[0]
        self.celdas_vistas.add(int(mapa.celda(z[None])[0]))
        error = float(cuerpo.distancia(p, meta)) if meta is not None else np.nan
        recompensa, nueva = self._valorar(est, p, error, cand, hab, red)
        self.valor[k] += 0.03 * (recompensa - self.valor[k])
        self.uso[k] += 1
        self._aprender()
        if self.intentos % 100 == 0 and self.listo:
            self._medir(mapa)
        if self.intentos - self.ultimo_hallazgo > SATURACION and not self.candidatas and self.radio > RADIO_MIN:
            self.radio, self.nivel, self.ultimo_hallazgo = max(self.radio * FINURA, RADIO_MIN), self.nivel + 1, self.intentos
            self.anotar(f"Ya no encuentra nada nuevo a simple vista: ahora distingue detalles más finos "
                        f"(finura {self.nivel}: poses distintas desde {self.radio:.2f})")
        return {"n": self.intentos, "etapa": self.etapa, "estrategia": est, "texto": texto,
                "meta": None if meta is None else np.asarray(meta).tolist(), "alcanzado": p.tolist(), "activacion": a.tolist(),
                "error": error, "z": z.tolist(), "zmeta": None if meta is None else mapa.proyectar(meta)[0].tolist(),
                "nueva": nueva}

    def _valorar(self, est, p, error, cand, hab, red):
        """Lo que le enseñó este intento: novedad, una habilidad nueva o progreso en algo que practica."""
        nov = self._novedad(p)
        r = 0.3 * min(nov / self.radio, 2.0) / 2.0
        nueva = None
        if cand is not None:
            cand["errores"].append(error)
            cand["intentos"] += 1
            ult = cand["errores"][-REPETIR:]
            if len(ult) == REPETIR and max(ult) < self._umbral():
                self.candidatas.remove(cand)
                if self._novedad(cand["pose"]) > 0.8 * self.radio:
                    nueva = self._agregar(cand, red=red)
                    r += 1.0
            elif cand["intentos"] >= MAX_PRACTICA:
                self.candidatas.remove(cand)                       # fue casualidad: todavía no la sabe repetir
                self.casualidades += 1
        elif hab is not None:
            antes = hab["error"]
            hab["error"] = 0.8 * antes + 0.2 * error
            hab["practicas"] += 1
            r += float(np.clip((antes - hab["error"]) * 5, 0, 0.5))
        if (cand is None and nov > self.radio and len(self.candidatas) < MAX_CANDIDATAS
                and all(cuerpo.distancia(c["pose"], p) > 0.5 * self.radio for c in self.candidatas)):
            self.candidatas.append({"pose": p.copy(), "intento": self.intentos, "origen": est, "errores": [], "intentos": 0})
            r += 0.5
        return r, nueva

    def _agregar(self, cand, red=None):
        pose = np.asarray(cand["pose"], float)
        parecido = None
        if red is not None:
            i, d = internet.espejo(red, pose)
            if i is not None:
                parecido = (str(red["nombres"][red["clase"][i]]), red["landmarks_vista"][i].tolist(), d)
        h = {"n": len(self.habilidades) + 1, "pose": pose.tolist(), "nombre": cuerpo.describir_libre(pose),
             "intento": self.intentos, "le_salio": cand["intento"], "origen": cand["origen"], "nivel": self.nivel,
             "error": float(np.mean(cand["errores"][-REPETIR:])), "practicas": len(cand["errores"]),
             "parecido": parecido[0] if parecido else None, "obs": parecido[1] if parecido else None,
             "foto": None, "t": time.time()}
        self.habilidades.append(h)
        self.H = np.vstack([self.H, pose[None]])
        self.ultimo_hallazgo = self.intentos
        self.anotar(f"★ Habilidad #{h['n']}: {h['nombre'].lower()}"
                    + (f" · en internet hay personas que hacen algo así («{h['parecido']}»)" if h["parecido"] else ""))
        return h

    def _aprender(self, lote=64):
        if self.n_mem < lote:
            return
        i = self.rng.integers(0, self.n_mem, lote)
        self.inverso.partial_fit(self._P[i], self._A[i])
        self.directo.partial_fit(self._A[i], self._P[i])
        self.listo = True

    def _medir(self, mapa):
        cob = len(self.celdas_vistas & set(mapa.alcanzables.tolist())) / len(mapa.alcanzables)
        A_al = cuerpo.balbuceo(np.random.default_rng(5), 300)
        r2_cuerpo = float(r2_score(cuerpo.pose(A_al), self.directo.predict(A_al)))
        logrado = cuerpo.pose(np.clip(self.inverso.predict(self.prueba), 0, 1))
        r2_control = float(r2_score(self.prueba, logrado))
        self.curva.append((self.intentos, r2_control, r2_cuerpo, cob, len(self.habilidades), self.nivel))

    def resumen(self, mapa):
        u = self.curva[-1] if self.curva else (0, np.nan, np.nan, 0.0, 0, 1)
        ult = (self.pos - 1 - np.arange(min(self.n_mem, 4000))) % MEMORIA
        sin_obs = [{k: v for k, v in h.items() if k != "obs"} for h in self.habilidades[-150:]]
        return {"intentos": self.intentos, "etapa": self.etapa, "r2_control": u[1], "r2_cuerpo": u[2], "cobertura": u[3],
                "curva": np.array(self.curva[-600:], float) if self.curva else np.zeros((0, 6)),
                "puntos": mapa.proyectar(self._P[ult]).astype(np.float32) if self.n_mem else np.zeros((0, 2)),
                "z_hab": mapa.proyectar(self.H).astype(np.float32) if len(self.H) else np.zeros((0, 2)),
                "z_cand": mapa.proyectar(np.array([c["pose"] for c in self.candidatas])) if self.candidatas else np.zeros((0, 2)),
                "habilidades": sin_obs, "n_habilidades": len(self.habilidades), "candidatas": len(self.candidatas),
                "parecidos": sum(h["parecido"] is not None for h in self.habilidades), "casualidades": self.casualidades,
                "nivel": self.nivel, "radio": self.radio, "preferencias": self.preferencias().tolist(), "uso": self.uso.tolist(),
                "diario": self.diario[-40:], "edad_s": time.time() - self.nacimiento}


# ------------------------------------------------------------------
# Proceso que lanza la ventana "Entrenamiento libre"
# ------------------------------------------------------------------
def _cargar():
    """Retoma lo último guardado. Si es de una versión anterior (que imitaba gestos), la guarda aparte y nace de cero."""
    ruta = RAIZ / "bebe.pkl"
    if not ruta.exists():
        return Bebe(), None
    try:
        with open(ruta, "rb") as f:
            b = pickle.load(f)
        if getattr(b, "version", 1) == VERSION:
            return b, None
        motivo = f"la versión {getattr(b, 'version', 1)} imitaba gestos de internet"
    except Exception as e:
        motivo = f"no se pudo leer ({e})"
    respaldo = RAIZ / f"bebe_respaldo_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    os.replace(ruta, respaldo)
    return Bebe(), f"Nace de cero: lo guardado antes ({motivo}) quedó aparte en {respaldo.name}."


def _guardar(b):
    RAIZ.mkdir(parents=True, exist_ok=True)
    tmp = RAIZ / "bebe.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(b, f)
    os.replace(tmp, RAIZ / "bebe.pkl")


def _foto(h, salida):
    """Foto de una habilidad: la mano bebé y, si existe, la persona de internet que hace algo parecido."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from ..motor import fotos
        carpeta = RAIZ / "habilidades"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / f"habilidad_{h['n']:04d}.png"
        fotos.foto_gesto(ruta, np.array(h["pose"]), 0, f"Habilidad #{h['n']}",
                         f"intento {h['intento']:,} · error {h['error']:.3f}".replace(",", "."))
        if h.get("obs") is not None:
            fig = Figure(figsize=(7.6, 4.6), dpi=110)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(mpimg.imread(ruta))
            ax.set_axis_off()
            ax2 = fig.add_subplot(1, 2, 2)
            L = np.asarray(h["obs"])
            for a, b in internet.CONEXIONES:
                ax2.plot(L[[a, b], 0], L[[a, b], 1], color="#2563c9", lw=3, solid_capstyle="round")
            ax2.scatter(L[:, 0], L[:, 1], s=18, color="white", edgecolors="#2563c9", zorder=3)
            ax2.set_aspect("equal")
            ax2.set_axis_off()
            ax2.set_title(f"En internet se parece a «{h['parecido']}»", fontsize=10, color="#52514e")
            fig.subplots_adjust(0.02, 0.02, 0.98, 0.92, 0.05)
            fig.savefig(ruta)
        h["foto"] = str(ruta)
        salida.put({"tipo": "foto", "n": h["n"], "foto": str(ruta)})
    except Exception as e:
        salida.put({"tipo": "aviso", "texto": f"No se pudo tomar la foto: {e}"})


def _fotografo(cola, salida):
    while True:
        h = cola.get()
        if h is None:
            return
        _foto(h, salida)


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
        mapa = cuerpo.Mapa()
        b, aviso = _cargar()
        if aviso:
            enviar({"tipo": "aviso", "texto": aviso})
        enviar({"tipo": "mapa", "nube": mapa.nube.astype(np.float32), "lo": mapa.lo, "hi": mapa.hi,
                "musculos": cuerpo.MUSCULOS, "fuente": "" if red is None else str(red["fuente"]),
                "n_manos": 0 if red is None else int(len(red["pose"]))})
        if b.listo and not b.curva:
            b._medir(mapa)
        salida.put({"tipo": "estado", "pausa": False, **b.resumen(mapa)})   # lo guardado se ve de inmediato
        fotos_pendientes = queue.Queue()
        threading.Thread(target=_fotografo, args=(fotos_pendientes, salida), daemon=True).start()
        for h in b.habilidades:                                             # fotos que faltaron (se cerró antes)
            if not h.get("foto") and (h["n"] <= FOTOS_MAX or h.get("obs") is not None):
                fotos_pendientes.put(h)
        velocidad, pausa, detener = 1.0, False, False
        ultimo_envio = ultimo_estado = 0.0
        ultimo_guardado = time.time()
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
                    _guardar(b)
                elif c["cmd"] == "renacer":
                    b = Bebe()
                    fotos_pendientes = queue.Queue()                    # las fotos pendientes eran de la vida anterior
                    threading.Thread(target=_fotografo, args=(fotos_pendientes, salida), daemon=True).start()
                    _guardar(b)
                    enviar({"tipo": "renacio"})
                elif c["cmd"] == "detener":
                    detener = True
            ahora = time.time()
            if pausa:
                if ahora - ultimo_estado > 0.5:
                    enviar({"tipo": "estado", "pausa": True, **b.resumen(mapa)})
                    ultimo_estado = ahora
                time.sleep(0.1)
                continue
            r = b.intento(mapa, red)
            if r["nueva"]:
                h = r["nueva"]
                enviar({"tipo": "habilidad", **{k: v for k, v in h.items() if k != "obs"}})
                if h["n"] <= FOTOS_MAX or h["obs"] is not None:
                    fotos_pendientes.put(h)
            intervalo = 0.6 / velocidad if velocidad < 1000 else 0.0
            if ahora - ultimo_envio >= max(intervalo, 0.08):
                i, d = internet.espejo(red, r["alcanzado"], umbral=0.25) if red is not None else (None, np.inf)
                enviar({"tipo": "intento", **{k: v for k, v in r.items() if k != "nueva"}, "velocidad": velocidad,
                        "espejo": None if i is None else red["landmarks_vista"][i].tolist(),
                        "espejo_nombre": None if i is None else str(red["nombres"][red["clase"][i]]), "espejo_d": d})
                ultimo_envio = ahora
            if ahora - ultimo_estado > 0.5:
                enviar({"tipo": "estado", "pausa": False, "velocidad": velocidad, **b.resumen(mapa)})
                ultimo_estado = ahora
            if ahora - ultimo_guardado > 20:                        # si se cierra a la fuerza, pierde a lo sumo 20 s
                _guardar(b)
                ultimo_guardado = ahora
            if intervalo:
                time.sleep(intervalo)
        _guardar(b)
        fotos_pendientes.put(None)
        enviar({"tipo": "detenido"})
    except Exception:
        enviar({"tipo": "error", "texto": traceback.format_exc()[-2500:]})
