"""
Proceso de la neuroevolución (lo lanza la ventana "Neuroevolución")
===================================================================

Cada generación:
  1. las 150 manos recorren 6 pistas (6 personas distintas del gemelo);
     la primera es la que se ve en la ventana
  2. fitness = segundos que aguanta sin chocar (promedio) + 30 × acierto
  3. la mejor rinde el examen: 8 personas que ninguna mano vio nunca
     (acierto, pseudo-R² de McFadden y R² de la pose de los dedos)
  4. selección por especies, cruce y mutación: nace la siguiente generación

Con "ver cada generación" el proceso espera a que la ventana termine de
mostrar una antes de mandar la siguiente (como el video de los carros).
Todo se guarda en %USERPROFILE%\\gemelo_mano\\motor\\neuro y al reabrir sigue.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import pickle
import queue
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import ia_mano as ia
from .. import config, datos_emg as de, metricas
from . import neat, pista

RAIZ = config.RAIZ_DATOS / "motor" / "neuro"
N_POBLACION, N_VISIBLES, PISTAS_POR_GEN = 150, 30, 6
PESO_ACIERTO = 30.0         # fitness = segundos que aguanta + 30 × acierto (el acierto es una señal más suave para evolucionar)
BANCO, EXAMEN = 40, 8
MEJORA_CAMPEON = 0.005


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _poses(probs):
    """Lo que harían los servos: la pose esperada, suavizada como un motor real (no salta de golpe)."""
    obj = probs @ de.POSES_6
    out = np.empty_like(obj)
    out[..., 0, :] = obj[..., 0, :]
    for t in range(1, obj.shape[-2]):
        out[..., t, :] = out[..., t - 1, :] + 0.45 * (obj[..., t, :] - out[..., t - 1, :])
    return out


class Neuro:
    def __init__(self, salida, entrada):
        self.salida, self.entrada = salida, entrada
        RAIZ.mkdir(parents=True, exist_ok=True)
        (RAIZ / "campeones").mkdir(exist_ok=True)
        self.pool = ProcessPoolExecutor(max_workers=3)
        self.modo, self.pausa, self.detener, self.pide_siguiente, self.recalcular = "ver", False, False, True, False
        self.pob = self._cargar()

    # ---------- comunicación ----------
    def enviar(self, m, esperar=False):
        try:
            self.salida.put(m, timeout=5) if esperar else self.salida.put_nowait(m)
        except queue.Full:
            pass

    def aviso(self, texto):
        self.enviar({"tipo": "aviso", "texto": texto})

    def atender(self, bloquear=False):
        while True:
            try:
                c = self.entrada.get(timeout=0.1) if bloquear else self.entrada.get_nowait()
            except queue.Empty:
                return
            bloquear = False
            if c["cmd"] == "siguiente":
                self.pide_siguiente = True
            elif c["cmd"] == "modo":
                self.modo = c["valor"]
                self.pide_siguiente = True
            elif c["cmd"] == "pausa":
                self.pausa = bool(c["valor"])
            elif c["cmd"] == "reiniciar":
                self._respaldar()
                self.pob = neat.Poblacion(N_POBLACION)
                self.pob.anotar("Vuelve a empezar desde cero: 150 redes casi vacías")
                self.pide_siguiente, self.recalcular = True, True
            elif c["cmd"] == "detener":
                self.detener = True

    # ---------- guardar ----------
    def _cargar(self):
        ruta = RAIZ / "poblacion.pkl"
        if ruta.exists():
            try:
                with open(ruta, "rb") as f:
                    pob = pickle.load(f)
                pob.anotar(f"Retoma la generación {pob.gen} con {len(pob.especies)} especies")
                return pob
            except Exception as e:
                self._respaldar()
                self.aviso(f"No se pudo leer la población guardada ({e}); quedó como respaldo y empieza de cero.")
        pob = neat.Poblacion(N_POBLACION)
        pob.anotar("Generación 0: 150 redes casi vacías (cada gesto conectado a 4 entradas al azar). Nadie sabe nada.")
        return pob

    def _respaldar(self):
        ruta = RAIZ / "poblacion.pkl"
        if ruta.exists():
            os.replace(ruta, RAIZ / f"poblacion_respaldo_{time.strftime('%Y%m%d_%H%M%S')}.pkl")

    def guardar(self):
        tmp = RAIZ / "poblacion.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.pob, f)
        os.replace(tmp, RAIZ / "poblacion.pkl")

    # ---------- pistas ----------
    def preparar_pistas(self):
        ruta = RAIZ / "pistas.pkl"
        self.banco, self.examen = [], []
        if ruta.exists():
            with open(ruta, "rb") as f:
                d = pickle.load(f)
            self.banco, self.examen = d["banco"], d["examen"]
        faltan_ex = [7000 + i for i in range(len(self.examen), EXAMEN)]
        faltan = [int(s) for s in np.random.default_rng().integers(1 << 30, size=max(0, BANCO - len(self.banco)))]
        if faltan_ex or faltan:
            self.aviso(f"Preparando las pistas: simulando {len(faltan_ex) + len(faltan)} personas con sus electrodos…")
            self.examen += list(self.pool.map(pista.recorrido, faltan_ex))
            self.banco += list(self.pool.map(pista.recorrido, faltan, [True] * len(faltan)))
            self._guardar_pistas()
        self.nueva = None

    def _guardar_pistas(self):
        with open(RAIZ / "pistas.pkl", "wb") as f:
            pickle.dump({"banco": self.banco, "examen": self.examen}, f)

    def _renovar_pistas(self):
        """Una persona nueva por generación reemplaza a la más vieja del banco (no se aprenden las pistas de memoria)."""
        if self.nueva is not None and self.nueva.done():
            self.banco = self.banco[1:] + [self.nueva.result()]
            if self.pob.gen % 10 == 0:
                self._guardar_pistas()
            self.nueva = None
        if self.nueva is None:
            self.nueva = self.pool.submit(pista.recorrido, int(np.random.default_rng().integers(1 << 30)), True)

    # ---------- una generación ----------
    def evaluar(self):
        pob, rng = self.pob, self.pob.rng
        pistas = [self.banco[int(i)] for i in rng.choice(len(self.banco), PISTAS_POR_GEN, replace=False)]
        X = np.concatenate([p["X"] for p in pistas])
        cortes = np.cumsum([0] + [len(p["X"]) for p in pistas])
        logits = np.stack([g.activar(X) for g in pob.genomas])                     # (G, T total, 7)
        pred = logits.argmax(-1)
        aguante, aciertos = [], []
        for k, p in enumerate(pistas):
            pr = pred[:, cortes[k]:cortes[k + 1]]
            aguante.append(pista.choques(pr, p) * pista.DT)
            aciertos.append((pr == p["gesto"][None])[:, p["estable"]])
        aguante = np.stack(aguante, 1)                                             # (G, pistas) en segundos
        acierto = np.concatenate(aciertos, 1).mean(1)
        for i, g in enumerate(pob.genomas):
            g.fitness = float(aguante[i].mean() + PESO_ACIERTO * acierto[i])
            g.aguante, g.acierto = float(aguante[i, 0]), float(acierto[i])
        return pistas[0], logits[:, :cortes[1]], aguante, acierto

    def rendir_examen(self, g):
        probs, ys, poses, objetivos, aguante = [], [], [], [], []
        for p in self.examen:
            pr = _softmax(g.activar(p["X"]))
            probs.append(pr[p["estable"]])
            ys.append(p["gesto"][p["estable"]])
            poses.append(_poses(pr))
            objetivos.append(p["pose"])
            aguante.append(pista.choques(pr.argmax(1)[None], p)[0] * pista.DT)
        probs, ys = np.concatenate(probs), np.concatenate(ys).astype(int)
        return {"acierto": float(np.mean(probs.argmax(1) == ys)),
                "pseudo_r2": metricas.pseudo_r2_mcfadden(ys, probs, np.full(ia.NUM_GESTOS, 1 / ia.NUM_GESTOS)),
                "r2_pose": metricas.r2(np.concatenate(objetivos), np.concatenate(poses))["r2"],
                "aguante": float(np.mean(aguante)), "largo": float(len(self.examen[0]["X"]) * pista.DT)}

    def _coronar(self, g, m):
        pob = self.pob
        pob.n_campeones += 1
        n = pob.n_campeones
        carpeta = RAIZ / "campeones" / f"c{n:04d}_g{pob.gen:05d}"
        carpeta.mkdir(parents=True, exist_ok=True)
        neat.exportar_c(g, carpeta / "red_neat.h", f"Campeón #{n}, generación {pob.gen}: acierto {m['acierto'] * 100:.1f} % "
                                                   f"en {EXAMEN} personas nuevas, pseudo-R² {m['pseudo_r2']:.3f}")
        (carpeta / "ficha.json").write_text(json.dumps({
            "campeon": n, "generacion": pob.gen, "especie": g.especie, "metricas": m,
            "neuronas_ocultas": len(g.ocultas), "conexiones": g.n_conexiones,
            "genoma": [[i, *c] for i, c in sorted(g.conexiones.items())]}, ensure_ascii=False, indent=1), encoding="utf-8")
        pob.campeon, pob.campeon_metricas = g.hijo(g.id), dict(m, n=n, gen=pob.gen)
        pob.anotar(f"★ Nuevo campeón #{n} (especie {g.especie}): {m['acierto'] * 100:.1f} % de acierto y aguanta "
                   f"{m['aguante']:.1f} s de {m['largo']:.0f} con personas que nunca vio · R² pose {m['r2_pose']:.3f}")

    def _red(self, g, X):
        """La red de la mejor, para dibujarla con sus activaciones en la pista que se ve."""
        _, v = g.activar(X, todas=True)
        nodos = list(range(N_ENT_SESGO)) + sorted(g.ocultas) + neat.SALIDAS
        prof = {n: 0 for n in range(N_ENT_SESGO)}
        for n, entradas in g.plan():
            prof[n] = 1 + max([prof.get(a, 0) for a, _ in entradas], default=0)
        maxp = max([prof[n] for n in g.ocultas], default=0) + 1
        for s in neat.SALIDAS:
            prof[s] = maxp
        return {"nodos": nodos, "prof": [prof.get(n, 0) for n in nodos], "maxp": maxp,
                "conexiones": [[a, b, w] for a, b, w, act in g.conexiones.values() if act],
                "act": np.stack([np.broadcast_to(v.get(n, 0.0), (len(X),)) for n in nodos]).astype(np.float16)}

    def generacion(self):
        t0 = time.time()
        pob = self.pob
        visible, logits, aguante, acierto = self.evaluar()
        orden = np.argsort([-g.fitness for g in pob.genomas])
        mejor = pob.genomas[orden[0]]
        m = self.rendir_examen(mejor)
        if pob.campeon_metricas is None or m["acierto"] > pob.campeon_metricas["acierto"] + MEJORA_CAMPEON:
            self._coronar(mejor, m)
        if mejor.especie != getattr(pob, "especie_lider", None):
            if pob.gen:
                pob.anotar(f"La especie {mejor.especie} toma la delantera: su mejor red tiene {len(mejor.ocultas)} "
                           f"neuronas ocultas y {mejor.n_conexiones} conexiones")
            pob.especie_lider = mejor.especie
        # 30 manos que representan a toda la población: la mejor y una de cada tramo del ranking
        elegidas = np.unique(np.linspace(0, len(orden) - 1, N_VISIBLES).round().astype(int))
        idx = orden[elegidas]
        especies = {sid: len(e["miembros"]) for sid, e in pob.especies.items()}
        fila = {"gen": pob.gen, "mejor": float(aguante[orden[0]].mean()), "media": float(aguante.mean()),
                "acierto": m["acierto"], "pseudo_r2": m["pseudo_r2"], "r2_pose": m["r2_pose"],
                "aguante_examen": m["aguante"], "especies": especies,
                "ocultas": len(mejor.ocultas), "conexiones": mejor.n_conexiones}
        pob.historial = (pob.historial + [fila])[-600:]
        msg = {"tipo": "generacion", "gen": pob.gen, "segundos": round(time.time() - t0, 2),
               "pista": {k: visible[k] for k in ("gesto", "estable", "pose", "nivel", "adc")}, "dt": pista.DT,
               "manos": [{"rank": int(r + 1), "id": pob.genomas[i].id, "especie": pob.genomas[i].especie,
                          "choque": int(round(pob.genomas[i].aguante / pista.DT)), "aguante": pob.genomas[i].aguante,
                          "acierto": pob.genomas[i].acierto, "ocultas": len(pob.genomas[i].ocultas),
                          "conexiones": pob.genomas[i].n_conexiones} for r, i in zip(elegidas, idx)],
               "poses": _poses(_softmax(logits[idx])).astype(np.float32),
               "pred": logits[idx].argmax(-1).astype(np.int8),
               "red": self._red(mejor, visible["X"]), "mejor": {"especie": mejor.especie, "id": mejor.id, **m},
               "campeon": pob.campeon_metricas, "historial": pob.historial[-300:], "eventos": pob.eventos[-60:],
               "n_poblacion": len(pob.genomas), "n_especies": len(pob.especies), "umbral": pob.umbral}
        pob.reproducir()
        self._renovar_pistas()
        self.guardar()
        return msg

    def correr(self):
        try:
            self.preparar_pistas()
            listo = self.generacion()
            while not self.detener:
                self.atender()
                if self.recalcular:                                # volvió a empezar: lo precalculado ya no sirve
                    listo, self.recalcular = self.generacion(), False
                if self.pausa or (self.modo == "ver" and not self.pide_siguiente):
                    self.atender(bloquear=True)
                    continue
                self.enviar(listo, esperar=True)
                self.pide_siguiente = False
                listo = self.generacion()                          # la siguiente se calcula mientras la ventana muestra esta
                if self.modo == "rapido":
                    time.sleep(0.05)
        except Exception:
            self.enviar({"tipo": "error", "texto": traceback.format_exc()[-2500:]})
        finally:
            self.guardar()
            self.pool.shutdown(cancel_futures=True)
            self.enviar({"tipo": "detenido"})


N_ENT_SESGO = neat.N_ENT + 1


def proceso(salida, entrada):
    Neuro(salida, entrada).correr()
