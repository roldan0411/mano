"""
Motor del gemelo: entrena sin parar mientras la aplicación de escritorio está abierta
=====================================================================================

  procesos de simulación ──► lotes en disco ──► memoria en la GPU
                                                   │
                     10 manos entrenan por turnos ◄┘
                                │
          examen fijo ──► ranking ──► las peores copian y mutan ──► ¿nueva campeona?
                                                                       │
                              modelo + .h para el ESP32 + fotos de los 7 gestos

Corre en un proceso aparte que lanza app_gemelo.py y le habla por dos colas:
  salida   ← estado, demostración, pesos de las 10 manos (una vez por generación)
  entrada  → pausar / reanudar, cambiar el amplificador, detener
"""

import json
import logging
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")       # cada proceso de simulación: ~110 MB en vez de ~850 MB

import numpy as np
import psutil
import torch

from .. import config
from . import fotos, gemelo2
from .almacen import Almacen
from .poblacion import DISPOSITIVO, Mano, genoma_aleatorio
import ia_mano as ia
import simulador as sm

logging.getLogger("absl").setLevel(logging.ERROR)

N_MANOS = 10
PERSONAS_POR_LOTE, SESIONES, RONDAS = 8, 2, 3
PASOS_POR_GENERACION, LOTE = 150, 512
MEMORIA_MAX = 800_000
REEMPLAZOS = 3
CADA_INMIGRANTE = 12            # cada 12 generaciones entra una mano nueva al azar (diversidad)
MEJORA_CAMPEONA = 0.003         # el puntaje debe superar a la campeona por esto


class Motor:
    def __init__(self, salida=None, entrada=None):
        config.crear_carpetas()
        self.salida, self.entrada = salida, entrada
        self.alm = Almacen()
        self.rng = np.random.default_rng(int(time.time()))
        self.gen_torch = torch.Generator(device=DISPOSITIVO).manual_seed(int(time.time()))
        self.amp = self._cargar_amplificador()
        self.pool = ProcessPoolExecutor(max_workers=4)
        self.pausa, self.detener = False, False
        self.eventos, self.en_vuelo, self.demo_futura = [], [], None
        self.manos, self.version_demo = [], 0
        self.gen = int(self.alm.sql("SELECT COALESCE(MAX(id), 0) FROM generaciones", uno=True)[0])
        fila = self.alm.sql("SELECT id, puntaje, gen, mano, acierto, pseudo_r2, r2_pose, carpeta FROM campeonas "
                            "ORDER BY id DESC LIMIT 1", uno=True)
        self.campeona = ({"id": fila[0], "puntaje": fila[1], "gen": fila[2], "mano": fila[3], "carpeta": fila[7],
                          "metricas": {"puntaje": fila[1], "acierto": fila[4], "pseudo_r2": fila[5], "r2_pose": fila[6]}}
                         if fila else {"id": 0, "puntaje": 0.0})
        self.n_persona = int(self.alm.sql("SELECT COALESCE(SUM(personas), 0) FROM lotes", uno=True)[0]) + 1000

    # ---------- comunicación con la ventana ----------
    def enviar(self, tipo, **datos):
        if self.salida is None:
            if tipo == "estado":
                print(f"[{time.strftime('%H:%M:%S')}] {datos.get('mensaje', '')}", flush=True)
            return
        try:
            self.salida.put_nowait({"tipo": tipo, **datos})
        except queue.Full:
            pass                                          # la ventana va atrasada: se salta esta actualización

    def aviso(self, fase, mensaje):
        self.enviar("estado", fase=fase, mensaje=mensaje, gen=self.gen)

    def atender(self):
        while self.entrada is not None:
            try:
                c = self.entrada.get_nowait()
            except queue.Empty:
                return
            if c["cmd"] == "pausa":
                self.pausa = bool(c["valor"])
                self.anotar("Entrenamiento en pausa" if self.pausa else "Entrenamiento reanudado")
            elif c["cmd"] == "amplificador":
                self.fijar_amplificador(c["valor"])
            elif c["cmd"] == "detener":
                self.detener = True

    # ---------- amplificador ----------
    def _cargar_amplificador(self):
        ruta = self.alm.raiz / "amplificador.json"
        if ruta.exists():
            return gemelo2.Amplificador(**json.loads(ruta.read_text(encoding="utf-8")))
        return gemelo2.Amplificador()

    def fijar_amplificador(self, cambios):
        d = asdict(self.amp)
        d.update({k: float(v) for k, v in cambios.items() if k in d})
        d["rg_ohm"] = float(np.clip(d["rg_ohm"], 100, 10_000))
        d["desfase_mv_sd"] = float(np.clip(d["desfase_mv_sd"], 0, 80))
        self.amp = gemelo2.Amplificador(**d)
        (self.alm.raiz / "amplificador.json").write_text(json.dumps(d), encoding="utf-8")
        self.anotar(f"Amplificador: RG = {d['rg_ohm']:.0f} Ω (ganancia {self.amp.g_ad620:.0f}), "
                    f"desfase de electrodos ±{d['desfase_mv_sd']:.0f} mV")
        self.demo_futura = None

    def anotar(self, texto):
        self.eventos = (self.eventos + [{"t": time.time(), "gen": self.gen, "texto": texto}])[-80:]
        self.alm.evento(self.gen, texto)

    # ---------- datos ----------
    def _pedir_lote(self):
        tareas = []
        for _ in range(PERSONAS_POR_LOTE):
            self.n_persona += 1
            tareas.append((self.n_persona, int(self.rng.integers(0, len(sm.NIVELES))), SESIONES, RONDAS,
                           int(self.rng.integers(1 << 31)), asdict(self.amp)))
        return [self.pool.submit(gemelo2.simular_persona, t) for t in tareas]

    def _a_memoria(self, d):
        nuevos = (torch.as_tensor(d["X"], device=DISPOSITIVO),
                  torch.as_tensor(d["gesto"].astype(np.int64), device=DISPOSITIVO),
                  torch.as_tensor(d["pose"], device=DISPOSITIVO))
        if not hasattr(self, "X"):
            self.X, self.y, self.P = nuevos
        else:
            self.X = torch.cat([self.X, nuevos[0]])[-MEMORIA_MAX:]
            self.y = torch.cat([self.y, nuevos[1]])[-MEMORIA_MAX:]
            self.P = torch.cat([self.P, nuevos[2]])[-MEMORIA_MAX:]

    def _recoger_lotes(self, esperar=False):
        listos = list(self.en_vuelo) if esperar else [f for f in self.en_vuelo if all(x.done() for x in f)]
        for futuros in listos:
            d, cal, sat = gemelo2.unir_lote([x.result() for x in futuros])
            n, _ = self.alm.guardar_lote(d, cal, sat)
            self._a_memoria(d)
            self.anotar(f"Lote {n} guardado: {len(d['gesto']):,} ventanas de {PERSONAS_POR_LOTE} personas, "
                        f"{len(cal)} sesiones calibradas" + (f", {sat} con el AD620 saturado" if sat else ""))
            self.en_vuelo.remove(futuros)

    def preparar(self):
        self.aviso("preparando", "Generando el examen fijo (personas que ninguna mano ve)…")
        ex = self.alm.examen()
        if ex is None:
            rng = np.random.default_rng(20261023)
            tareas = [(900 + i, i % len(sm.NIVELES), 1, RONDAS, int(rng.integers(1 << 31)), asdict(gemelo2.Amplificador()))
                      for i in range(20)]
            d, _, _ = gemelo2.unir_lote(list(self.pool.map(gemelo2.simular_persona, tareas)))
            m = d["ronda"] >= 1
            self.alm.guardar_examen({k: v[m] for k, v in d.items()})
            ex = self.alm.examen()
        self.examen = ex
        self.aviso("preparando", "Cargando los datos guardados del gemelo…")
        for archivo in self.alm.lotes_en_disco(ultimos=MEMORIA_MAX // 8000 + 1):
            self._a_memoria(self.alm.leer_lote(archivo))
        while not hasattr(self, "X") or self.X.shape[0] < 30_000:
            self.aviso("preparando", "Primeros datos: simulando personas, sesiones y calibraciones…")
            self.en_vuelo.append(self._pedir_lote())
            self._recoger_lotes(esperar=True)
        self.muestra = self.X[torch.randint(0, self.X.shape[0], (20_000,), device=DISPOSITIVO)].cpu().numpy()
        for i in range(N_MANOS):
            ruta = self.alm.raiz / "poblacion" / f"mano_{i:02d}.keras"
            try:
                self.manos.append(Mano.cargar(i, ruta) if ruta.exists() else Mano(i, genoma_aleatorio(self.rng), self.muestra))
            except Exception:
                self.manos.append(Mano(i, genoma_aleatorio(self.rng), self.muestra))
        self.anotar(f"Población lista: {N_MANOS} manos · {self.X.shape[0]:,} ventanas en memoria")

    # ---------- una generación ----------
    def generacion(self):
        t0 = time.time()
        self.gen += 1
        for k, m in enumerate(self.manos):
            self.atender()
            if self.detener:
                return
            self.enviar("progreso", gen=self.gen, mano=m.idx, k=k, total=N_MANOS)
            m.entrenar(self.X, self.y, self.P, PASOS_POR_GENERACION, LOTE, self.gen_torch)
        for m in self.manos:
            m.evaluar(self.examen)
        orden = sorted(self.manos, key=lambda m: -m.metricas["puntaje"])
        for m in self.manos:
            self.alm.sql("INSERT INTO candidatas VALUES (?,?,?,?,?,?,?,?)",
                         (self.gen, m.idx, m.metricas["puntaje"], m.metricas["acierto"], m.metricas["pseudo_r2"],
                          m.metricas["r2_pose"], json.dumps(m.genoma), ""))
        if orden[0].metricas["puntaje"] > self.campeona["puntaje"] + MEJORA_CAMPEONA:
            self._coronar(orden[0])
        ranking = {m.idx: r + 1 for r, m in enumerate(orden)}        # el ranking que se ve es el del examen
        if self.gen % CADA_INMIGRANTE == 0:
            nueva = orden[-1]
            nueva.__init__(nueva.idx, genoma_aleatorio(self.rng), self.muestra)
            self.anotar(f"Entra una mano nueva al azar en el lugar {nueva.idx + 1}: "
                        f"{nueva.genoma['h1']}·{nueva.genoma['h2']} neuronas, lr {nueva.genoma['lr']:.1e}")
            perdedoras = orden[-REEMPLAZOS:-1]
        else:
            perdedoras = orden[-REEMPLAZOS:]
        for perdedora in perdedoras:
            ganadora = orden[int(self.rng.integers(0, REEMPLAZOS))]
            antes = perdedora.genoma["lr"]
            perdedora.copiar_de(ganadora, self.rng)
            self.anotar(f"Mano {perdedora.idx + 1} ({perdedora.metricas['acierto'] * 100:.1f} %) copió a la mano "
                        f"{ganadora.idx + 1} y mutó: lr {antes:.1e} → {perdedora.genoma['lr']:.1e}")
        puntajes = [m.metricas["puntaje"] for m in self.manos]
        self.alm.sql("INSERT INTO generaciones VALUES (?,?,?,?,?,?)",
                     (self.gen, time.time(), max(puntajes), float(np.mean(puntajes)), int(self.X.shape[0]),
                      time.time() - t0))
        if self.gen % 10 == 0:
            self.guardar_poblacion()
        self._publicar(ranking, time.time() - t0)

    def _coronar(self, mano):
        n = self.campeona["id"] + 1
        carpeta = self.alm.carpeta_campeona(n, self.gen)
        mano.modelo.save(carpeta / "modelo.keras")
        mano.modelo.save(carpeta / "modelo.h5")
        ia.exportar_red_c(mano.modelo_gesto(), carpeta / "modelo_red_calibrado.h")
        (carpeta / "ficha.json").write_text(json.dumps({"campeona": n, "generacion": self.gen, "mano": mano.idx + 1,
                                                        "linaje": mano.linaje, "metricas": mano.metricas,
                                                        "genoma": mano.genoma, "parametros": mano.parametros,
                                                        "entradas": "8 características calibradas × 3 ventanas"},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
        self.alm.registrar_campeona(self.gen, mano.idx, mano.metricas, mano.genoma, carpeta)
        self.campeona = {"id": n, "puntaje": mano.metricas["puntaje"], "gen": self.gen, "mano": mano.idx,
                         "metricas": dict(mano.metricas), "carpeta": str(carpeta)}
        self.anotar(f"★ Nueva campeona #{n}: mano {mano.idx + 1}, acierto {mano.metricas['acierto'] * 100:.1f} %, "
                    f"R² pose {mano.metricas['r2_pose']:.3f}")
        probs, pose = mano.predecir(self.examen["X"])       # aquí, no en el hilo: la mano sigue entrenando
        threading.Thread(target=self._fotos, args=(carpeta, probs, pose, n, self.gen), daemon=True).start()

    def _fotos(self, carpeta, probs, pose, n, gen):
        try:
            fotos.fotos_campeona(carpeta, probs, pose, self.examen, n, gen)
            self.anotar(f"Fotos de los 7 gestos de la campeona #{n} guardadas")
        except Exception as e:
            self.anotar(f"No se pudieron tomar las fotos de la campeona #{n}: {e}")

    def guardar_poblacion(self):
        for m in self.manos:
            m.guardar(self.alm.raiz / "poblacion" / f"mano_{m.idx:02d}.keras")

    def campeonas(self):
        filas = self.alm.sql("SELECT id, gen, t, mano, puntaje, acierto, pseudo_r2, r2_pose, genoma, carpeta "
                             "FROM campeonas ORDER BY id DESC LIMIT 30")
        out = []
        for f in filas:
            c = Path(f[9]) if f[9] else None
            out.append({"id": f[0], "gen": f[1], "t": f[2], "mano": f[3], "puntaje": f[4], "acierto": f[5],
                        "pseudo_r2": f[6], "r2_pose": f[7], "genoma": json.loads(f[8]), "carpeta": str(c) if c else "",
                        "fotos": [str(p) for p in sorted(c.glob("gesto_*.png"))] if c and c.exists() else []})
        return out

    # ---------- lo que ve la ventana ----------
    def _publicar(self, ranking, segundos):
        if self.demo_futura is None:
            self.demo_futura = self.pool.submit(gemelo2.demostracion, int(self.rng.integers(1 << 31)), self.amp)
        if self.demo_futura.done():
            demo = self.demo_futura.result()
            self.demo_futura = self.pool.submit(gemelo2.demostracion, int(self.rng.integers(1 << 31)), self.amp)
            preds = [m.predecir(demo["X"]) for m in self.manos]
            self.version_demo += 1
            self.enviar("demo", version=self.version_demo, gen=self.gen, X=demo["X"], gesto=demo["gesto"],
                        objetivo=demo["pose"], probs=np.stack([p[0] for p in preds]), pose=np.stack([p[1] for p in preds]),
                        vista=demo["vista"])
        historial = self.alm.sql("SELECT id, mejor, media FROM generaciones ORDER BY id DESC LIMIT 200")[::-1]
        self.enviar(
            "estado", fase="pausa" if self.pausa else "entrenando", gen=self.gen, segundos_gen=round(segundos, 1),
            mensaje=f"Generación {self.gen} lista en {segundos:.1f} s",
            manos=[{"mano": m.idx, "rank": ranking[m.idx], "linaje": m.linaje, "genoma": m.genoma,
                    "parametros": m.parametros, "pasos": m.pasos, **m.metricas} for m in self.manos],
            pesos={m.idx: m.vista_pesos() for m in self.manos},
            campeona=self.campeona, campeonas=self.campeonas(), eventos=self.eventos[-60:],
            historial=np.array(historial, float), memoria=int(self.X.shape[0]), datos=self.alm.resumen(),
            salud=self.salud(), amplificador={**asdict(self.amp), "g_ad620": round(self.amp.g_ad620, 1)},
            dispositivo=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    def salud(self):
        bat = psutil.sensors_battery()
        return {"cargador": True if bat is None else bool(bat.power_plugged),
                "bateria": None if bat is None else round(bat.percent),
                "ram_libre_gb": round(psutil.virtual_memory().available / 1e9, 1),
                "gpu_mb": round(torch.cuda.memory_allocated() / 1e6) if torch.cuda.is_available() else 0}

    # ---------- bucle ----------
    def correr(self):
        try:
            self.preparar()
            while not self.detener:
                self.atender()
                s = self.salud()
                motivo = None if s["cargador"] else "el cargador está desconectado (en batería Windows apaga la GPU)"
                motivo = motivo or (f"solo hay {s['ram_libre_gb']} GB de RAM libre" if s["ram_libre_gb"] < 1.2 else None)
                if self.pausa or motivo:
                    self.aviso("pausa", "En pausa" + (f": {motivo}" if motivo else ""))
                    time.sleep(1)
                    continue
                while len(self.en_vuelo) < 2:
                    self.en_vuelo.append(self._pedir_lote())
                self._recoger_lotes()
                self.generacion()
        except Exception:
            self.enviar("error", texto=traceback.format_exc()[-3000:])
            traceback.print_exc()
        finally:
            if self.manos:
                self.aviso("deteniendo", "Guardando la población…")
                self.guardar_poblacion()
            self.pool.shutdown(cancel_futures=True)
            self.enviar("detenido")


def proceso(salida, entrada):
    """Punto de entrada del proceso que lanza la aplicación de escritorio."""
    Motor(salida, entrada).correr()


def main():
    """Sin ventana: python -m gemelo.motor (útil para dejarlo entrenando)."""
    Motor().correr()


if __name__ == "__main__":
    main()
