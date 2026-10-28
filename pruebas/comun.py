"""
Base común de las pruebas técnicas
==================================

Cada prueba guarda un registro en pruebas/resultados/<ID>.json (objetivo,
requerimientos que verifica, procedimiento, criterio, resultado, si cumple
y sus figuras) y sus gráficas en pruebas/resultados/figuras/. Con esos
registros se arman el documento técnico y el documento de pruebas.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_PROYECTO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTADOS = RAIZ_PROYECTO / "pruebas" / "resultados"
FIGURAS = RESULTADOS / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)
MOTOR = Path(os.environ.get("GEMELO_DATOS", Path.home() / "gemelo_mano")) / "motor"

REQUERIMIENTOS = {
    "RF-01": "Simular la señal EMG de 10 músculos del antebrazo, con diferencias entre personas, fatiga, ruido, interferencia de 60 Hz y artefactos de movimiento.",
    "RF-02": "Simular 10 electrodos (F1–F5 sobre los flexores, E1–E5 sobre los extensores) con acople músculo–electrodo por geometría y desplazamiento en cada sesión.",
    "RF-03": "Simular la cadena de adquisición: AD620 (G = 1 + 49,4 kΩ/RG), pasa-banda 16–482 Hz ×10, ADC de 12 bits a 1 kHz, desfase de electrodos y saturación.",
    "RF-04": "Extraer MAV, RMS, ZC y WL de cada electrodo en ventanas de 200 ms cada 100 ms (40 características).",
    "RF-05": "Calibrar cada sesión (cada gesto una vez) y expresar las características relativas a la calibración; contexto de 3 ventanas (120 entradas).",
    "RF-06": "Reconocer 7 gestos (reposo, abierta, puño, pinza, señalar, flexión y extensión de muñeca) y estimar la pose de 6 grados de libertad.",
    "RF-07": "Entrenar sin parar una población de 10 redes (Population-Based Training) y guardar cada campeona (modelo, código C y fotos).",
    "RF-08": "Neuroevolución (NEAT): 150 redes que aprenden desde cero a seguir la señal de una persona; campeones exportados a C.",
    "RF-09": "Mano bebé: aprendizaje autónomo con 10 músculos, sin metas predeterminadas, repertorio sin límite y espejo de manos reales de internet.",
    "RF-10": "Aplicación de escritorio con visualización en vivo: manos 3D, redes, electrodos, señales por etapa y curvas.",
    "RF-11": "Persistencia: guardar todo lo aprendido y reanudar al reabrir.",
    "RF-12": "Exportar los modelos al ESP32 y un firmware que reproduzca el procesamiento (10 electrodos, calibración, contexto, inferencia y servos).",
    "RF-13": "Reportar siempre el R²: pseudo-R² de McFadden para los gestos y R² para la pose.",
    "RNF-01": "Acierto ≥ 90 % y recall ≥ 85 % en cada gesto con personas que el modelo no vio.",
    "RNF-02": "Robustez: degradación acotada ante desplazamiento de electrodos, ruido y fatiga.",
    "RNF-03": "Tolerancia a la falla (desprendimiento) de un electrodo.",
    "RNF-04": "Latencia de reconocimiento (del inicio del gesto a la mano) menor a 300 ms en la mediana.",
    "RNF-05": "Equivalencia del firmware: mismo gesto que Python en el 100 % de las ventanas.",
    "RNF-06": "Recursos del ESP32: inferencia menor a 10 ms por ventana y el modelo cabe en memoria.",
    "RNF-07": "Estabilidad: sin fugas de memoria ni bloqueos en operación prolongada.",
    "RNF-08": "Operación en portátil: se adapta a batería y poca RAM sin perder datos.",
    "RNF-09": "Reproducibilidad (semillas) y trazabilidad (todo queda registrado).",
    "RNF-10": "Aplicación de escritorio nativa para Windows (no web).",
}

# estilo de las gráficas de los documentos (fondo blanco, imprimible)
AZULES = ["#9cc3f5", "#6fa6ee", "#3f86e0", "#2867c2", "#1a4b96"]
VERDES = ["#8fdcc2", "#5cc7a4", "#2fa982", "#21876a", "#166650"]
COLORES_CANAL = AZULES + VERDES
ACENTO, GRIS, ROJO, NARANJA = "#2867c2", "#7d8a94", "#d64541", "#e08a2e"
GESTOS_ES = ["Reposo", "Abierta", "Puño", "Pinza", "Señalar", "Flexión", "Extensión"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9.5, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "font.family": "DejaVu Sans"})


def figura(id_prueba, nombre, fig, ancho_cm=16):
    ruta = FIGURAS / f"{id_prueba}_{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"archivo": str(ruta.relative_to(RESULTADOS)), "ancho_cm": ancho_cm}


def _limpio(v):
    if isinstance(v, dict):
        return {str(k): _limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_limpio(x) for x in v]
    if isinstance(v, np.ndarray):
        return _limpio(v.tolist())
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class Prueba:
    """with Prueba("A1", ...) as p:  p.resultado[...] = ...;  p.cumple = ...;  p.figuras.append(figura(...))"""

    def __init__(self, id_prueba, nombre, grupo, objetivo, requerimientos, procedimiento, criterio):
        self.registro = {"id": id_prueba, "nombre": nombre, "grupo": grupo, "objetivo": objetivo,
                         "requerimientos": requerimientos, "procedimiento": procedimiento, "criterio": criterio}
        self.resultado, self.figuras, self.cumple, self.conclusion = {}, [], None, ""

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {self.registro['id']} · {self.registro['nombre']}…", flush=True)
        return self

    def __exit__(self, tipo, valor, tb):
        if tipo is not None:
            self.resultado["error"] = f"{tipo.__name__}: {valor}"
            self.cumple = False
        self.registro.update(resultado=self.resultado, figuras=self.figuras, cumple=self.cumple, conclusion=self.conclusion,
                             fecha=time.strftime("%Y-%m-%d %H:%M"), duracion_s=round(time.time() - self.t0, 1))
        (RESULTADOS / f"{self.registro['id']}.json").write_text(json.dumps(_limpio(self.registro), ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
        estado = {True: "CUMPLE", False: "NO CUMPLE", None: "INFORMATIVA"}[self.cumple]
        print(f"    → {estado} ({time.time() - self.t0:.0f} s) {self.conclusion}", flush=True)
        return False


# ------------------------------------------------------------------
# Datos del motor (lo que el gemelo generó y guardó)
# ------------------------------------------------------------------
def leer_lotes(campos=("X", "F", "gesto", "pose", "persona", "sesion", "ronda", "ventana_idx"), max_lotes=None, semilla=0):
    import h5py
    archivos = sorted((MOTOR / "lotes").glob("*.h5"))
    if max_lotes and len(archivos) > max_lotes:
        archivos = [archivos[i] for i in np.sort(np.random.default_rng(semilla).choice(len(archivos), max_lotes, replace=False))]
    partes = {c: [] for c in campos}
    cal = {}
    for a in archivos:
        with h5py.File(a, "r") as f:
            for c in campos:
                partes[c].append(f[c][()])
            for s, v in zip(f["cal_sesion"][()], f["cal_valor"][()]):
                cal[int(s)] = v
    d = {c: np.concatenate(v) for c, v in partes.items()}
    d["cal"] = cal
    return d


def leer_examen():
    import h5py
    with h5py.File(MOTOR / "examen.h5", "r") as f:
        return {k: f[k][()] for k in f}


def columnas_canales(canales, por_ventana=40, ventanas=3):
    """Índices de X (120) que pertenecen a esos electrodos (0 = F1 … 9 = E5)."""
    return np.array([j * por_ventana + c * 4 + k for j in range(ventanas) for c in canales for k in range(4)])


def campeona():
    """La campeona más nueva del motor: (modelo Keras, ficha)."""
    import keras
    carpetas = sorted((MOTOR / "campeonas").glob("c*"))
    c = carpetas[-1]
    ficha = json.loads((c / "ficha.json").read_text(encoding="utf-8"))
    return keras.models.load_model(c / "modelo.keras", compile=False), ficha, c


def predecir(modelo, X, lote=8192):
    import torch
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    probs, poses = [], []
    with torch.no_grad():
        for i in range(0, len(X), lote):
            g, p = modelo(torch.as_tensor(np.asarray(X[i:i + lote], np.float32), device=dispositivo), training=False)
            probs.append(torch.softmax(g, 1).cpu().numpy())
            poses.append(p.cpu().numpy())
    return np.concatenate(probs), np.concatenate(poses)


def entrenar_mlp(Xtr, ytr, Xte, Ptr=None, ocultas=(64, 32), epocas=8, lote=1024, lr=2e-3, semilla=0, aumento=None):
    """MLP (PyTorch) de referencia para comparar configuraciones. Devuelve las probabilidades en Xte
    (y la pose predicha si se da Ptr). aumento(xb, ib) puede modificar cada lote de entrenamiento."""
    import torch
    from torch import nn
    torch.manual_seed(semilla)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(ytr.astype(np.int64), device=dev)
    Pt = None if Ptr is None else torch.as_tensor(Ptr, dtype=torch.float32, device=dev)
    capas, n = [], Xtr.shape[1]
    for h in ocultas:
        capas += [nn.Linear(n, h), nn.ReLU()]
        n = h
    cuerpo = nn.Sequential(*capas).to(dev)
    cab_g, cab_p = nn.Linear(n, 7).to(dev), nn.Linear(n, 6).to(dev)
    params = list(cuerpo.parameters()) + list(cab_g.parameters()) + list(cab_p.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    g = torch.Generator(device=dev).manual_seed(semilla)
    for _ in range(epocas):
        orden = torch.randperm(len(Xt), device=dev, generator=g)
        for i in range(0, len(Xt), lote):
            ib = orden[i:i + lote]
            xb = Xt[ib] if aumento is None else (aumento(Xt[ib] * torch.as_tensor(sd, device=dev) + torch.as_tensor(mu, device=dev), ib)
                                                 - torch.as_tensor(mu, device=dev)) / torch.as_tensor(sd, device=dev)
            h = cuerpo(xb.float())
            perdida = nn.functional.cross_entropy(cab_g(h), yt[ib])
            if Pt is not None:
                perdida = perdida + nn.functional.mse_loss(cab_p(h), Pt[ib])
            opt.zero_grad()
            perdida.backward()
            opt.step()
    with torch.no_grad():
        h = cuerpo(torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=dev))
        probs = torch.softmax(cab_g(h), 1).cpu().numpy()
        pose = cab_p(h).cpu().numpy()
    return (probs, pose) if Ptr is not None else probs


def separar_personas(personas, fraccion_prueba=0.2, semilla=0):
    u = np.unique(personas)
    rng = np.random.default_rng(semilla)
    prueba = rng.choice(u, max(1, int(len(u) * fraccion_prueba)), replace=False)
    return ~np.isin(personas, prueba), np.isin(personas, prueba)


def intervalo_bootstrap(valores_por_grupo, n=2000, semilla=0):
    """IC 95 % de la media remuestreando grupos (personas)."""
    v = np.asarray(valores_por_grupo, float)
    rng = np.random.default_rng(semilla)
    medias = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))
