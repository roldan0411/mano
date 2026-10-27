"""
============================================================
 PRÓTESIS MIOELÉCTRICA CON RECONOCIMIENTO DE GESTOS POR IA
 Núcleo de IA compartido (gemelo digital + modelo físico)
------------------------------------------------------------
 Roldán Rodríguez — Ing. CCIA
============================================================

Lo usan gemelo_digital.py, simulador.py y entrenar_modelo.py.
Aquí vive TODO lo que tiene que ser idéntico en el gemelo, en
el entrenamiento con datos reales y en el ESP32:

  - Los 7 gestos y sus poses.
  - La extracción de características (MAV, RMS, ZC, WL).
  - La red neuronal en Keras, guardada en formato .h5.
  - La exportación de la red y de las poses a C para el ESP32.
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

# ------------------------------------------------------------------
# Gestos y parámetros de procesamiento (iguales en el ESP32)
# ------------------------------------------------------------------
GESTOS = ["reposo", "abierta", "puno", "pinza", "senalar", "flexion", "extension"]
NOMBRES = {"reposo": "Reposo", "abierta": "Abierta", "puno": "Puño", "pinza": "Pinza",
           "senalar": "Señalar", "flexion": "Flexión muñeca", "extension": "Extensión muñeca"}
NUM_GESTOS = len(GESTOS)
FS = 1000          # muestras por segundo
VENTANA = 200      # 200 ms
PASO = 100         # una predicción cada 100 ms
UMBRAL_ZC = 20     # cuentas del ADC
# 10 electrodos, cada uno un canal contra una referencia común (en el codo):
# F1-F5 sobre los flexores (cara palmar) y E1-E5 sobre los extensores (cara dorsal)
CANALES = [f"F{i}" for i in range(1, 6)] + [f"E{i}" for i in range(1, 6)]
NUM_CANALES = len(CANALES)
CARAC_CANAL = ("MAV", "RMS", "ZC", "WL")
NUM_CARACTERISTICAS = len(CARAC_CANAL) * NUM_CANALES      # 40
NOMBRES_CARAC = [f"{c}_{n}" for c in CANALES for n in CARAC_CANAL]
UMBRAL_CONFIANZA = 0.60
DEDOS = ["pulgar", "indice", "medio", "anular", "menique"]

# Poses de comando de los servos (las mismas del ESP32).
# dedos: 0 = estirado, 1 = cerrado.  muñeca: -1 = flexión, 0 = recta, +1 = extensión
POSES_BASE = {
    "reposo":    ([0.2, 0.2, 0.2, 0.2, 0.2], 0.0),
    "abierta":   ([0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
    "puno":      ([1.0, 1.0, 1.0, 1.0, 1.0], 0.0),
    "pinza":     ([0.8, 0.8, 0.0, 0.0, 0.0], 0.0),
    "senalar":   ([1.0, 0.0, 1.0, 1.0, 1.0], 0.0),
    "flexion":   ([0.2, 0.2, 0.2, 0.2, 0.2], -1.0),
    "extension": ([0.2, 0.2, 0.2, 0.2, 0.2], 1.0),
}


# ------------------------------------------------------------------
# Características (misma fórmula que en el ESP32)
# ------------------------------------------------------------------
def caracteristicas_canal(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    mav = np.mean(np.abs(x))
    rms = np.sqrt(np.mean(x * x))
    d = np.diff(x)
    zc = np.sum((x[:-1] * x[1:] < 0) & (np.abs(d) >= UMBRAL_ZC))
    wl = np.sum(np.abs(d))
    return [mav, rms, float(zc), wl]


def caracteristicas_ventana(*canales):
    return [v for c in canales for v in caracteristicas_canal(c)]


def caracteristicas_lote(ventanas):
    """ventanas: arreglo (n, VENTANA, canales) en cuentas del ADC -> (n, 4 × canales). Vectorizado."""
    v = np.asarray(ventanas, dtype=np.float64)
    v = v - v.mean(axis=1, keepdims=True)
    mav = np.mean(np.abs(v), axis=1)
    rms = np.sqrt(np.mean(v * v, axis=1))
    d = np.diff(v, axis=1)
    zc = np.sum((v[:, :-1] * v[:, 1:] < 0) & (np.abs(d) >= UMBRAL_ZC), axis=1).astype(np.float64)
    wl = np.sum(np.abs(d), axis=1)
    # orden: canal 1 MAV, RMS, ZC, WL, canal 2 MAV, RMS, ZC, WL, …
    return np.stack([mav, rms, zc, wl], axis=2).reshape(len(v), -1)


def ventanear(senal, descarte=0):
    """senal (n, canales) -> ventanas (k, VENTANA, canales) con paso PASO."""
    s = np.asarray(senal)[descarte:]
    idx = range(0, len(s) - VENTANA + 1, PASO)
    if len(s) < VENTANA:
        return np.empty((0, VENTANA, s.shape[1] if s.ndim == 2 else NUM_CANALES))
    return np.stack([s[i:i + VENTANA] for i in idx])


# ------------------------------------------------------------------
# Red neuronal en Keras (.h5)
# ------------------------------------------------------------------
class ErrorModelo(Exception):
    pass


def _keras():
    import keras
    return keras


def crear_red(capas=(32, 16), X_adaptar=None, tasa=1e-3):
    """Perceptrón multicapa 8-32-16-7 con la estandarización DENTRO del modelo,
    así el .h5 queda completo: no hay que guardar medias aparte."""
    keras = _keras()
    entradas = keras.Input(shape=(NUM_CARACTERISTICAS,), name="caracteristicas")
    norm = keras.layers.Normalization(name="normalizacion")
    x = norm(entradas)
    for i, n in enumerate(capas):
        x = keras.layers.Dense(n, activation="relu", name=f"oculta_{i + 1}",
                               kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    salida = keras.layers.Dense(NUM_GESTOS, activation="softmax", name="salida")(x)
    modelo = keras.Model(entradas, salida, name="red_mano")
    if X_adaptar is not None:
        norm.adapt(np.asarray(X_adaptar, dtype=np.float32))
    compilar(modelo, tasa)
    return modelo


def compilar(modelo, tasa=1e-3):
    keras = _keras()
    modelo.compile(optimizer=keras.optimizers.Adam(tasa),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def capa_normalizacion(modelo):
    for capa in modelo.layers:
        if capa.__class__.__name__ == "Normalization":
            return capa
    return None


def readaptar_normalizacion(modelo, X):
    """Recalcula media y varianza con datos nuevos (por ejemplo, los reales del
    brazo). Alinea las escalas del simulador y del circuito físico sin tocar
    los pesos aprendidos."""
    norm = capa_normalizacion(modelo)
    if norm is None:
        return False
    X = np.asarray(X, dtype=np.float32)
    media = X.mean(axis=0)
    var = X.var(axis=0)
    shapes = [tuple(w.shape) for w in norm.get_weights()]
    nuevos = []
    for s in shapes:
        if s == media.shape or s == (1,) + media.shape:
            nuevos.append(media.reshape(s) if not nuevos else var.reshape(s))
        else:  # contador u otra variable interna
            nuevos.append(np.array(len(X), dtype=np.float32).reshape(s) if s != () else np.array(len(X)))
    norm.set_weights(nuevos)
    if hasattr(norm, "finalize_state"):
        norm.finalize_state()  # Keras 3 recalcula la media y escala internas
    return True


def cargar_red(ruta):
    """Carga un .h5 y verifica que sirva para la mano (8 entradas, 7 salidas)."""
    keras = _keras()
    if not os.path.exists(ruta):
        raise ErrorModelo(f"No existe el archivo {ruta}")
    try:
        modelo = keras.models.load_model(ruta, compile=False)
    except Exception as e:
        raise ErrorModelo(f"No pude leer el .h5: {e}")
    ent = modelo.input_shape
    sal = modelo.output_shape
    if ent[-1] != NUM_CARACTERISTICAS:
        raise ErrorModelo(f"La red recibe {ent[-1]} entradas; la mano usa {NUM_CARACTERISTICAS} "
                          f"características ({', '.join(NOMBRES_CARAC)}).")
    if sal[-1] != NUM_GESTOS:
        raise ErrorModelo(f"La red tiene {sal[-1]} salidas; la mano tiene {NUM_GESTOS} gestos "
                          f"({', '.join(GESTOS)}).")
    compilar(modelo)
    return modelo


def leer_metadatos(ruta):
    try:
        import h5py
        with h5py.File(ruta, "r") as f:
            m = f.attrs.get("metadatos_mano")
            return json.loads(m) if m is not None else {}
    except Exception:
        return {}


def guardar_red(modelo, ruta, metadatos=None):
    modelo.save(ruta)
    import h5py
    datos = {"gestos": GESTOS, "caracteristicas": NOMBRES_CARAC, "ventana_ms": VENTANA,
             "paso_ms": PASO, "autor": "Roldán Rodríguez — Ing. CCIA"}
    datos.update(metadatos or {})
    with h5py.File(ruta, "a") as f:
        f.attrs["metadatos_mano"] = json.dumps(datos, ensure_ascii=False)


def contar_parametros(modelo):
    return int(sum(np.prod(w.shape) for w in modelo.trainable_weights))


# ------------------------------------------------------------------
# Copia NumPy de la red: misma cuenta que hace el ESP32
# ------------------------------------------------------------------
class RedNumpy:
    """Copia congelada de la red para inferencia rápida y sin hilos de TF.
    Hace exactamente las operaciones del código C del ESP32."""

    def __init__(self, modelo):
        self.media = np.zeros(NUM_CARACTERISTICAS)
        self.escala = np.ones(NUM_CARACTERISTICAS)
        self.capas = []  # (W, b, activacion)
        for capa in modelo.layers:
            tipo = capa.__class__.__name__
            if tipo == "InputLayer" or tipo == "Dropout":
                continue
            if tipo == "Normalization":
                pesos = capa.get_weights()  # [media, varianza, contador]
                media = np.ravel(np.array(pesos[0], dtype=np.float64))
                var = np.ravel(np.array(pesos[1], dtype=np.float64))
                self.media = media
                self.escala = np.maximum(np.sqrt(var), 1e-7)
            elif tipo == "Dense":
                W, b = capa.get_weights()
                act = capa.get_config().get("activation", "linear")
                if act not in ("relu", "linear", "softmax"):
                    raise ErrorModelo(f"Activación '{act}' no soportada en el ESP32 "
                                      "(usa relu en ocultas y softmax en la salida).")
                self.capas.append((W.astype(np.float64), b.astype(np.float64), act))
            else:
                raise ErrorModelo(f"Capa '{tipo}' no soportada por el ESP32. "
                                  "Usa solo Normalization y Dense.")
        if not self.capas:
            raise ErrorModelo("La red no tiene capas densas.")

    def probabilidades(self, X):
        a = (np.atleast_2d(X) - self.media) / self.escala
        for W, b, act in self.capas:
            a = a @ W + b
            if act == "relu":
                a = np.maximum(a, 0)
        a = a - a.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------
# Exportación a C para el ESP32
# ------------------------------------------------------------------
def _arr(v):
    return ", ".join(f"{float(x):.8e}f" for x in np.ravel(v))


def exportar_red_c(modelo, ruta):
    red = RedNumpy(modelo)
    tam = [red.capas[0][0].shape[0]] + [W.shape[1] for W, _, _ in red.capas]   # 8 o 24 (con calibración y contexto)
    L = ["/*",
         f" * modelo_red.h — Red neuronal {'-'.join(map(str, tam))} exportada desde Keras (.h5)",
         " * Generado automáticamente. NO editar a mano.",
         " * Roldán Rodríguez — Ing. CCIA",
         " */",
         "#pragma once",
         "#include <math.h>",
         "",
         f"#define VENTANA {VENTANA}",
         f"#define PASO {PASO}",
         f"#define UMBRAL_ZC {UMBRAL_ZC}",
         f"#define NUM_CARACTERISTICAS {tam[0]}",
         f"#define NUM_CLASES {NUM_GESTOS}",
         "",
         "static const char* const NOMBRES_GESTOS[NUM_CLASES] = {",
         "  " + ", ".join(f'"{g}"' for g in GESTOS),
         "};",
         "",
         f"static const float RED_MEDIA[{tam[0]}] = {{{_arr(red.media)}}};",
         f"static const float RED_ESCALA[{tam[0]}] = {{{_arr(red.escala)}}};",
         ""]
    for k, (W, b, _) in enumerate(red.capas):
        L.append(f"static const float RED_W{k}[{W.shape[0]}][{W.shape[1]}] = {{")
        L.append(",\n".join("  {" + _arr(f) + "}" for f in W))
        L.append("};")
        L.append(f"static const float RED_B{k}[{W.shape[1]}] = {{{_arr(b)}}};")
        L.append("")
    L += ["static void red_capa(const float *x, int n_in, const float *W, const float *b,",
          "                     int n_out, float *salida, int relu) {",
          "  for (int j = 0; j < n_out; j++) {",
          "    float z = b[j];",
          "    for (int i = 0; i < n_in; i++) z += x[i] * W[i * n_out + j];",
          "    salida[j] = (relu && z < 0.0f) ? 0.0f : z;",
          "  }",
          "}",
          "",
          "/* Devuelve el gesto y su confianza (probabilidad softmax, 0 a 1) */",
          "static int predecir_gesto(const float *x, float *confianza) {",
          f"  float a0[{tam[0]}];",
          f"  for (int i = 0; i < {tam[0]}; i++) a0[i] = (x[i] - RED_MEDIA[i]) / RED_ESCALA[i];"]
    n = len(red.capas)
    for k, (W, b, act) in enumerate(red.capas):
        L.append(f"  float a{k + 1}[{tam[k + 1]}];")
        L.append(f"  red_capa(a{k}, {tam[k]}, &RED_W{k}[0][0], RED_B{k}, {tam[k + 1]}, "
                 f"a{k + 1}, {1 if act == 'relu' else 0});")
    s = f"a{n}"
    L += ["  int mejor = 0;",
          f"  for (int k = 1; k < NUM_CLASES; k++) if ({s}[k] > {s}[mejor]) mejor = k;",
          "  float suma = 0.0f;",
          f"  for (int k = 0; k < NUM_CLASES; k++) suma += expf({s}[k] - {s}[mejor]);",
          "  if (confianza) *confianza = 1.0f / suma;",
          "  return mejor;",
          "}"]
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def exportar_poses_c(poses, ruta, origen="aprendizaje por refuerzo en el gemelo digital"):
    """poses: dict gesto -> (lista de 5 dedos, muñeca). Mismo formato que el ESP32."""
    L = ["/*",
         " * poses_aprendidas.h — Comandos de servos por gesto",
         f" * Origen: {origen}.",
         " * Si este archivo está junto a protesis_tiempo_real.ino, el ESP32 lo usa",
         " * en lugar de la tabla de poses escrita a mano.",
         " * Roldán Rodríguez — Ing. CCIA",
         " */",
         "#pragma once",
         "#define POSES_APRENDIDAS 1",
         "",
         "// dedos: 0 = estirado, 1 = cerrado.  muñeca: -1 = flexión, 0 = recta, +1 = extensión",
         "//                pulgar  indice  medio   anular  menique   muneca"]
    filas = []
    for g in GESTOS:
        d, m = poses[g]
        dd = ", ".join(f"{float(np.clip(v, 0, 1)):.3f}" for v in d)
        filas.append(f'  {{"{g}",{" " * (10 - len(g))}{{{dd}}}, {float(np.clip(m, -1, 1)):+.3f}}}')
    L.append("const Pose POSES[] = {")
    L.append(",\n".join(filas))
    L.append("};")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
