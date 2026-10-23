"""
Población de manos (Population-Based Training)
==============================================

Cada mano es una red pequeña, del tamaño que cabe en el ESP32, con su propio
"genoma" (tamaño de capas, tasa de aprendizaje, olvido, ruido de aumento...).
Entrada: las 8 características calibradas de las 3 últimas ventanas (24).
Dos salidas: el gesto (7) y la pose continua de la mano (6 grados de libertad).

En cada generación:
  1. cada mano entrena unos pasos con los datos acumulados (GPU);
  2. todas rinden el MISMO examen (personas que ninguna vio);
  3. las 3 peores copian los pesos y el genoma de una de las 3 mejores y mutan;
  4. si la mejor supera a la campeona, pasa a ser la nueva campeona.
"""

import copy
import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
import torch.nn.functional as Fn
from keras import layers

from .. import metricas
import ia_mano as ia

N_ENTRADAS = 24
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPCIONES = {"h1": [16, 24, 32, 48, 64], "h2": [0, 8, 16, 24, 32], "olvido": [0.0, 0.1, 0.2],
            "l2": [0.0, 1e-5, 1e-4], "peso_pose": [0.5, 1.0, 2.0], "ruido": [0.0, 0.03, 0.08]}
NOMBRES_ENTRADAS = [f"{c} t{'' if k == 2 else f'-{2 - k}'}" for k in range(3) for c in ia.NOMBRES_CARAC]


def genoma_aleatorio(rng):
    g = {k: v[int(rng.integers(len(v)))] for k, v in OPCIONES.items()}
    g["lr"] = float(10 ** rng.uniform(-3.4, -2.3))
    return g


def mutar(g, rng):
    """Los pesos se heredan, así que la arquitectura no cambia; sí el entrenamiento."""
    g = dict(g)
    g["lr"] = float(np.clip(g["lr"] * rng.choice([0.7, 0.85, 1.2, 1.4]), 1e-4, 1e-2))
    for k in ("olvido", "l2", "peso_pose", "ruido"):
        if rng.random() < 0.35:
            ops = OPCIONES[k]
            i = ops.index(g[k]) + int(rng.choice([-1, 1]))
            g[k] = ops[int(np.clip(i, 0, len(ops) - 1))]
    return g


def construir(g, X_norm):
    ent = keras.Input((N_ENTRADAS,), name="entrada")
    norm = layers.Normalization(name="normalizacion")
    norm.adapt(X_norm)
    h = norm(ent)
    h = layers.Dense(g["h1"], activation="relu", name="oculta_1")(h)
    if g["olvido"]:
        h = layers.Dropout(g["olvido"], name="olvido_1")(h)
    if g["h2"]:
        h = layers.Dense(g["h2"], activation="relu", name="oculta_2")(h)
    gesto = layers.Dense(ia.NUM_GESTOS, name="gesto")(h)          # logits (softmax en la pérdida y en el ESP32)
    pose = layers.Dense(6, name="pose")(h)
    return keras.Model(ent, [gesto, pose], name="mano")


class Mano:
    def __init__(self, idx, genoma, X_norm, linaje=None):
        self.idx, self.genoma = idx, genoma
        self.linaje = linaje or f"M{idx + 1}"
        self.modelo = construir(genoma, X_norm)
        self._optimizador()
        self.metricas = {"puntaje": 0.0, "acierto": 0.0, "pseudo_r2": 0.0, "r2_pose": 0.0}
        self.historial = []
        self.pasos = 0

    def _optimizador(self):
        params = [v.value for v in self.modelo.trainable_variables]
        self.opt = torch.optim.AdamW(params, lr=self.genoma["lr"], weight_decay=self.genoma["l2"] * 100)

    @property
    def parametros(self):
        return int(sum(np.prod(v.shape) for v in self.modelo.trainable_variables))

    # ---------- entrenar ----------
    def entrenar(self, X, y, P, pasos, lote, gen_torch):
        """X, y, P: tensores en la GPU (memoria de datos acumulados)."""
        n = X.shape[0]
        perdidas = []
        for _ in range(pasos):
            i = torch.randint(0, n, (lote,), device=X.device, generator=gen_torch)
            xb = X[i]
            if self.genoma["ruido"]:
                xb = xb + torch.randn_like(xb) * self.genoma["ruido"]
            g, p = self.modelo(xb, training=True)
            perdida = Fn.cross_entropy(g, y[i]) + self.genoma["peso_pose"] * Fn.mse_loss(p, P[i])
            self.opt.zero_grad(set_to_none=True)
            perdida.backward()
            self.opt.step()
            perdidas.append(float(perdida.detach()))
        self.pasos += pasos
        return float(np.mean(perdidas))

    # ---------- evaluar ----------
    @torch.no_grad()
    def predecir(self, X):
        g, p = self.modelo(torch.as_tensor(X, device=DISPOSITIVO), training=False)
        probs = torch.softmax(g, dim=1).cpu().numpy()
        p = p.cpu().numpy()
        return probs, np.c_[np.clip(p[:, :5], 0, 1), np.clip(p[:, 5], -1, 1)]

    def evaluar(self, ex):
        probs, pose = self.predecir(ex["X"])
        y = ex["gesto"].astype(int)
        acierto = float(np.mean(probs.argmax(1) == y))
        pr2 = metricas.pseudo_r2_mcfadden(y, probs, np.full(ia.NUM_GESTOS, 1 / ia.NUM_GESTOS))
        r2p = metricas.r2(ex["pose"], pose)["r2"]
        puntaje = 0.6 * acierto + 0.2 * max(pr2, 0) + 0.2 * max(r2p, 0)
        self.metricas = {"puntaje": round(puntaje, 4), "acierto": round(acierto, 4), "pseudo_r2": round(pr2, 4),
                         "r2_pose": round(r2p, 4)}
        rec = [float(np.mean(probs.argmax(1)[y == g] == g)) for g in range(ia.NUM_GESTOS)]
        self.metricas["recall"] = [round(r, 3) for r in rec]
        self.historial = (self.historial + [self.metricas["puntaje"]])[-300:]
        return self.metricas

    # ---------- explotar / explorar ----------
    def copiar_de(self, otra, rng):
        self.genoma = mutar(otra.genoma, rng)
        self.modelo = construir(otra.genoma, np.zeros((2, N_ENTRADAS), np.float32))
        self.modelo.set_weights(otra.modelo.get_weights())
        norm = self.modelo.get_layer("normalizacion")
        if hasattr(norm, "finalize_state"):
            norm.finalize_state()                    # Keras 3: recalcula media y escala internas
        self._optimizador()
        self.linaje = f"{otra.linaje}>{self.idx + 1}"[-24:]

    # ---------- vistas y exportación ----------
    def vista_pesos(self):
        capas = []
        for nombre in ("oculta_1", "oculta_2", "gesto", "pose"):
            try:
                W, b = self.modelo.get_layer(nombre).get_weights()
            except ValueError:
                continue
            capas.append({"nombre": nombre, "W": W.astype(np.float32), "b": b.astype(np.float32)})
        norm = self.modelo.get_layer("normalizacion").get_weights()
        return {"media": np.ravel(norm[0]).astype(np.float32),
                "escala": np.sqrt(np.maximum(np.ravel(norm[1]), 1e-12)).astype(np.float32),
                "capas": capas, "entradas": NOMBRES_ENTRADAS, "gestos": [ia.NOMBRES[g] for g in ia.GESTOS],
                "dof": ["pulgar", "índice", "medio", "anular", "meñique", "muñeca"]}

    def modelo_gesto(self):
        """Solo el camino del gesto (Normalization + Dense): lo que cabe en el ESP32."""
        capas = [keras.Input((N_ENTRADAS,)), self.modelo.get_layer("normalizacion"), self.modelo.get_layer("oculta_1")]
        if self.genoma["h2"]:
            capas.append(self.modelo.get_layer("oculta_2"))
        capas.append(self.modelo.get_layer("gesto"))
        return keras.Sequential(capas, name="mano_esp32")

    def guardar(self, ruta):
        self.modelo.save(ruta)
        with open(str(ruta) + ".json", "w", encoding="utf-8") as f:
            json.dump({"genoma": self.genoma, "linaje": self.linaje, "metricas": self.metricas,
                       "historial": self.historial, "pasos": self.pasos}, f)

    @classmethod
    def cargar(cls, idx, ruta):
        with open(str(ruta) + ".json", encoding="utf-8") as f:
            info = json.load(f)
        m = cls.__new__(cls)
        m.idx, m.genoma, m.linaje = idx, info["genoma"], info["linaje"]
        m.modelo = keras.models.load_model(ruta, compile=False)
        m._optimizador()
        m.metricas, m.historial, m.pasos = info["metricas"], info["historial"], info.get("pasos", 0)
        return m
