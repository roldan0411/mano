"""
Red de formas de la mano (D2, y D3 cuando haya webcam)
======================================================

Autoencoder de los 21 puntos con un espacio latente de 3 dimensiones (se
puede ver tal cual en 3D) y una cabeza que recupera los 6 grados de
libertad. Aprende "qué formas puede tomar la mano" y sirve para:

  - ver el espacio de formas: cada gesto ocupa una región, las
    transiciones son caminos entre regiones;
  - pasar poses de la webcam (MediaPipe) a comandos de 6 GDL;
  - medir con R² la reconstrucción y la recuperación de la pose.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from keras import layers

from . import config, metricas
from .datos_emg import NOMBRES_DOF


def crear_red_formas(dim_latente=3):
    ent = keras.Input((21, 3), name="landmarks")
    h = layers.Flatten(name="aplanar")(ent)
    h = layers.Dense(128, activation="relu", name="cod_1")(h)
    h = layers.Dense(64, activation="relu", name="cod_2")(h)
    z = layers.Dense(dim_latente, name="latente")(h)
    d = layers.Dense(64, activation="relu", name="dec_1")(z)
    d = layers.Dense(128, activation="relu", name="dec_2")(d)
    rec = layers.Reshape((21, 3), name="reconstruccion")(layers.Dense(63, name="dec_salida")(d))
    hp = layers.Dense(32, activation="relu", name="dof_oculta")(h)
    dof = layers.Concatenate(name="dof")([layers.Dense(5, activation="sigmoid", name="dedos")(hp),
                                          layers.Dense(1, activation="tanh", name="muneca")(hp)])
    m = keras.Model(ent, {"reconstruccion": rec, "dof": dof}, name="red_formas_mano")
    m.compile(optimizer=keras.optimizers.Adam(2e-3), loss={"reconstruccion": "mse", "dof": "mse"},
              loss_weights={"reconstruccion": 1.0, "dof": 2.0})
    return m


def codificador_de(red):
    return keras.Model(red.input, red.get_layer("latente").output, name="codificador_formas")


def evaluar(red, X, dof):
    s = red.predict(X, batch_size=4096, verbose=0)
    return {"reconstruccion": metricas.r2(X.reshape(len(X), -1), s["reconstruccion"].reshape(len(X), -1))["r2"],
            "dof": metricas.r2(dof, s["dof"], NOMBRES_DOF)}


class CallbackFormas(keras.callbacks.Callback):
    """En cada época: R² de validación y la nube del espacio latente, en vivo."""

    def __init__(self, panel, X_val, dof_val, y_val, n_nube=3000):
        super().__init__()
        self.panel, self.X, self.dof, self.y = panel, X_val, dof_val, y_val
        self.sel = np.random.default_rng(0).choice(len(X_val), min(n_nube, len(X_val)), replace=False)
        self.historial = []

    def on_epoch_end(self, epoca, logs=None):
        ev = evaluar(self.model, self.X, self.dof)
        self.historial.append({"perdida": float(logs["loss"]), "perdida_val": float(logs.get("val_loss", np.nan)),
                               "r2_reconstruccion": ev["reconstruccion"], "r2_dof": ev["dof"]["r2"]})
        if self.panel is not None:
            z = codificador_de(self.model).predict(self.X[self.sel], verbose=0)
            self.panel.actualizar(self.historial, [("r2_reconstruccion", "R² reconstrucción"),
                                                   ("r2_dof", "R² 6 GDL")], z, self.y[self.sel])
