"""
Red estudiante para el ESP32 (destilación del profesor contrastivo)
===================================================================

El profesor (codificador CNN + cabezas) no cabe en el firmware actual. El
estudiante sí: es la misma familia que ia_mano.crear_red (Normalization +
Dense con relu/softmax), sobre las MISMAS 8 características que calcula
protesis_tiempo_real.ino, así que sale por ia_mano.exportar_red_c sin tocar
el firmware.

Aprende de dos cosas a la vez:
  objetivo = α · gesto real (one-hot) + (1 - α) · probabilidades del profesor
Las probabilidades del profesor llevan "conocimiento oscuro": cuánto se
parece un puño a una pinza, algo que la etiqueta dura no dice.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

from . import config  # noqa: F401
import ia_mano as ia


def objetivos_suaves(y, probs_profesor, alfa=0.5, temperatura=2.0):
    p = np.clip(probs_profesor, 1e-8, 1) ** (1 / temperatura)
    p = p / p.sum(1, keepdims=True)
    return (alfa * np.eye(ia.NUM_GESTOS)[y] + (1 - alfa) * p).astype(np.float32)


def entrenar_estudiante(X_tr, y_tr, probs_tr, X_va, y_va, capas=(64, 32, 16), epocas=150, alfa=0.5,
                        temperatura=2.0, verbose=0, callbacks=()):
    m = ia.crear_red(capas=capas, X_adaptar=X_tr)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    parar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    h = m.fit(X_tr.astype(np.float32), objetivos_suaves(y_tr, probs_tr, alfa, temperatura), epochs=epocas,
              batch_size=256, validation_data=(X_va.astype(np.float32), np.eye(ia.NUM_GESTOS)[y_va].astype(np.float32)),
              callbacks=[parar, *callbacks], verbose=verbose)
    return m, h.history


def entrenar_red_actual(X_tr, y_tr, X_va, y_va, epocas=150, verbose=0, callbacks=()):
    """La red 8-32-16-7 de siempre, entrenada con los mismos datos (línea base justa)."""
    m = ia.crear_red(X_adaptar=X_tr)
    parar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    h = m.fit(X_tr.astype(np.float32), y_tr, epochs=epocas, batch_size=256,
              validation_data=(X_va.astype(np.float32), y_va), callbacks=[parar, *callbacks], verbose=verbose)
    return m, h.history


class CadaEpoca(keras.callbacks.Callback):
    """Llama fn(modelo, época) cada `cada` épocas (por ejemplo, para publicar un intento en el visor)."""

    def __init__(self, fn, cada=1):
        super().__init__()
        self.fn, self.cada = fn, cada

    def on_epoch_end(self, epoca, logs=None):
        if epoca % self.cada == 0:
            self.fn(self.model, epoca + 1)


def verificar_equivalencia_esp32(modelo, X):
    """La copia NumPy hace exactamente las cuentas del código C del ESP32.
    Si coincide con Keras, el .h exportado predice lo mismo que el .h5."""
    p_keras = keras.ops.convert_to_numpy(modelo(X[:2000].astype(np.float32), training=False))
    p_c = ia.RedNumpy(modelo).probabilidades(X[:2000])
    return float(np.max(np.abs(p_keras - p_c))), float(np.mean(p_keras.argmax(1) == p_c.argmax(1)))
