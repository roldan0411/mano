"""
Agente de servos por refuerzo (REINFORCE) — versión PyTorch/GPU
===============================================================

Misma idea que simulador.AgenteServos (que usa TensorFlow), para que corra
en el kernel del notebook con Keras sobre PyTorch:

  Estado: el gesto (one-hot de 7) · Acción: 6 comandos (5 dedos y muñeca)
  Recompensa: -(error entre la pose lograda por la mano física virtual y la
  pose objetivo) - un pequeño costo de esfuerzo.

Métrica agregada: R² entre la pose lograda y la objetivo (7 gestos × 6 GDL).
La política se guarda en .h5 con el mismo formato que la versión anterior
y sus comandos medios se exportan a poses_aprendidas.h.
"""

import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import h5py
import keras
import numpy as np
import torch
from keras import layers

from . import config, metricas
from .datos_emg import NOMBRES_DOF
import ia_mano as ia
import simulador as sm

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgenteServos:
    def __init__(self, mano, semilla=0, tasa=3e-3, sigma_inicial=0.25):
        config.fijar_semillas(semilla)
        self.mano = mano
        self.rng = np.random.default_rng(semilla)
        ent = keras.Input((ia.NUM_GESTOS,), name="gesto")
        x = layers.Dense(32, activation="tanh", name="oculta")(ent)
        self.politica = keras.Model(ent, layers.Dense(6, name="comandos_crudos")(x), name="politica_servos")
        self.log_sigma = torch.nn.Parameter(torch.full((6,), float(np.log(sigma_inicial)), device=DISPOSITIVO))
        self.opt = torch.optim.Adam([v.value for v in self.politica.trainable_variables] + [self.log_sigma], lr=tasa)
        self.base = np.zeros(ia.NUM_GESTOS)
        self.historial = []
        self.iteracion = 0
        self.objetivos = np.array([sm.POSES_OBJETIVO[g][0] + [sm.POSES_OBJETIVO[g][1]] for g in ia.GESTOS])
        self.eye = torch.eye(ia.NUM_GESTOS, device=DISPOSITIVO)

    def _media(self, onehot):
        z = self.politica(onehot)
        return torch.cat([torch.sigmoid(z[:, :5]), torch.tanh(z[:, 5:])], dim=1)

    def recompensa(self, acciones, idx):
        dedos, muneca = self.mano.respuesta(acciones[:, :5], acciones[:, 5])
        obj = self.objetivos[idx]
        err = np.mean((dedos - obj[:, :5]) ** 2, axis=1) + 2 * (muneca - obj[:, 5]) ** 2
        return -(err + 0.01 * np.mean(np.clip(acciones[:, :5], 0, 1) ** 2, axis=1))

    def iteracion_completa(self, lote=128):
        self.iteracion += 1
        idx = self.rng.integers(0, ia.NUM_GESTOS, lote)
        onehot = self.eye[torch.as_tensor(idx, device=DISPOSITIVO)]
        mu = self._media(onehot)
        sigma = torch.exp(self.log_sigma)
        eps = torch.as_tensor(self.rng.normal(0, 1, (lote, 6)), dtype=torch.float32, device=DISPOSITIVO)
        accion = (mu + sigma * eps).detach()
        a = accion.cpu().numpy()
        r = self.recompensa(np.c_[np.clip(a[:, :5], 0, 1), np.clip(a[:, 5], -1, 1)], idx)
        ventaja = r - self.base[idx]
        ventaja = (ventaja - ventaja.mean()) / (ventaja.std() + 1e-8)
        logp = torch.sum(-0.5 * ((accion - mu) / sigma) ** 2 - self.log_sigma, dim=1)
        perdida = -torch.mean(torch.as_tensor(ventaja, dtype=torch.float32, device=DISPOSITIVO) * logp)
        self.opt.zero_grad()
        perdida.backward()
        self.opt.step()
        with torch.no_grad():
            self.log_sigma.clamp_(min=float(np.log(0.02)))
        for g in range(ia.NUM_GESTOS):
            m = idx == g
            if m.any():
                self.base[g] = 0.9 * self.base[g] + 0.1 * r[m].mean()
        reg = {"iteracion": self.iteracion, "recompensa": float(r.mean()), **self.evaluar(),
               "sigma": float(torch.exp(self.log_sigma.detach()).mean())}
        self.historial.append(reg)
        return reg

    def poses(self):
        with torch.no_grad():
            mu = self._media(self.eye).cpu().numpy()
        return {g: (list(map(float, mu[i, :5])), float(mu[i, 5])) for i, g in enumerate(ia.GESTOS)}

    def logradas(self, poses=None):
        poses = poses or self.poses()
        out = []
        for g in ia.GESTOS:
            d, m = poses[g]
            dd, mm = self.mano.respuesta(np.array(d), m)
            out.append(np.r_[dd[0], mm[0]])
        return np.array(out)

    def evaluar(self, poses=None):
        lograda = self.logradas(poses)
        err = float(np.mean([sm.error_pose(l[:5], l[5], g) for l, g in zip(lograda, ia.GESTOS)]))
        return {"error_medio": err, "r2_pose": metricas.r2(self.objetivos.ravel(), lograda.ravel())["r2"]}

    def r2_por_gdl(self, poses=None):
        return metricas.r2(self.objetivos, self.logradas(poses), NOMBRES_DOF)

    def guardar(self, ruta):
        self.politica.save(ruta)
        with h5py.File(ruta, "a") as f:
            f.attrs["metadatos_mano"] = json.dumps({
                "tipo": "politica de servos (REINFORCE)", "gestos": ia.GESTOS, "salidas": NOMBRES_DOF,
                "activacion_salida": "sigmoid (dedos), tanh (muñeca)",
                "sigma": torch.exp(self.log_sigma.detach()).tolist(), "r2_pose": self.evaluar()["r2_pose"],
                "autor": config.AUTOR}, ensure_ascii=False)
