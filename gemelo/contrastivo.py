"""
Aprendizaje contrastivo del EMG (profesor) y cabezas multitarea
================================================================

Codificador 1D-CNN sobre la señal cruda (200 ms × 10 electrodos) que aprende un
espacio de 64 dimensiones donde "lo mismo" queda junto y "lo distinto" lejos.

Dos fases con UNA sola pérdida (SupCon de Khosla et al. 2020):
  1. SimCLR (no supervisado): el único positivo de cada ventana es su otra
     vista (misma contracción, otro sensor). La red nunca ve el gesto.
  2. SupCon (supervisado): además, todas las ventanas del mismo gesto son
     positivas entre sí.

Las ventanas sin etiqueta (gesto = -1, por ejemplo datos reales de UCI sin
marcar) siempre se tratan como en SimCLR.

Entrenamiento con PyTorch en la GPU (Keras 3, backend "torch"): los modelos
son Keras, así se guardan en .h5 / .keras como el resto del proyecto.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import time

import keras
import numpy as np
import torch
import torch.nn.functional as F
from keras import layers, ops
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from . import config, metricas
from .datos_emg import NOMBRES_DOF, POSES_6, indices_contexto
import ia_mano as ia

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Capas propias (registradas: el .h5 se puede volver a abrir)
# ------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="gemelo")
class PreprocesarEMG(layers.Layer):
    """Cuentas del ADC -> quita la media de la ventana (como las características) y escala."""

    def __init__(self, escala=100.0, **kw):
        super().__init__(**kw)
        self.escala = escala

    def call(self, x):
        x = ops.cast(x, "float32")
        return (x - ops.mean(x, axis=1, keepdims=True)) / self.escala

    def get_config(self):
        return {**super().get_config(), "escala": self.escala}


@keras.saving.register_keras_serializable(package="gemelo")
class LogEnergia(layers.Layer):
    """log de la energía de cada canal: le da a la red la amplitud de forma directa."""

    def call(self, x):
        return ops.log(ops.mean(ops.square(x), axis=1) + 1e-3)


@keras.saving.register_keras_serializable(package="gemelo")
class NormaL2(layers.Layer):
    def call(self, x):
        return x / ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + 1e-12)


def crear_codificador(dim=64, canales=ia.NUM_CANALES):
    ent = keras.Input((ia.VENTANA, canales), name="emg")
    x = PreprocesarEMG(name="preprocesar")(ent)
    energia = LogEnergia(name="log_energia")(x)
    h = x
    for i, (filtros, k, paso) in enumerate([(32, 7, 1), (64, 5, 2), (64, 5, 2), (128, 3, 2)]):
        h = layers.Conv1D(filtros, k, strides=paso, padding="same", use_bias=False, name=f"conv{i + 1}")(h)
        h = layers.BatchNormalization(name=f"bn{i + 1}")(h)
        h = layers.Activation("relu", name=f"relu{i + 1}")(h)
    h = layers.Concatenate(name="resumen")([layers.GlobalAveragePooling1D(name="promedio")(h),
                                            layers.GlobalMaxPooling1D(name="maximo")(h), energia])
    h = layers.Dense(128, activation="relu", name="denso")(h)
    z = layers.Dense(dim, name="embedding")(h)
    return keras.Model(ent, z, name="codificador_contrastivo")


def crear_proyector(dim=64, salida=32):
    ent = keras.Input((dim,), name="embedding")
    h = layers.Dense(128, activation="relu", name="proy_1")(ent)
    h = layers.Dense(salida, name="proy_2")(h)
    return keras.Model(ent, NormaL2(name="norma")(h), name="proyector")


# ------------------------------------------------------------------
# Aumentos en la GPU (además del par físico del gemelo)
# ------------------------------------------------------------------
def aumentar(x):
    """x: (B, 200, 10) float en cuentas. Ganancia por canal, cruce con los electrodos vecinos,
    ruido, corrimiento temporal y, a veces, un electrodo flojo."""
    B, T, C = x.shape
    dev = x.device
    x = x - x.mean(1, keepdim=True)
    x = x * torch.exp(torch.randn(B, 1, C, device=dev) * 0.15)
    x = x + torch.rand(B, 1, 1, device=dev) * 0.15 * 0.5 * (x.roll(1, -1) + x.roll(-1, -1))   # vecinos alrededor del brazo
    x = x + torch.randn_like(x) * torch.rand(B, 1, 1, device=dev) * 6.0
    corr = torch.randint(-20, 21, (B, 1, 1), device=dev)
    idx = (torch.arange(T, device=dev)[None, :, None] + corr) % T
    x = torch.gather(x, 1, idx.expand(B, T, C))
    flojo = (torch.rand(B, 1, C, device=dev) < 0.03).float()
    return x * (1 - 0.9 * flojo)


def perdida_supcon(p1, p2, etiquetas=None, tau=0.1):
    """p1, p2: proyecciones normalizadas (B, d) de las dos vistas.
    Sin etiquetas = SimCLR (NT-Xent). Con etiquetas = SupCon; las de -1 solo
    tienen como positivo su otra vista."""
    B = p1.shape[0]
    z = torch.cat([p1, p2])
    sim = z @ z.T / tau
    n = 2 * B
    yo = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(yo, -1e9)
    pareja = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    pos = torch.zeros(n, n, dtype=torch.bool, device=z.device)
    pos[torch.arange(n, device=z.device), pareja] = True
    if etiquetas is not None:
        y = torch.cat([etiquetas, etiquetas])
        pos = pos | ((y[:, None] == y[None, :]) & (y[:, None] >= 0) & ~yo)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return (-(log_prob * pos).sum(1) / pos.sum(1)).mean()


def _a_gpu(x):
    return torch.as_tensor(np.ascontiguousarray(x), device=DISPOSITIVO)


def _parametros(*modelos):
    return [v.value for m in modelos for v in m.trainable_variables]


@torch.no_grad()
def embeddings(codificador, x, lote=4096, normalizar=False):
    """Embedding de todas las ventanas x (n, 200, 10) con un codificador entrenado."""
    out = []
    for i in range(0, len(x), lote):
        out.append(ops.convert_to_numpy(codificador(_a_gpu(x[i:i + lote]).float(), training=False)))
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True) if normalizar else e


# ------------------------------------------------------------------
# Entrenador
# ------------------------------------------------------------------
class EntrenadorContrastivo:
    """Entrena codificador + proyector y, al final de cada época, mide todo lo que
    se ve en el panel en vivo (proyección 3D, grupos, sondas, R²)."""

    def __init__(self, d1, d4=None, dim=64, lote=512, tasa=1e-3, tau=0.1, n_monitor=2000, semilla=0):
        config.fijar_semillas(semilla)
        self.rng = np.random.default_rng(semilla)
        self.cod = crear_codificador(dim)
        self.proy = crear_proyector(dim)
        self.lote, self.tasa, self.tau = lote, tasa, tau
        tr = np.flatnonzero(d1["split"] == 0)
        self.xa, self.xb = _a_gpu(d1["x_a"][tr]), _a_gpu(d1["x_b"][tr])
        self.y = _a_gpu(d1["gesto"][tr].astype(np.int64))
        self.d4 = None
        if d4 is not None:
            tr4 = np.flatnonzero(d4["split"] == 0)
            self.d4 = (_a_gpu(d4["x_a"][tr4]), _a_gpu(d4["gesto"][tr4].astype(np.int64)))
        # conjuntos fijos de monitoreo (siempre las mismas ventanas, así la animación es comparable)
        va = np.flatnonzero((d1["split"] == 1) & ~d1["transicion"])
        self.i_mon = np.sort(self.rng.choice(va, min(n_monitor, len(va)), replace=False))
        tr_ok = tr[~d1["transicion"][tr]]
        self.i_ref = np.sort(self.rng.choice(tr_ok, min(3 * n_monitor, len(tr_ok)), replace=False))
        self.d1 = d1
        self.mon4 = None
        if d4 is not None:
            v4 = np.flatnonzero((d4["split"] == 1) & (d4["gesto"] >= 0))
            sel = self.rng.choice(v4, min(n_monitor, len(v4)), replace=False)
            self.mon4 = (d4["x_a"][sel], d4["gesto"][sel])
        self.historial = []
        self.instantaneas = []
        self._proy_prev = None
        self.epoca = 0

    # ---------- utilidades ----------
    def embeddings(self, x, lote=4096):
        return embeddings(self.cod, x, lote)

    def registro(self):
        """Lo necesario para redibujar el panel y la animación sin reentrenar."""
        return {"historial": self.historial, "instantaneas": self.instantaneas, "i_mon": self.i_mon}

    def _lote_d4(self, n):
        if self.d4 is None or n == 0:
            return None
        i = torch.randint(0, len(self.d4[1]), (n,), device=DISPOSITIVO)
        return self.d4[0][i].float(), self.d4[1][i]

    # ---------- una época ----------
    def _epoca(self, opt, sched, supervisada):
        n = len(self.y)
        orden = torch.randperm(n, device=DISPOSITIVO)
        n_d4 = self.lote // 4 if self.d4 is not None else 0
        perdidas = []
        for k in range(0, n - self.lote + 1, self.lote):
            i = orden[k:k + self.lote]
            v1, v2, y = self.xa[i].float(), self.xb[i].float(), self.y[i]
            extra = self._lote_d4(n_d4)
            if extra is not None:                           # datos reales: dos aumentos de la misma ventana
                v1 = torch.cat([v1, extra[0]])
                v2 = torch.cat([v2, extra[0]])
                y = torch.cat([y, extra[1]])
            p1 = self.proy(self.cod(aumentar(v1), training=True), training=True)
            p2 = self.proy(self.cod(aumentar(v2), training=True), training=True)
            loss = perdida_supcon(p1, p2, y if supervisada else None, self.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            perdidas.append(float(loss.detach()))
        return float(np.mean(perdidas))

    @torch.no_grad()
    def _perdida_val(self, supervisada):
        d = self.d1
        v1 = _a_gpu(d["x_a"][self.i_mon]).float()
        v2 = _a_gpu(d["x_b"][self.i_mon]).float()
        y = _a_gpu(d["gesto"][self.i_mon].astype(np.int64))
        p1 = self.proy(self.cod(v1, training=False), training=False)
        p2 = self.proy(self.cod(v2, training=False), training=False)
        return float(perdida_supcon(p1, p2, y if supervisada else None, self.tau))

    def _monitorear(self, fase, perdida, perdida_val, segundos):
        d = self.d1
        e_mon = self.embeddings(d["x_a"][self.i_mon])
        e_mon_b = self.embeddings(d["x_b"][self.i_mon])
        e_ref = self.embeddings(d["x_a"][self.i_ref])
        y_mon, y_ref = d["gesto"][self.i_mon].astype(int), d["gesto"][self.i_ref].astype(int)
        en = e_mon / np.linalg.norm(e_mon, axis=1, keepdims=True)
        er = e_ref / np.linalg.norm(e_ref, axis=1, keepdims=True)
        s = metricas.sondas(er, y_ref, en, y_mon, d["pose"][self.i_ref], d["pose"][self.i_mon], NOMBRES_DOF)
        g = metricas.grupos(en, y_mon, semilla=self.epoca)
        alin, unif = metricas.alineacion_uniformidad(e_mon, e_mon_b)
        reg = {"epoca": self.epoca, "fase": fase, "perdida": perdida, "perdida_val": perdida_val,
               "knn": s["knn"], "sonda_logistica": s["sonda_logistica"], "pseudo_r2": s["pseudo_r2_sonda"],
               "r2_pose": s["r2_pose_sonda"], "nmi": g["nmi"], "ari": g["ari"], "pureza": g["pureza"],
               "silueta": g["silueta"], "alineacion": alin, "uniformidad": unif, "segundos": segundos}
        if self.mon4 is not None:                         # ¿lo aprendido en el gemelo sirve en datos reales?
            e4 = self.embeddings(self.mon4[0])
            e4 = e4 / np.linalg.norm(e4, axis=1, keepdims=True)
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(10, metric="cosine").fit(er, y_ref)
            reg["knn_real_uci"] = float(np.mean(knn.predict(e4) == self.mon4[1]))
        # proyección 3D alineada con la época anterior (la nube no salta)
        P = PCA(3, random_state=0).fit_transform(en)
        if self._proy_prev is not None:
            R, _ = orthogonal_procrustes(P, self._proy_prev)
            P = P @ R
        self._proy_prev = P
        self.ridge = s["_ridge"]                          # sonda de pose de esta época (para el visor 3D)
        manos = np.clip(self.ridge.predict(g["gmm"].means_), [0, 0, 0, 0, 0, -1], 1)
        self.historial.append(reg)
        self.instantaneas.append({"epoca": self.epoca, "fase": fase, "proy": P.astype(np.float32),
                                  "grupo": g["asignacion"], "manos": manos,
                                  "gesto_de_grupo": g["gesto_de_grupo"]})
        return reg

    # ---------- bucle principal ----------
    def entrenar(self, fases, al_terminar_epoca=None):
        """fases: lista de (nombre, épocas, supervisada)."""
        total = sum(e for _, e, _ in fases) * (len(self.y) // self.lote)
        opt = torch.optim.AdamW(_parametros(self.cod, self.proy), lr=self.tasa, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.tasa, total_steps=max(total, 1),
                                                    pct_start=0.1)
        if not self.historial:                            # época 0: antes de aprender nada
            reg = self._monitorear("inicio", float("nan"), float("nan"), 0.0)
            if al_terminar_epoca:
                al_terminar_epoca(self, reg)
        for nombre, epocas, supervisada in fases:
            for _ in range(epocas):
                t0 = time.time()
                self.epoca += 1
                perdida = self._epoca(opt, sched, supervisada)
                reg = self._monitorear(nombre, perdida, self._perdida_val(supervisada), time.time() - t0)
                if al_terminar_epoca:
                    al_terminar_epoca(self, reg)
        return self.historial


# ------------------------------------------------------------------
# Cabezas multitarea: gesto + pose (6 GDL) + fuerza, con contexto de 3 ventanas
# ------------------------------------------------------------------
def entradas_cabezas(emb, carac, ventana_idx, k=3):
    ctx = indices_contexto(ventana_idx, k)
    return {"emb_ctx": emb[ctx].reshape(len(emb), -1).astype(np.float32),
            "carac_ctx": carac[ctx].reshape(len(carac), -1).astype(np.float32)}


def crear_cabezas(dim_emb=64, k=3, carac_adaptar=None):
    e = keras.Input((dim_emb * k,), name="emb_ctx")
    c = keras.Input((ia.NUM_CARACTERISTICAS * k,), name="carac_ctx")
    norm = layers.Normalization(name="norm_carac")
    if carac_adaptar is not None:
        norm.adapt(carac_adaptar)
    h = layers.Concatenate(name="union")([e, norm(c)])
    h = layers.Dense(128, activation="relu", name="comun_1")(h)
    h = layers.Dropout(0.2, name="olvido")(h)
    h = layers.Dense(64, activation="relu", name="comun_2")(h)
    gesto = layers.Dense(ia.NUM_GESTOS, activation="softmax", name="gesto")(h)
    hp = layers.Dense(32, activation="relu", name="pose_oculta")(h)
    pose = layers.Concatenate(name="pose")([layers.Dense(5, activation="sigmoid", name="dedos")(hp),
                                            layers.Dense(1, activation="tanh", name="muneca")(hp)])
    fuerza = layers.Dense(1, name="fuerza")(layers.Dense(16, activation="relu", name="fuerza_oculta")(h))
    m = keras.Model({"emb_ctx": e, "carac_ctx": c}, {"gesto": gesto, "pose": pose, "fuerza": fuerza},
                    name="cabezas_multitarea")
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss={"gesto": "sparse_categorical_crossentropy", "pose": "mse", "fuerza": "mse"},
              loss_weights={"gesto": 1.0, "pose": 4.0, "fuerza": 2.0})
    return m


def evaluar_cabezas(modelo, X, d, idx, prior):
    """Todas las métricas de la red multitarea sobre un subconjunto (índices de D1)."""
    sal = modelo.predict({k: v[idx] for k, v in X.items()}, batch_size=4096, verbose=0)
    estable = ~d["transicion"][idx]
    y = d["gesto"][idx].astype(int)
    res = {"gesto_todas": metricas.clasificacion(y, sal["gesto"], prior),
           "gesto_estables": metricas.clasificacion(y[estable], sal["gesto"][estable], prior),
           "pose": metricas.r2(d["pose"][idx], sal["pose"], NOMBRES_DOF),
           "fuerza": metricas.r2(d["fuerza"][idx], sal["fuerza"][:, 0], ["fuerza"])}
    return res, sal
