"""
Métricas del proyecto. Regla: TODO modelo reporta un R².

  Regresión (pose, fuerza, reconstrucción)   R² = 1 - SSE / SST
  Clasificación (gesto)                      pseudo-R² de McFadden = 1 - LL_modelo / LL_nulo
      (LL_nulo = modelo que solo conoce la frecuencia de cada gesto; 0 = no
       aprendió nada, 1 = probabilidades perfectas). Se reporta junto con
       precisión y F1, porque el R² clásico no existe para clases.
  Embedding (contrastivo)                    R² de una sonda lineal (Ridge) que
       predice la pose desde el embedding, + precisión kNN y de sonda logística.
  Grupos (no supervisado)                    NMI, ARI, pureza y silueta.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, f1_score, normalized_mutual_info_score,
                             r2_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from . import config  # noqa: F401
import ia_mano as ia


def r2(y, y_pred, nombres=None):
    """R² por columna y promedio."""
    y, y_pred = np.atleast_2d(np.asarray(y, float).T).T, np.atleast_2d(np.asarray(y_pred, float).T).T
    por_col = r2_score(y, y_pred, multioutput="raw_values")
    nombres = nombres or [f"y{i}" for i in range(len(por_col))]
    return {"r2": float(np.mean(por_col)), "r2_por_salida": dict(zip(nombres, map(float, por_col)))}


def pseudo_r2_mcfadden(y, probs, prior=None):
    y = np.asarray(y)
    probs = np.clip(np.asarray(probs, float), 1e-12, 1)
    if prior is None:
        prior = np.bincount(y, minlength=probs.shape[1]) / len(y)
    ll_modelo = np.sum(np.log(probs[np.arange(len(y)), y]))
    ll_nulo = np.sum(np.log(np.clip(prior[y], 1e-12, 1)))
    return float(1 - ll_modelo / ll_nulo)


def clasificacion(y, probs, prior=None):
    y = np.asarray(y)
    pred = np.asarray(probs).argmax(1)
    cm = confusion_matrix(y, pred, labels=range(ia.NUM_GESTOS))
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {"precision": float(np.mean(pred == y)),
            "f1_macro": float(f1_score(y, pred, average="macro", labels=np.unique(y))),
            "pseudo_r2": pseudo_r2_mcfadden(y, probs, prior),
            "recall": dict(zip(ia.GESTOS, map(float, recall))), "confusion": cm}


def alineacion_uniformidad(z1, z2, max_n=2000):
    """Wang & Isola (2020) sobre embeddings normalizados.
    Alineación: qué tan cerca quedan las dos vistas de lo mismo (menor = mejor).
    Uniformidad: qué tan repartido está el espacio (más negativo = mejor)."""
    z1 = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    z2 = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    alin = float(np.mean(np.sum((z1 - z2) ** 2, axis=1)))
    z = z1[:max_n]
    d2 = np.maximum(2 - 2 * z @ z.T, 0)
    iu = np.triu_indices(len(z), 1)
    unif = float(np.log(np.mean(np.exp(-2 * d2[iu]))))
    return alin, unif


def sondas(emb_tr, y_tr, emb_te, y_te, pose_tr=None, pose_te=None, nombres_pose=None):
    """Qué tanto sirve el embedding sin reentrenarlo: kNN, sonda logística y R² de pose."""
    m = y_tr >= 0
    knn = KNeighborsClassifier(10, metric="cosine").fit(emb_tr[m], y_tr[m])
    log = LogisticRegression(max_iter=300, C=1.0).fit(emb_tr[m], y_tr[m])
    mt = y_te >= 0
    res = {"knn": float(np.mean(knn.predict(emb_te[mt]) == y_te[mt])),
           "sonda_logistica": float(np.mean(log.predict(emb_te[mt]) == y_te[mt])),
           "pseudo_r2_sonda": pseudo_r2_mcfadden(y_te[mt], log.predict_proba(emb_te[mt]))}
    if pose_tr is not None:
        ridge = Ridge(alpha=1.0).fit(emb_tr, pose_tr)
        res["r2_pose_sonda"] = r2(pose_te, ridge.predict(emb_te), nombres_pose)["r2"]
        res["_ridge"] = ridge
    return res


def grupos(emb, y=None, k=ia.NUM_GESTOS, semilla=0):
    """Agrupamiento no supervisado (mezcla gaussiana) y qué tan 'completo' es:
    si los grupos coinciden con los gestos sin haberlos visto."""
    gmm = GaussianMixture(k, covariance_type="diag", random_state=semilla, n_init=2).fit(emb)
    asig = gmm.predict(emb)
    res = {"gmm": gmm, "asignacion": asig, "bic": float(gmm.bic(emb))}
    n_sil = min(len(emb), 2000)
    res["silueta"] = float(silhouette_score(emb[:n_sil], asig[:n_sil])) if len(np.unique(asig[:n_sil])) > 1 else 0.0
    if y is not None:
        m = y >= 0
        res["nmi"] = float(normalized_mutual_info_score(y[m], asig[m]))
        res["ari"] = float(adjusted_rand_score(y[m], asig[m]))
        mayoria, pureza = [], 0
        for c in range(k):
            yc = y[m][asig[m] == c]
            if len(yc):
                cnt = np.bincount(yc, minlength=ia.NUM_GESTOS)
                mayoria.append(int(cnt.argmax()))
                pureza += cnt.max()
            else:
                mayoria.append(-1)
        res["pureza"] = float(pureza / max(m.sum(), 1))
        res["gesto_de_grupo"] = mayoria
    return res
