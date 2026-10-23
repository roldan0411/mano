"""
Panel en vivo (Plotly) y animación del aprendizaje
==================================================

Se redibuja al final de cada época dentro del notebook:

  ┌ espacio contrastivo 3D ──────┬ manos 3D (una por grupo) ────┐
  │ cada punto = 1 ventana EMG   │ pose que la red "imagina"    │
  │ (color = gesto real, solo    │ para cada grupo descubierto  │
  │  para que TÚ juzgues)        │ + pose objetivo en gris      │
  ├ pérdida ─────────────────────┼ sondas y R² ─────────────────┤
  ├ grupos (completitud) ────────┼ alineación / uniformidad ────┤
  └──────────────────────────────┴──────────────────────────────┘

Colores: paleta categórica de referencia en orden fijo; como son 7 gestos,
cada uno lleva además su propio símbolo (codificación secundaria).
"""

import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

from . import config  # noqa: F401
from .datos_emg import POSES_6
from .datos_poses import CONEXIONES, landmarks
import ia_mano as ia

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SIMBOLOS = ["circle", "square", "diamond", "cross", "x", "circle-open", "diamond-open"]
TINTA, TINTA_2, GRIS, FONDO = "#0b0b0b", "#52514e", "#b4b2a9", "#fcfcfb"
SEP_MANOS = 15.0
NOMBRES = [ia.NOMBRES[g] for g in ia.GESTOS]
_EJES_OCULTOS = dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title="")
TABLERO = None     # gemelo.tablero_vivo.TableroVivo: si existe, todo lo que se dibuja llega también al navegador


def _titulo(fig):
    t = fig.layout.title.text or "figura"
    return t.split("<")[0].split(" — época")[0].strip() or "figura"


def mostrar(fig, clave=None):
    """fig.show() que además publica la figura en el tablero en vivo."""
    if TABLERO is not None:
        TABLERO.figura(clave or _titulo(fig), fig)
    display(fig)
    return fig


_N_TABLAS = [0]


def tabla(obj, titulo=None):
    """display() de una tabla (DataFrame/Series) que además la publica en el tablero."""
    display(obj)
    if TABLERO is None or not hasattr(obj, "to_frame") and not hasattr(obj, "columns"):
        return obj
    df = obj.to_frame() if not hasattr(obj, "columns") else obj
    df = df.reset_index()
    _N_TABLAS[0] += 1
    celdas = [[f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v)) for v in df[c]] for c in df.columns]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in df.columns], fill_color="#efeee9", align="left",
                    font=dict(color=TINTA, size=12)),
        cells=dict(values=celdas, align="left", fill_color=FONDO, font=dict(color=TINTA, size=12), height=26)))
    fig.update_layout(title=titulo or "Tabla", height=min(110 + 26 * len(df), 620), paper_bgcolor=FONDO,
                      margin=dict(l=10, r=10, t=40, b=10))
    TABLERO.figura(f"tabla_{_N_TABLAS[0]}", fig)
    return obj


def trazas_nube(P, y, scene="scene", leyenda=True, tam=3):
    trazas = []
    for g in range(ia.NUM_GESTOS):
        m = y == g
        trazas.append(go.Scatter3d(
            x=P[m, 0], y=P[m, 1], z=P[m, 2], mode="markers", name=NOMBRES[g], legendgroup=NOMBRES[g],
            showlegend=leyenda, scene=scene,
            marker=dict(size=tam, color=COLORES[g], symbol=SIMBOLOS[g], opacity=0.75, line=dict(width=0)),
            hovertemplate=f"{NOMBRES[g]}<extra></extra>"))
    return trazas


def _huesos(L):
    x, y, z = [], [], []
    for a, b in CONEXIONES:
        x += [L[a, 0], L[b, 0], None]
        y += [L[a, 1], L[b, 1], None]
        z += [L[a, 2], L[b, 2], None]
    return x, y, z


def trazas_manos(dofs, gestos, etiquetas=None, objetivos=True, scene="scene2"):
    """Una mano por fila de dofs, en fila a lo largo del eje x."""
    trazas, gx, gy, gz = [], [], [], []
    tx, ty, tz, tt = [], [], [], []
    for i, (dof, g) in enumerate(zip(dofs, gestos)):
        off = np.array([SEP_MANOS * i, 0, 0])
        L = landmarks(np.asarray(dof)[None])[0] + off
        color = COLORES[g] if g >= 0 else GRIS
        palma = L[[0, 1, 5, 9, 13, 17]]
        trazas.append(go.Mesh3d(x=palma[:, 0], y=palma[:, 1], z=palma[:, 2], i=[0, 0, 0, 0], j=[1, 2, 3, 4],
                                k=[2, 3, 4, 5], color=color, opacity=0.35, scene=scene, showlegend=False,
                                hoverinfo="skip"))
        x, y, z = _huesos(L)
        nombre = NOMBRES[g] if g >= 0 else "sin gesto"
        trazas.append(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers", scene=scene, showlegend=False,
                                   line=dict(color=color, width=9), marker=dict(size=3, color="white",
                                                                                line=dict(color=color, width=1)),
                                   hovertemplate=f"{nombre}<extra></extra>"))
        if objetivos and g >= 0:
            Lo = landmarks(POSES_6[g][None])[0] + off
            x, y, z = _huesos(Lo)
            gx += x
            gy += y
            gz += z
        tx.append(L[:, 0].mean())
        ty.append(L[:, 1].max() + 2.5)
        tz.append(0)
        tt.append(etiquetas[i] if etiquetas else nombre)
    trazas.append(go.Scatter3d(x=gx or [None], y=gy or [None], z=gz or [None], mode="lines", scene=scene,
                               showlegend=False, line=dict(color=GRIS, width=4), opacity=0.5, hoverinfo="skip"))
    trazas.append(go.Scatter3d(x=tx, y=ty, z=tz, mode="text", text=tt, scene=scene, showlegend=False,
                               textfont=dict(color=TINTA_2, size=11), hoverinfo="skip"))
    return trazas


def escena_manos(n):
    xr = [-7, SEP_MANOS * (n - 1) + 7]
    return dict(xaxis=dict(range=xr, **_EJES_OCULTOS), yaxis=dict(range=[-9, 21], **_EJES_OCULTOS),
                zaxis=dict(range=[-14, 10], **_EJES_OCULTOS), aspectmode="manual",
                aspectratio=dict(x=(xr[1] - xr[0]) / 30, y=1, z=0.8),
                camera=dict(eye=dict(x=0.2, y=-1.3, z=1.0), up=dict(x=0, y=1, z=0)))


def figura_manos(dofs, gestos, titulo, etiquetas=None, objetivos=True, alto=380):
    fig = go.Figure(trazas_manos(dofs, gestos, etiquetas, objetivos, scene="scene"))
    fig.update_layout(scene=escena_manos(len(dofs)), title=titulo, height=alto, paper_bgcolor=FONDO,
                      margin=dict(l=0, r=0, t=40, b=0), font=dict(color=TINTA))
    return fig


def figura_nube(P, y, titulo, alto=520):
    fig = go.Figure(trazas_nube(P, y, scene="scene", tam=2.5))
    fig.update_layout(title=titulo, height=alto, paper_bgcolor=FONDO, font=dict(color=TINTA),
                      scene=dict(xaxis_title="", yaxis_title="", zaxis_title="", aspectmode="cube"),
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"))
    return fig


def _manos_de_instantanea(inst):
    orden = np.argsort([g if g >= 0 else 99 for g in inst["gesto_de_grupo"]], kind="stable")
    dofs = inst["manos"][orden]
    gestos = [inst["gesto_de_grupo"][i] for i in orden]
    etiquetas = [f"grupo {i + 1}: {NOMBRES[g] if g >= 0 else '?'}" for i, g in zip(orden, gestos)]
    return dofs, gestos, etiquetas


def _linea(fig, x, y, nombre, color, fila, col, discontinua=False):
    fig.add_trace(go.Scatter(x=x, y=y, name=nombre, mode="lines+markers", legendgroup=f"m{fila}{col}",
                             line=dict(color=color, width=2, dash="dot" if discontinua else "solid"),
                             marker=dict(size=5), hovertemplate=f"{nombre}: %{{y:.3f}}<extra>época %{{x}}</extra>"),
                  row=fila, col=col)


class PanelContrastivo:
    """Se actualiza en el mismo lugar del notebook en cada época."""

    def __init__(self):
        self.handle = None

    def figura(self, ent):
        """ent: el EntrenadorContrastivo (o un registro con historial, instantaneas, d1 e i_mon)."""
        h, inst = ent.historial, ent.instantaneas[-1]
        r = h[-1]
        fig = make_subplots(
            rows=3, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}], [{}, {}], [{}, {}]],
            row_heights=[0.52, 0.24, 0.24], vertical_spacing=0.08, horizontal_spacing=0.07,
            subplot_titles=["Espacio contrastivo (proyección 3D de 2.000 ventanas de validación)",
                            "Pose que la red asocia a cada grupo descubierto (gris = objetivo)",
                            "Pérdida contrastiva", "Sondas sobre el embedding (0 a 1)",
                            "Completitud de los grupos (no supervisado)", "Alineación y uniformidad"])
        y_mon = ent.d1["gesto"][ent.i_mon]
        for t in trazas_nube(inst["proy"], y_mon, scene="scene"):
            fig.add_trace(t)
        dofs, gestos, etq = _manos_de_instantanea(inst)
        for t in trazas_manos(dofs, gestos, etq, scene="scene2"):
            fig.add_trace(t)
        ep = [x["epoca"] for x in h]
        _linea(fig, ep, [x["perdida"] for x in h], "pérdida entrenamiento", COLORES[0], 2, 1)
        _linea(fig, ep, [x["perdida_val"] for x in h], "pérdida validación", COLORES[0], 2, 1, True)
        _linea(fig, ep, [x["knn"] for x in h], "kNN gesto", COLORES[0], 2, 2)
        _linea(fig, ep, [x["r2_pose"] for x in h], "R² pose (sonda)", COLORES[1], 2, 2)
        _linea(fig, ep, [x["pseudo_r2"] for x in h], "pseudo-R² gesto", COLORES[2], 2, 2)
        if "knn_real_uci" in r:
            _linea(fig, ep, [x["knn_real_uci"] for x in h], "kNN datos reales UCI", COLORES[6], 2, 2, True)
        _linea(fig, ep, [x["nmi"] for x in h], "NMI", COLORES[0], 3, 1)
        _linea(fig, ep, [x["ari"] for x in h], "ARI", COLORES[1], 3, 1)
        _linea(fig, ep, [x["pureza"] for x in h], "pureza", COLORES[2], 3, 1)
        _linea(fig, ep, [x["alineacion"] for x in h], "alineación (menor = mejor)", COLORES[0], 3, 2)
        _linea(fig, ep, [x["uniformidad"] for x in h], "uniformidad (menor = mejor)", COLORES[1], 3, 2)
        cambios = [x["epoca"] for a, x in zip(h, h[1:]) if x["fase"] != a["fase"] and a["fase"] != "inicio"]
        for c in cambios:
            for fila, col in ((2, 1), (2, 2), (3, 1), (3, 2)):
                fig.add_vline(x=c - 0.5, line=dict(color=GRIS, width=1, dash="dash"), row=fila, col=col,
                              exclude_empty_subplots=False)
        for fila, col in ((2, 2), (3, 1)):
            fig.update_yaxes(range=[0, 1.02], row=fila, col=col)
        fig.update_xaxes(title_text="época", row=3)
        fig.update_layout(
            height=1100, paper_bgcolor=FONDO, plot_bgcolor=FONDO, font=dict(color=TINTA, size=11),
            title=dict(text=f"<b>Época {r['epoca']}</b> · fase: {r['fase']} · kNN {r['knn'] * 100:.1f} % · "
                            f"R² pose {r['r2_pose']:.3f} · pseudo-R² {r['pseudo_r2']:.3f} · "
                            f"NMI {r['nmi']:.3f} · {r['segundos']:.1f} s/época", x=0.01),
            scene=dict(xaxis_title="", yaxis_title="", zaxis_title="", aspectmode="cube", uirevision="nube"),
            scene2=escena_manos(len(dofs)), legend=dict(itemsizing="constant", font=dict(size=10)),
            margin=dict(l=40, r=10, t=70, b=40), uirevision="panel")
        return fig

    def actualizar(self, ent, reg=None):
        fig = self.figura(ent)
        if TABLERO is not None:
            TABLERO.figura("panel_contrastivo", fig)
        if self.handle is None:
            self.handle = display(fig, display_id=True)
        else:
            self.handle.update(fig)
        return fig


def animacion(insts, y_mon, titulo="Cómo aprendió el espacio contrastivo, época por época"):
    """Figura con deslizador: recorre todas las instantáneas guardadas."""
    todo = np.concatenate([i["proy"] for i in insts])
    lim = np.abs(todo).max() * 1.05

    def datos(inst):
        dofs, gestos, etq = _manos_de_instantanea(inst)
        return trazas_nube(inst["proy"], y_mon, scene="scene") + trazas_manos(dofs, gestos, etq, scene="scene2")

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], column_widths=[0.45, 0.55],
                        subplot_titles=["Espacio contrastivo", "Pose de cada grupo descubierto"])
    for t in datos(insts[0]):
        fig.add_trace(t)
    fig.frames = [go.Frame(data=datos(i), name=str(i["epoca"]),
                           layout=dict(title_text=f"{titulo} — época {i['epoca']} ({i['fase']})"))
                  for i in insts]
    pasos = [dict(method="animate", label=str(i["epoca"]),
                  args=[[str(i["epoca"])], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
             for i in insts]
    eje = dict(range=[-lim, lim], title="")
    fig.update_layout(
        title=f"{titulo} — época {insts[0]['epoca']}", height=620, paper_bgcolor=FONDO, font=dict(color=TINTA),
        scene=dict(xaxis=eje, yaxis=eje, zaxis=eje, aspectmode="cube"), scene2=escena_manos(len(insts[0]["manos"])),
        sliders=[dict(active=0, steps=pasos, currentvalue=dict(prefix="época "), pad=dict(t=30))],
        updatemenus=[dict(type="buttons", x=0, y=0, xanchor="left", yanchor="top", pad=dict(t=60),
                          buttons=[dict(label="▶ reproducir", method="animate",
                                        args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True)]),
                                   dict(label="⏸ pausa", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=60, b=0))
    return fig


def figura_confusion(cm, titulo, alto=430):
    """Matriz de confusión en % por fila (secuencial de un solo tono)."""
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    texto = [[f"{v:.0f}" if v >= 0.5 else "" for v in fila] for fila in cmn]
    fig = go.Figure(go.Heatmap(z=cmn, x=NOMBRES, y=NOMBRES, zmin=0, zmax=100, text=texto, texttemplate="%{text}",
                               colorscale=[[0, "#fcfcfb"], [1, COLORES[0]]], xgap=2, ygap=2,
                               colorbar=dict(title="%"),
                               hovertemplate="real %{y} → detectado %{x}: %{z:.1f} %<extra></extra>"))
    fig.update_layout(title=titulo, height=alto, paper_bgcolor=FONDO, plot_bgcolor=FONDO, font=dict(color=TINTA),
                      xaxis_title="detectado", yaxis=dict(title="real", autorange="reversed"),
                      margin=dict(l=40, r=10, t=50, b=40))
    return fig


def figura_barras(valores, titulo, eje="R²", rango=(0, 1), alto=320):
    """Barras horizontales de una sola serie (por ejemplo, R² por grado de libertad)."""
    nombres, v = list(valores), list(valores.values())
    fig = go.Figure(go.Bar(x=v, y=nombres, orientation="h", marker=dict(color=COLORES[0], cornerradius=4),
                           text=[f"{x:.3f}" for x in v], textposition="outside",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=titulo, height=alto, paper_bgcolor=FONDO, plot_bgcolor=FONDO, font=dict(color=TINTA),
                      xaxis=dict(title=eje, range=list(rango)), yaxis=dict(autorange="reversed"),
                      margin=dict(l=40, r=30, t=50, b=40), bargap=0.35)
    return fig


class PanelCurvas:
    """Panel simple para entrenamientos con keras.fit: curvas + una nube 3D opcional."""

    def __init__(self, titulo, total=None):
        self.titulo, self.handle, self.total = titulo, None, total

    def actualizar(self, historial, series, nube=None, y_nube=None):
        fig = make_subplots(rows=1, cols=2 if nube is not None else 1,
                            specs=[[{}, {"type": "scene"}]] if nube is not None else [[{}]],
                            column_widths=[0.5, 0.5] if nube is not None else None,
                            subplot_titles=["Métricas por época", "Espacio latente 3D"] if nube is not None else None)
        ep = list(range(1, len(historial) + 1))
        for k, (clave, nombre) in enumerate(series):
            _linea(fig, ep, [h[clave] for h in historial], nombre, COLORES[k % len(COLORES)], 1, 1)
        if nube is not None:
            for t in trazas_nube(nube, y_nube, scene="scene", tam=2):
                fig.add_trace(t)
        fig.update_layout(title=f"{self.titulo} — época {len(historial)}", height=480, paper_bgcolor=FONDO,
                          plot_bgcolor=FONDO, font=dict(color=TINTA), legend=dict(itemsizing="constant"),
                          margin=dict(l=40, r=10, t=60, b=40))
        if TABLERO is not None:
            TABLERO.figura(self.titulo, fig)
            if self.total:
                TABLERO.progreso(min(len(historial), self.total), self.total, self.titulo)
        if self.handle is None:
            self.handle = display(fig, display_id=True)
        else:
            self.handle.update(fig)
        return fig
