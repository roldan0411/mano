"""
Tablero en vivo del entrenamiento (navegador)
=============================================

Mientras el notebook corre, cada sección publica su estado (progreso, figuras,
registro de épocas, R²) en reportes/vivo/estado.json. Un servidor local
(solo 127.0.0.1) sirve reportes/vivo/index.html, que se refresca solo cada
1,5 s: se ve la secuencia completa (terminada / en curso / pendiente) y las
figuras se redibujan sin perder el giro de la cámara.

Al terminar, la misma página queda como tablero completo de todo el diseño.
"""

import json
import os
import re
import shutil
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import plotly

_SERVIDOR = None   # uno por proceso: re-ejecutar la celda no abre otro puerto
WEB = Path(__file__).parent / "web"


class _Manejador(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def _a_json(fig):
    return json.loads(fig.to_json())


class TableroVivo:
    def __init__(self, carpeta, secciones, titulo="Gemelo digital de la mano — entrenamiento en vivo", puerto=8765):
        self.carpeta = Path(carpeta)
        self.puerto = puerto
        self.lock = threading.Lock()
        self.estado = {"titulo": titulo, "inicio": time.time(), "actualizado": time.time(), "actual": None,
                       "terminado": False, "log": [], "r2": [], "episodios": [],
                       "secciones": [{"n": i, "titulo": t, "estado": "pendiente", "inicio": None, "fin": None,
                                      "progreso": None, "figuras": []} for i, t in enumerate(secciones)]}
        self._ultimo = 0.0

    # ---------- servidor ----------
    @property
    def url(self):
        return f"http://127.0.0.1:{self.puerto}/index.html"

    def iniciar(self, abrir=True):
        global _SERVIDOR
        self.carpeta.mkdir(parents=True, exist_ok=True)
        js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if not (self.carpeta / "plotly.min.js").exists():
            shutil.copy(js, self.carpeta / "plotly.min.js")
        for archivo in ("three.min.js", "OrbitControls.js"):            # visor 3D (Three.js r147, MIT)
            shutil.copy(WEB / archivo, self.carpeta / archivo)
        if not (self.carpeta / "mano_inmoov.json").exists():
            try:
                from . import mano_inmoov
                mano_inmoov.exportar_web(self.carpeta / "mano_inmoov.json")
            except Exception as e:                                       # sin el .zip: el visor se oculta
                print("Visor 3D sin mano InMoov:", e)
        (self.carpeta / "index.html").write_text(
            PAGINA.replace("__PLOTLY__", "plotly.min.js").replace("__THREE__", "three.min.js")
            .replace("__ORBIT__", "OrbitControls.js"), encoding="utf-8")
        self._guardar(forzar=True)
        if _SERVIDOR is None:
            for p in range(self.puerto, self.puerto + 10):
                try:
                    _SERVIDOR = ThreadingHTTPServer(("127.0.0.1", p), partial(_Manejador, directory=str(self.carpeta)))
                    break
                except OSError:
                    continue
            threading.Thread(target=_SERVIDOR.serve_forever, daemon=True).start()
        self.puerto = _SERVIDOR.server_address[1]
        if abrir:
            webbrowser.open(self.url)
        return self.url

    # ---------- publicar ----------
    def _escribir(self, ruta, texto):
        tmp = ruta.with_suffix(".tmp")
        with self.lock:
            tmp.write_text(texto, encoding="utf-8")
            for _ in range(20):                           # en Windows el lector puede tener el archivo abierto
                try:
                    os.replace(tmp, ruta)
                    return
                except PermissionError:
                    time.sleep(0.05)

    def _guardar(self, forzar=False):
        ahora = time.time()
        if not forzar and ahora - self._ultimo < 0.5:     # como máximo 2 escrituras por segundo
            return
        self._ultimo = ahora
        self.estado["actualizado"] = ahora
        self._escribir(self.carpeta / "estado.json", json.dumps(self.estado, ensure_ascii=False, default=float))

    def _sec(self):
        a = self.estado["actual"]
        return None if a is None else self.estado["secciones"][a]

    def seccion(self, n):
        """Marca el inicio de la sección n (y termina la anterior)."""
        s = self._sec()
        if s is not None and s["estado"] == "en curso":
            s["estado"], s["fin"], s["progreso"] = "terminada", time.time(), None
        self.estado["actual"] = n
        s = self.estado["secciones"][n]
        s["estado"], s["inicio"], s["fin"], s["figuras"] = "en curso", time.time(), None, []
        self.log(f"▶ {s['titulo']}")
        self._guardar(forzar=True)

    def progreso(self, paso, total, texto=""):
        s = self._sec()
        if s is not None:
            s["progreso"] = {"paso": paso, "total": total, "texto": texto}
            self._guardar(forzar=paso >= total)

    def figura(self, clave, fig, titulo=None):
        """Agrega o reemplaza (misma clave) una figura de la sección en curso."""
        s = self._sec()
        if s is None:
            return
        datos = _a_json(fig)
        datos.setdefault("layout", {})["uirevision"] = clave
        archivo = f"figs/{s['n']:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', clave)[:60]}.json"
        (self.carpeta / "figs").mkdir(exist_ok=True)
        self._escribir(self.carpeta / archivo, json.dumps(datos, ensure_ascii=False))
        for f in s["figuras"]:
            if f["clave"] == clave:
                f["v"] += 1
                break
        else:
            s["figuras"].append({"clave": clave, "titulo": titulo, "archivo": archivo, "v": 1})
        self._guardar(forzar=True)

    def episodio(self, datos):
        """Un intento de la mano (gemelo.episodios): el visor 3D lo reproduce en tiempo real."""
        s = self._sec()
        n = len(self.estado["episodios"]) + 1
        archivo = f"episodios/ep_{n:04d}.json"
        (self.carpeta / "episodios").mkdir(exist_ok=True)
        self._escribir(self.carpeta / archivo, json.dumps(datos, ensure_ascii=False))
        self.estado["episodios"].append({"n": n, "titulo": datos["titulo"], "archivo": archivo,
                                         "seccion": None if s is None else s["n"]})
        self._guardar(forzar=True)

    def log(self, linea):
        self.estado["log"] = (self.estado["log"] + [f"{time.strftime('%H:%M:%S')}  {linea}"])[-300:]
        self._guardar()

    def r2(self, fila):
        self.estado["r2"].append(fila)
        self._guardar(forzar=True)

    def terminar(self):
        s = self._sec()
        if s is not None:
            s["estado"], s["fin"], s["progreso"] = "terminada", time.time(), None
        self.estado["terminado"] = True
        self.log("✔ Entrenamiento completo")
        self._guardar(forzar=True)
        return self.guardar_estatico()

    def guardar_estatico(self, ruta=None):
        """El tablero completo en un solo archivo que se abre sin servidor (reportes/tablero.html)."""
        ruta = Path(ruta or self.carpeta.parent / "tablero.html")
        estado = json.loads(json.dumps(self.estado, default=float))
        for sec in estado["secciones"]:
            for f in sec["figuras"]:
                f["fig"] = json.loads((self.carpeta / f["archivo"]).read_text(encoding="utf-8"))
        ultimos = {}                                   # el último intento de cada sección va incrustado
        for e in estado["episodios"]:
            ultimos[e["seccion"]] = e
        estado["episodios"] = list(ultimos.values())
        for e in estado["episodios"]:
            e["datos"] = json.loads((self.carpeta / e["archivo"]).read_text(encoding="utf-8"))
        datos = json.dumps(estado, ensure_ascii=False).replace("</", "<\\/")
        mano = self.carpeta / "mano_inmoov.json"
        incrustar = f"window.ESTADO = {datos};"
        if mano.exists():
            incrustar += f"\nwindow.MANO = {mano.read_text(encoding='utf-8')};"
        rel = lambda a: os.path.relpath(self.carpeta / a, ruta.parent).replace(os.sep, "/")
        ruta.write_text(PAGINA.replace("__PLOTLY__", rel("plotly.min.js")).replace("__THREE__", rel("three.min.js"))
                        .replace("__ORBIT__", rel("OrbitControls.js")).replace("/*__ESTADO__*/", incrustar),
                        encoding="utf-8")
        return ruta


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemelo mano en vivo</title>
<script src="__PLOTLY__"></script>
<script src="__THREE__"></script>
<script src="__ORBIT__"></script>
<script>/*__ESTADO__*/</script>
<style>
:root { --fondo:#f6f5f2; --sup:#fcfcfb; --tinta:#0b0b0b; --tinta2:#52514e; --borde:#e2e0da; --acento:#2a78d6;
        --ok:#008300; --codigo:#efeee9; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --fondo:#121211; --sup:#1a1a19;
  --tinta:#f0efec; --tinta2:#c3c2b7; --borde:#33322f; --acento:#3987e5; --ok:#2fa52f; --codigo:#252523; } }
:root[data-theme="dark"] { --fondo:#121211; --sup:#1a1a19; --tinta:#f0efec; --tinta2:#c3c2b7; --borde:#33322f;
  --acento:#3987e5; --ok:#2fa52f; --codigo:#252523; }
* { box-sizing:border-box; }
body { margin:0; background:var(--fondo); color:var(--tinta); font:15px/1.5 system-ui,"Segoe UI",Roboto,sans-serif; }
header { max-width:1280px; margin:0 auto; padding:22px 16px 6px; display:flex; flex-wrap:wrap; gap:8px 20px;
         align-items:baseline; justify-content:space-between; }
header h1 { margin:0; font-size:22px; }
#reloj { color:var(--tinta2); font-size:13px; font-variant-numeric:tabular-nums; }
.pasos { position:sticky; top:0; z-index:5; background:var(--fondo); border-bottom:1px solid var(--borde); }
.pasos div { max-width:1280px; margin:0 auto; padding:8px 16px; display:flex; gap:6px; overflow-x:auto; }
.paso { white-space:nowrap; font-size:12.5px; padding:4px 10px; border-radius:999px; border:1px solid var(--borde);
        color:var(--tinta2); text-decoration:none; }
.paso.terminada { color:var(--ok); border-color:var(--ok); }
.paso.encurso { color:#fff; background:var(--acento); border-color:var(--acento); }
main { max-width:1280px; margin:0 auto; padding:14px 16px 40px; display:grid; gap:16px;
       grid-template-columns:minmax(0,1fr) 340px; }
@media (max-width: 960px) { main { grid-template-columns:minmax(0,1fr); } }
.tarjeta { background:var(--sup); border:1px solid var(--borde); border-radius:12px; padding:16px; }
.tarjeta h2 { margin:0 0 4px; font-size:18px; }
.tarjeta h3 { margin:0 0 8px; font-size:15px; }
.meta { color:var(--tinta2); font-size:13px; }
.barra { height:8px; background:var(--codigo); border-radius:999px; overflow:hidden; margin:10px 0 4px; }
.barra i { display:block; height:100%; background:var(--acento); border-radius:999px; transition:width .4s; }
.figura { width:100%; min-height:360px; margin-top:10px; }
#log { background:var(--codigo); border-radius:8px; padding:8px 10px; font:12px/1.45 ui-monospace,Consolas,monospace;
       height:360px; overflow:auto; white-space:pre-wrap; }
table { border-collapse:collapse; width:100%; font-size:12.5px; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--borde); padding:4px 6px; text-align:left; vertical-align:top; }
td.n { text-align:right; font-weight:600; }
.lado { display:grid; gap:16px; align-content:start; }
label { font-size:13px; color:var(--tinta2); }
.vacio { color:var(--tinta2); font-size:13px; }
.columna { display:grid; gap:16px; align-content:start; min-width:0; }
.visor-cab { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between; }
.visor-cab select, .visor-cab button { font:inherit; font-size:13px; padding:3px 8px; border-radius:8px;
  border:1px solid var(--borde); background:var(--sup); color:var(--tinta); max-width:100%; }
.lienzo-3d { position:relative; height:460px; border-radius:10px; overflow:hidden; margin-top:8px;
  background:linear-gradient(180deg, var(--codigo), var(--sup)); }
#mano3d { width:100%; height:100%; display:block; cursor:grab; }
.hud { position:absolute; left:10px; top:10px; right:10px; display:flex; flex-wrap:wrap; gap:6px; pointer-events:none; }
.chip { background:color-mix(in srgb, var(--sup) 88%, transparent); border:1px solid var(--borde); border-radius:999px;
  padding:2px 10px; font-size:12.5px; font-variant-numeric:tabular-nums; }
.chip b { font-weight:600; }
.chip.ok { border-color:var(--ok); color:var(--ok); }
.chip.mal { border-color:#e34948; color:#e34948; }
.leyenda3d { position:absolute; left:10px; bottom:8px; font-size:12px; color:var(--tinta2); pointer-events:none; }
#emg { width:100%; height:86px; display:block; margin-top:8px; border-radius:8px; background:var(--codigo); }
</style></head>
<body>
<header><h1 id="titulo">Gemelo digital de la mano — entrenamiento en vivo</h1>
  <span id="reloj">conectando…</span></header>
<nav class="pasos"><div id="pasos"></div></nav>
<main>
  <div class="columna">
    <section class="tarjeta" id="visor" hidden>
      <div class="visor-cab"><h2>Mano InMoov — intentos en tiempo real</h2>
        <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center">
          <select id="ep-lista" aria-label="intento"></select>
          <button id="ep-pausa" type="button">⏸ pausa</button>
          <label><input type="checkbox" id="ep-ultimo" checked> seguir el último intento</label>
        </div></div>
      <div class="meta" id="ep-titulo">Esperando el primer intento de la red…</div>
      <div class="lienzo-3d"><canvas id="mano3d"></canvas><div class="hud" id="hud"></div>
        <div class="leyenda3d">sólida = lo que hace la mano · azul transparente = pose objetivo · arrastra para girar</div></div>
      <canvas id="emg" aria-label="señal EMG del intento"></canvas>
    </section>
    <div id="secciones" class="columna"></div>
  </div>
  <aside class="lado">
    <div class="tarjeta"><h3>Resumen R² (crece en vivo)</h3><div id="r2"><p class="vacio">Aún sin modelos.</p></div></div>
    <div class="tarjeta"><h3>Registro</h3>
      <label><input type="checkbox" id="seguir"> seguir la sección en curso</label>
      <div id="log"></div></div>
  </aside>
</main>
<script>
const dibujadas = {};      // clave -> versión ya dibujada
let actualPrevio = null;
const dur = (s) => { s = Math.max(0, Math.round(s)); return (s >= 60 ? Math.floor(s / 60) + " min " : "") + (s % 60) + " s"; };
const esc = (t) => String(t ?? "").replace(/[&<>]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));

function tarjeta(s) {
  let el = document.getElementById("sec" + s.n);
  if (!el) {
    el = document.createElement("section");
    el.id = "sec" + s.n; el.className = "tarjeta";
    el.innerHTML = `<h2>${esc(s.titulo)}</h2><div class="meta"></div><div class="barra" hidden><i></i></div>
                    <div class="meta prog"></div><div class="figs"></div>`;
    const cont = document.getElementById("secciones");
    const despues = [...cont.children].find((c) => +c.id.slice(3) > s.n);
    cont.insertBefore(el, despues || null);
  }
  return el;
}

async function dibujar(div, f) {
  const fig = f.fig || await (await fetch(f.archivo + "?v=" + f.v, {cache: "no-store"})).json();
  const lay = Object.assign({}, fig.layout, {autosize: true});
  delete lay.width;
  await Plotly.react(div, fig.data, lay, {responsive: true, displaylogo: false});
  if (fig.frames) Plotly.addFrames(div, fig.frames);
}

// ================= Visor 3D: la mano InMoov intentando los gestos =================
const Visor = (() => {
  if (!window.THREE) return null;
  const DEDOS = ["pulgar", "indice", "medio", "anular", "menique"];
  const lienzo = document.getElementById("mano3d"), lienzoEmg = document.getElementById("emg");
  const renderer = new THREE.WebGLRenderer({canvas: lienzo, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const escena = new THREE.Scene();
  const camara = new THREE.PerspectiveCamera(32, 1, 1, 5000);
  camara.position.set(250, 210, -360);
  const control = new THREE.OrbitControls(camara, lienzo);
  control.target.set(0, 85, 0); control.enableDamping = true; control.update();
  escena.add(new THREE.HemisphereLight(0xffffff, 0x7d7b75, 0.95));
  const sol = new THREE.DirectionalLight(0xffffff, 0.75); sol.position.set(220, 420, -320); escena.add(sol);
  const contra = new THREE.DirectionalLight(0xffffff, 0.35); contra.position.set(-260, 120, 300); escena.add(contra);
  const antebrazo = new THREE.Mesh(new THREE.CylinderGeometry(31, 37, 170, 40),
    new THREE.MeshStandardMaterial({color: 0xc9c6bd, roughness: 0.85}));
  antebrazo.position.set(0, -88, 0); escena.add(antebrazo);

  const b64 = (s, Tipo) => { const t = atob(s), u = new Uint8Array(t.length);
    for (let i = 0; i < t.length; i++) u[i] = t.charCodeAt(i); return new Tipo(u.buffer); };
  let geometrias = null, plantilla = null, real = null, fantasma = null;

  function crearMano(material) {
    const P = plantilla, raiz = new THREE.Group(), dedos = {};
    for (const k of ["palma", "dorso", "nudillos"]) if (geometrias[k]) raiz.add(new THREE.Mesh(geometrias[k], material(k)));
    for (const dedo of DEDOS) {
      if (dedo === "pulgar") {
        dedos.pulgar = {segs: [0, 1].map((k) => { const g = new THREE.Group();
          g.add(new THREE.Mesh(geometrias["pulgar_" + k], material("dedo"))); raiz.add(g); return g; })};
        continue;
      }
      const base = new THREE.Group(); base.position.fromArray(P.base_dedos[dedo]); raiz.add(base);
      let padre = base; const juntas = [];
      for (let k = 0; k < 3; k++) {
        const j = new THREE.Group();
        if (k > 0) j.position.set(0, P.largos[dedo][k - 1], 0);
        j.add(new THREE.Mesh(geometrias[dedo + "_" + k], material("dedo")));
        padre.add(j); juntas.push(j); padre = j;
      }
      dedos[dedo] = {base, juntas};
    }
    escena.add(raiz);
    return {raiz, dedos};
  }

  const EJE_Y = new THREE.Vector3(0, 1, 0);
  function poner(m, dof) {           // misma cinemática que gemelo.mano_inmoov.poses_mundo
    const P = plantilla;
    m.raiz.rotation.x = P.muneca_max * Math.max(-1, Math.min(1, dof[5]));
    DEDOS.forEach((dedo, i) => {
      const f = Math.max(0, Math.min(1, dof[i])), d = m.dedos[dedo];
      if (dedo === "pulgar") {
        const d0 = new THREE.Vector3().fromArray(P.dir_pulgar[0]), d1 = new THREE.Vector3().fromArray(P.dir_pulgar[1]);
        const pos = new THREE.Vector3().fromArray(P.base_pulgar);
        let acum = 0;
        d.segs.forEach((g, k) => {
          acum = Math.min(1, acum + f * 0.45 * (k ? 1.5 : 1));
          const dir = d0.clone().multiplyScalar(1 - acum).add(d1.clone().multiplyScalar(acum)).normalize();
          g.position.copy(pos); g.quaternion.setFromUnitVectors(EJE_Y, dir);
          pos.add(dir.multiplyScalar(P.largos.pulgar[k]));
        });
        return;
      }
      d.base.rotation.z = -P.separacion[i] * (1 - 0.8 * f);
      d.juntas.forEach((j, k) => { j.rotation.x = -f * P.flex_max[k]; });
    });
  }

  function iniciar(datos) {
    plantilla = datos.plantilla; geometrias = {};
    for (const [k, p] of Object.entries(datos.piezas)) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(b64(p.posiciones, Float32Array), 3));
      g.setIndex(new THREE.BufferAttribute(b64(p.indices, Uint32Array), 1));
      g.computeVertexNormals(); geometrias[k] = g;
    }
    const plastico = {palma: 0xe9e6dd, dorso: 0xf4f2ec, nudillos: 0xd6d3ca, dedo: 0xdcd9cf};
    const mats = {};
    real = crearMano((k) => mats[k] ||= new THREE.MeshStandardMaterial({color: plastico[k], roughness: 0.55, metalness: 0.02}));
    const vidrio = new THREE.MeshStandardMaterial({color: 0x2a78d6, transparent: true, opacity: 0.2, depthWrite: false,
                                                   roughness: 0.4});
    fantasma = crearMano(() => vidrio);
    poner(real, [0.25, 0.25, 0.25, 0.25, 0.25, 0]); poner(fantasma, [0.25, 0.25, 0.25, 0.25, 0.25, 0]);
    document.getElementById("visor").hidden = false;
    requestAnimationFrame(cuadro);
  }

  // ---------- reproducción en tiempo real ----------
  let ep = null, t0 = 0, pausa = false, tPausa = 0, pendiente = null, lista = [], cargado = 0, fondoEmg = null;
  const seguir = () => document.getElementById("ep-ultimo").checked;

  function mostrar(datos) {
    ep = datos; t0 = performance.now(); tPausa = 0;
    document.getElementById("ep-titulo").textContent = datos.titulo;
    fondoEmg = null;
  }

  function dibujarEmg(avance) {
    const c = lienzoEmg, w = c.clientWidth, h = c.clientHeight, dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); fondoEmg = null; }
    const g = c.getContext("2d");
    if (!fondoEmg) {
      fondoEmg = document.createElement("canvas"); fondoEmg.width = c.width; fondoEmg.height = c.height;
      const f = fondoEmg.getContext("2d");
      if (ep && ep.emg) {
        const canales = [[ep.emg.c1, "#2a78d6", 0.28], [ep.emg.c2, "#1baf7a", 0.72]];
        const n = ep.emg.c1.length, esc = 0.42 * c.height / Math.max(1, ...ep.emg.c1.map(Math.abs), ...ep.emg.c2.map(Math.abs));
        f.lineWidth = dpr;
        for (const [s, color, y0] of canales) {
          f.strokeStyle = color; f.beginPath();
          for (let i = 0; i < n; i++) { const x = i / (n - 1) * c.width, y = y0 * c.height - s[i] * esc * 0.55;
            i ? f.lineTo(x, y) : f.moveTo(x, y); }
          f.stroke();
        }
        f.fillStyle = "#52514e"; f.font = `${11 * dpr}px system-ui`;
        f.fillText("EMG canal 1 · flexores", 6 * dpr, 13 * dpr); f.fillText("EMG canal 2 · extensores", 6 * dpr, 0.56 * c.height);
      } else {
        f.fillStyle = "#52514e"; f.font = `${12 * dpr}px system-ui`;
        f.fillText("Este intento no viene de una señal EMG (refuerzo de servos o red de formas).", 8 * dpr, c.height / 2);
      }
    }
    g.clearRect(0, 0, c.width, c.height); g.drawImage(fondoEmg, 0, 0);
    g.fillStyle = "rgba(42,120,214,0.15)"; g.fillRect(0, 0, avance * c.width, c.height);
    g.fillStyle = "#e34948"; g.fillRect(avance * c.width - dpr, 0, 2 * dpr, c.height);
  }

  function hud(i, t, total) {
    const n = ep.nombres, g = ep.gesto[i], p = ep.pred_gesto ? ep.pred_gesto[i] : null;
    let html = `<span class="chip">objetivo: <b>${esc(n[g])}</b></span>`;
    if (p !== null) html += `<span class="chip ${p === g ? "ok" : "mal"}">la red detecta: <b>${esc(n[p])}</b> ${p === g ? "✓" : "✗"}</span>`;
    for (const [k, v] of Object.entries(ep.metricas)) html += `<span class="chip">${esc(k)} <b>${v}</b></span>`;
    html += `<span class="chip">${t.toFixed(1)} / ${total.toFixed(1)} s</span>`;
    const nuevos = lista.length ? lista[lista.length - 1].n - cargado : 0;
    if (nuevos > 0 && seguir()) html += `<span class="chip">${nuevos} intento(s) más nuevos en cola</span>`;
    document.getElementById("hud").innerHTML = html;
  }

  function cuadro(ahora) {
    requestAnimationFrame(cuadro);
    const w = lienzo.clientWidth, h = lienzo.clientHeight;
    if (lienzo.width !== Math.round(w * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camara.aspect = w / h; camara.updateProjectionMatrix();
    }
    if (ep) {
      const T = ep.dof.length, total = T * ep.dt;
      let t = pausa ? tPausa : (ahora - t0) / 1000;
      if (!pausa && t > total + 0.8) {                 // terminó el intento: el más nuevo, o se repite
        if (pendiente && seguir()) { mostrar(pendiente); pendiente = null; } else { t0 = ahora; }
        t = 0;
      }
      const x = Math.min(Math.max(t, 0) / ep.dt, T - 1), i = Math.floor(x), a = x - i, j = Math.min(i + 1, T - 1);
      const mezcla = (A) => A[i].map((v, k) => v + (A[j][k] - v) * a);
      poner(real, mezcla(ep.dof)); poner(fantasma, mezcla(ep.objetivo));
      hud(i, Math.min(t, total), total); dibujarEmg(Math.min(t / total, 1));
    }
    control.update();
    renderer.render(escena, camara);
  }

  document.getElementById("ep-pausa").onclick = (ev) => {
    if (!ep) return;
    if (pausa) { t0 = performance.now() - tPausa * 1000; } else { tPausa = (performance.now() - t0) / 1000; }
    pausa = !pausa; ev.target.textContent = pausa ? "▶ seguir" : "⏸ pausa";
  };
  async function traer(e) { return e.datos || await (await fetch(e.archivo, {cache: "no-store"})).json(); }
  document.getElementById("ep-lista").onchange = async (ev) => {
    const e = lista.find((x) => String(x.n) === ev.target.value);
    if (!e) return;
    document.getElementById("ep-ultimo").checked = false; pendiente = null;
    mostrar(await traer(e));
  };

  return {
    iniciar,
    async episodios(eps) {
      if (!eps || !eps.length || !geometrias) return;
      if (eps.length !== lista.length) {
        lista = eps;
        const sel = document.getElementById("ep-lista"), actual = sel.value;
        sel.innerHTML = eps.slice().reverse().slice(0, 300).map((e) => `<option value="${e.n}">#${e.n} · ${esc(e.titulo)}</option>`).join("");
        if (actual) sel.value = actual;
      }
      const ultimo = eps[eps.length - 1];
      if (ultimo.n === cargado) return;
      cargado = ultimo.n;
      const datos = await traer(ultimo);
      if (!ep) mostrar(datos); else if (seguir()) pendiente = datos;
    },
  };
})();

if (Visor) {
  if (window.MANO) Visor.iniciar(window.MANO);
  else fetch("mano_inmoov.json").then((r) => r.ok ? r.json() : null).then((d) => d && Visor.iniciar(d)).catch(() => {});
}

function pintar(e) {
  if (Visor) Visor.episodios(e.episodios);
  document.getElementById("titulo").textContent = e.titulo;
  const ahora = Date.now() / 1000;
  document.getElementById("reloj").textContent = (e.terminado ? "terminado · " : "en curso · ") +
    "duración " + dur((e.terminado ? e.actualizado : ahora) - e.inicio);
  document.getElementById("pasos").innerHTML = e.secciones.map((s) =>
    `<a class="paso ${s.estado.replace(" ", "")}" href="#sec${s.n}">${s.estado === "terminada" ? "✓" :
      s.estado === "en curso" ? "●" : "○"} ${esc(s.titulo)}</a>`).join("");
  for (const s of e.secciones) {
    if (s.estado === "pendiente") continue;
    const el = tarjeta(s);
    el.querySelector(".meta").textContent = s.estado === "terminada"
      ? `terminada en ${dur(s.fin - s.inicio)}` : `en curso desde hace ${dur(ahora - s.inicio)}`;
    const barra = el.querySelector(".barra"), prog = el.querySelector(".prog");
    if (s.progreso && s.estado === "en curso") {
      barra.hidden = false;
      barra.querySelector("i").style.width = (100 * s.progreso.paso / s.progreso.total).toFixed(1) + "%";
      prog.textContent = `${s.progreso.paso} / ${s.progreso.total}  ${s.progreso.texto || ""}`;
    } else { barra.hidden = true; prog.textContent = ""; }
    const figs = el.querySelector(".figs");
    for (const f of s.figuras) {
      const id = "f" + s.n + "_" + f.clave.replace(/[^a-zA-Z0-9]/g, "_");
      let div = document.getElementById(id);
      if (!div) { div = document.createElement("div"); div.id = id; div.className = "figura"; figs.appendChild(div); }
      const k = s.n + "|" + f.clave;
      if (dibujadas[k] === f.v) continue;
      dibujadas[k] = f.v;
      dibujar(div, f);
    }
  }
  if (e.r2.length) {
    document.getElementById("r2").innerHTML = "<table><tr><th>modelo · tarea</th><th>R²</th></tr>" +
      e.r2.map((r) => `<tr><td>${esc(r.modelo)}<br><span class="meta">${esc(r.tarea)} · ${esc(r["tipo de R²"])}
        · ${esc(r.datos)}</span></td><td class="n">${Number(r["R²"]).toFixed(3)}</td></tr>`).join("") + "</table>";
  }
  const log = document.getElementById("log");
  const abajo = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = e.log.join("\n");
  if (abajo) log.scrollTop = log.scrollHeight;
  if (document.getElementById("seguir").checked && e.actual !== null && e.actual !== actualPrevio && !e.terminado) {
    document.getElementById("sec" + e.actual)?.scrollIntoView({behavior: "smooth", block: "start"});
  }
  actualPrevio = e.actual;
}

async function consultar() {
  try {
    const r = await fetch("estado.json?t=" + Date.now(), {cache: "no-store"});
    pintar(await r.json());
  } catch (err) {
    document.getElementById("reloj").textContent = "esperando al notebook…";
  }
}
if (window.ESTADO) { pintar(window.ESTADO); }          // tablero guardado: sin servidor
else { consultar(); setInterval(consultar, 1500); }
</script>
</body></html>
"""
