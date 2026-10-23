"""
Almacén del motor: todo lo que el gemelo genera y aprende queda en disco
========================================================================

  C:\\Users\\<usuario>\\gemelo_mano\\motor\\
    gemelo.db                 SQLite: lotes, generaciones, candidatas, campeonas, eventos
    examen.h5                 examen fijo (personas que ninguna mano ve al entrenar)
    lotes\\lote_000123.h5      datos simulados (señal cruda + características + calibración)
    poblacion\\mano_03.keras   estado actual de cada mano (para retomar al reabrir)
    campeonas\\c0007_g0412\\   modelo, cabecera para el ESP32, ficha y fotos de los 7 gestos

Retención: los lotes más viejos se borran al pasar el límite de disco; las
campeonas (las últimas 30) y el examen no se borran.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

import h5py
import numpy as np

from .. import config

RAIZ = config.RAIZ_DATOS / "motor"
LIMITE_LOTES_GB = 3.0
MAX_CAMPEONAS = 30
CAMPOS_MEMORIA = ("X", "gesto", "pose", "fuerza", "transicion", "persona", "sesion")


class Almacen:
    def __init__(self, raiz=RAIZ):
        self.raiz = Path(raiz)
        for c in ("lotes", "poblacion", "campeonas"):
            (self.raiz / c).mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.raiz / "gemelo.db", check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS lotes(id INTEGER PRIMARY KEY, t REAL, archivo TEXT, ventanas INT, personas INT,
                                         sesiones INT, saturadas INT, mb REAL);
        CREATE TABLE IF NOT EXISTS generaciones(id INTEGER PRIMARY KEY, t REAL, mejor REAL, media REAL, ventanas INT,
                                                segundos REAL);
        CREATE TABLE IF NOT EXISTS candidatas(gen INT, mano INT, puntaje REAL, acierto REAL, pseudo_r2 REAL,
                                              r2_pose REAL, genoma TEXT, evento TEXT);
        CREATE TABLE IF NOT EXISTS campeonas(id INTEGER PRIMARY KEY, gen INT, t REAL, mano INT, puntaje REAL,
                                             acierto REAL, pseudo_r2 REAL, r2_pose REAL, genoma TEXT, carpeta TEXT);
        CREATE TABLE IF NOT EXISTS eventos(t REAL, gen INT, texto TEXT);
        """)
        self.db.commit()

    # ---------- base de datos ----------
    def sql(self, consulta, args=(), uno=False):
        with self.lock:
            cur = self.db.execute(consulta, args)
            filas = cur.fetchall()
            self.db.commit()
        return (filas[0] if filas else None) if uno else filas

    def evento(self, gen, texto):
        self.sql("INSERT INTO eventos VALUES (?,?,?)", (time.time(), gen, texto))

    # ---------- lotes de datos ----------
    def guardar_lote(self, d, cal, saturadas):
        n = (self.sql("SELECT COALESCE(MAX(id), 0) FROM lotes", uno=True)[0] or 0) + 1
        ruta = self.raiz / "lotes" / f"lote_{n:06d}.h5"
        with h5py.File(ruta, "w") as f:
            for k, v in d.items():
                f.create_dataset(k, data=v, compression="gzip" if v.nbytes > 1e6 else None)
            f.create_dataset("cal_sesion", data=np.array(list(cal.keys()), np.int32))
            f.create_dataset("cal_valor", data=np.stack(list(cal.values())))
        mb = ruta.stat().st_size / 1e6
        self.sql("INSERT INTO lotes VALUES (?,?,?,?,?,?,?,?)",
                 (n, time.time(), ruta.name, int(len(d["gesto"])), int(len(np.unique(d["persona"]))),
                  int(len(cal)), int(saturadas), mb))
        self._retener()
        return n, ruta

    def _retener(self):
        filas = self.sql("SELECT id, archivo, mb FROM lotes ORDER BY id DESC")
        total = 0.0
        for i, archivo, mb in filas:
            total += mb or 0
            if total > LIMITE_LOTES_GB * 1000:
                (self.raiz / "lotes" / archivo).unlink(missing_ok=True)
                self.sql("UPDATE lotes SET archivo = '' WHERE id = ?", (i,))

    def leer_lote(self, archivo, campos=CAMPOS_MEMORIA):
        with h5py.File(self.raiz / "lotes" / archivo, "r") as f:
            return {k: f[k][()] for k in campos}

    def lotes_en_disco(self, ultimos=400):
        return [a for (a,) in self.sql("SELECT archivo FROM lotes WHERE archivo != '' ORDER BY id DESC LIMIT ?",
                                       (ultimos,))][::-1]

    # ---------- examen ----------
    def examen(self):
        ruta = self.raiz / "examen.h5"
        if not ruta.exists():
            return None
        with h5py.File(ruta, "r") as f:
            return {k: f[k][()] for k in f.keys()}

    def guardar_examen(self, d):
        with h5py.File(self.raiz / "examen.h5", "w") as f:
            for k in CAMPOS_MEMORIA:
                f.create_dataset(k, data=d[k])

    # ---------- campeonas ----------
    def carpeta_campeona(self, n, gen):
        c = self.raiz / "campeonas" / f"c{n:04d}_g{gen:05d}"
        c.mkdir(parents=True, exist_ok=True)
        return c

    def registrar_campeona(self, gen, mano, m, genoma, carpeta):
        self.sql("INSERT INTO campeonas(gen, t, mano, puntaje, acierto, pseudo_r2, r2_pose, genoma, carpeta) "
                 "VALUES (?,?,?,?,?,?,?,?,?)", (gen, time.time(), mano, m["puntaje"], m["acierto"], m["pseudo_r2"],
                                                m["r2_pose"], json.dumps(genoma), str(carpeta)))
        filas = self.sql("SELECT id, carpeta FROM campeonas ORDER BY id DESC")
        for i, c in filas[MAX_CAMPEONAS:]:
            for p in Path(c).glob("*"):
                p.unlink(missing_ok=True)
            Path(c).rmdir() if Path(c).exists() else None
            self.sql("UPDATE campeonas SET carpeta = '' WHERE id = ?", (i,))

    def resumen(self):
        f = self.sql("SELECT COUNT(*), COALESCE(SUM(ventanas),0), COALESCE(SUM(personas),0), COALESCE(SUM(sesiones),0), "
                     "COALESCE(SUM(saturadas),0) FROM lotes", uno=True)
        en_disco = sum(p.stat().st_size for p in (self.raiz / "lotes").glob("*.h5")) / 1e9
        ultimos = self.sql("SELECT id, t, archivo, ventanas, personas, sesiones, saturadas, mb FROM lotes "
                           "ORDER BY id DESC LIMIT 12")
        return {"lotes": f[0], "ventanas": f[1], "personas": f[2], "sesiones": f[3], "saturadas": f[4],
                "gb_en_disco": round(en_disco, 3), "limite_gb": LIMITE_LOTES_GB, "carpeta": str(self.raiz),
                "ultimos": [dict(zip(("id", "t", "archivo", "ventanas", "personas", "sesiones", "saturadas", "mb"), r))
                            for r in ultimos]}
