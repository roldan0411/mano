"""Datos compartidos por los dos documentos: resultados de las pruebas, requerimientos y formato."""

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "pruebas"))

from comun import FIGURAS, MOTOR, REQUERIMIENTOS, RESULTADOS  # noqa: E402

CAPTURAS = RESULTADOS / "capturas"
ORDEN = ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
         "D1", "D2", "D3", "D4", "G1", "G2", "G3", "G4"]
R = {k: json.loads((RESULTADOS / f"{k}.json").read_text(encoding="utf-8")) for k in ORDEN if (RESULTADOS / f"{k}.json").exists()}


def es(texto):
    """Coma decimal (12.5 → 12,5) sin tocar exponentes ni versiones."""
    return re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.]*[a-zA-Z_])", r"\1,\2", str(texto)) if texto is not None else ""


def n(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


def pc(v, d=1):
    return f"{v * 100:.{d}f}".replace(".", ",") + " %"


def estado(k):
    c = R[k]["cumple"]
    return {True: "Cumple", False: "No cumple", None: "Informativa"}[c]


def fig(nombre):
    return FIGURAS / nombre


def ficha_campeona():
    c = sorted((MOTOR / "campeonas").glob("c*"))[-1]
    return json.loads((c / "ficha.json").read_text(encoding="utf-8"))


def ficha_neat():
    c = sorted((MOTOR / "neuro" / "campeones").glob("c*"))[-1]
    return json.loads((c / "ficha.json").read_text(encoding="utf-8"))


def trazabilidad():
    t = {k: [] for k in REQUERIMIENTOS}
    for i, r in R.items():
        for q in r["requerimientos"]:
            t.setdefault(q, []).append(i)
    t["RF-10"] = t.get("RF-10", []) + ["Inspección (capturas)"]
    t["RNF-10"] = t.get("RNF-10", []) + ["Inspección (capturas)"]
    return t
