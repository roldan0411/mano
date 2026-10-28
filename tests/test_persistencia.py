"""Persistencia y operación (G1 y G3): lo guardado se recupera igual al reabrir."""

import pickle

import numpy as np
import pytest


def test_g1_mano_de_la_competencia_se_guarda_y_carga_igual(tmp_path):
    from gemelo.motor.poblacion import Mano, genoma_aleatorio
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 120)).astype(np.float32)
    m = Mano(0, genoma_aleatorio(rng), X)
    m.metricas = {"puntaje": 0.5, "acierto": 0.6, "pseudo_r2": 0.4, "r2_pose": 0.3}
    m.pasos = 1234
    m.guardar(tmp_path / "mano_00.keras")
    c = Mano.cargar(0, tmp_path / "mano_00.keras")
    assert c.pasos == 1234 and c.metricas == m.metricas and c.genoma == m.genoma
    a, b = m.predecir(X[:20]), c.predecir(X[:20])
    assert np.allclose(a[0], b[0], atol=1e-6) and np.allclose(a[1], b[1], atol=1e-6)


def test_g1_almacen_no_borra_la_carpeta_al_pasar_el_limite(tmp_path, monkeypatch):
    """Regresión: al pasar el límite de disco intentaba borrar lotes ya borrados (la carpeta) y el motor se caía."""
    from gemelo.motor import almacen
    monkeypatch.setattr(almacen, "LIMITE_LOTES_GB", 0.0)
    alm = almacen.Almacen(tmp_path)
    for i in range(3):
        (alm.raiz / "lotes" / f"l{i}.h5").write_bytes(b"x" * 10)
        alm.sql("INSERT INTO lotes VALUES (?,?,?,?,?,?,?,?)", (i + 1, 0.0, f"l{i}.h5", 1, 1, 1, 0, 1.0))
    alm._retener()
    alm._retener()                                           # la segunda vez ya están borrados
    assert (alm.raiz / "lotes").is_dir()
    assert alm.sql("SELECT COUNT(*) FROM lotes WHERE archivo != ''", uno=True)[0] == 0


def test_g1_mano_bebe_retoma_y_aparta_versiones_viejas(tmp_path, monkeypatch):
    from gemelo.libre import bebe, cuerpo
    monkeypatch.setattr(bebe, "RAIZ", tmp_path)
    b = bebe.Bebe(semilla=1)
    mapa = cuerpo.Mapa()
    for _ in range(400):
        b.intento(mapa, None)
    bebe._guardar(b)
    c, aviso = bebe._cargar()
    assert aviso is None and c.intentos == 400 and len(c.habilidades) == len(b.habilidades)
    viejo = bebe.Bebe(semilla=2)
    viejo.version = 2
    with open(tmp_path / "bebe.pkl", "wb") as f:
        pickle.dump(viejo, f)
    d, aviso = bebe._cargar()
    assert d.intentos == 0 and "Nace de cero" in aviso
    assert list(tmp_path.glob("bebe_respaldo_*.pkl"))


def test_g1_neuroevolucion_se_guarda_y_sigue_igual(tmp_path):
    from gemelo.neuro import neat
    pob = neat.Poblacion(30, semilla=3)
    for g in pob.genomas:
        g.fitness = float(g.id % 7)
    pob.reproducir()
    with open(tmp_path / "p.pkl", "wb") as f:
        pickle.dump(pob, f)
    with open(tmp_path / "p.pkl", "rb") as f:
        q = pickle.load(f)
    X = np.random.default_rng(0).normal(size=(5, neat.N_ENT))
    assert q.gen == pob.gen and all(np.allclose(a.activar(X), b.activar(X)) for a, b in zip(pob.genomas, q.genomas))


@pytest.mark.parametrize("salud, dispositivo, pausa", [
    ({"cargador": True, "ram_libre_gb": 4.0}, "cuda", False),
    ({"cargador": False, "ram_libre_gb": 4.0}, "cuda", True),     # en batería con GPU: se pausa
    ({"cargador": False, "ram_libre_gb": 4.0}, "cpu", False),     # en batería sin GPU: sigue en la CPU
    ({"cargador": True, "ram_libre_gb": 0.8}, "cuda", True),      # poca RAM: se pausa
])
def test_g3_motor_se_adapta_a_bateria_y_ram(salud, dispositivo, pausa):
    from gemelo.motor.servicio import motivo_pausa
    assert (motivo_pausa(salud, dispositivo) is not None) == pausa


def test_g2_sin_procesos_huerfanos():
    """Si el motor muere a la fuerza, sus procesos de simulación se cierran solos."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    import psutil
    raiz = Path(__file__).resolve().parents[1]
    script = raiz / "tests" / "_padre_que_muere.py"
    script.write_text(
        "import os, sys, time\n"
        f"sys.path.insert(0, r'{raiz}')\n"
        "from gemelo import config\n"
        "def pid(_):\n"
        "    time.sleep(0.3)\n"
        "    return os.getpid()\n"
        "if __name__ == '__main__':\n"
        "    p = config.pool_vigilado(2)\n"
        "    print(' '.join(map(str, set(p.map(pid, range(8))))), flush=True)\n"
        "    time.sleep(60)\n", encoding="utf-8")
    try:
        padre = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
        hijos = [int(x) for x in padre.stdout.readline().split()]
        assert hijos and all(psutil.pid_exists(h) for h in hijos)
        padre.kill()                                         # muere a la fuerza, como un cierre abrupto
        fin = time.time() + 10
        while time.time() < fin and any(psutil.pid_exists(h) for h in hijos):
            time.sleep(0.5)
        assert not any(psutil.pid_exists(h) for h in hijos)
    finally:
        script.unlink(missing_ok=True)
