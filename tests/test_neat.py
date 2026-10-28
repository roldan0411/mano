"""Neuroevolución NEAT y la pista (pruebas unitarias, G4)."""

import re

import numpy as np

from gemelo.neuro import neat, pista


def _sin_ciclos(g):
    sig = {}
    for a, b, _, _ in g.conexiones.values():
        sig.setdefault(a, []).append(b)
    estado = {}

    def visitar(n):
        if estado.get(n) == 1:
            return False
        if estado.get(n) == 2:
            return True
        estado[n] = 1
        ok = all(visitar(m) for m in sig.get(n, []))
        estado[n] = 2
        return ok

    return all(visitar(n) for n in list(sig))


def test_las_redes_crecen_sin_ciclos():
    rng, innov = np.random.default_rng(0), neat.Innovaciones()
    g = neat.Genoma.minimo(1, rng, innov)
    for _ in range(400):
        g.mutar(rng, innov)
    assert _sin_ciclos(g)
    assert len(g.ocultas) > 0
    y = g.activar(np.random.default_rng(1).normal(size=(10, neat.N_ENT)))
    assert y.shape == (10, 7) and np.all(np.isfinite(y))


def test_misma_division_misma_neurona():
    innov = neat.Innovaciones()
    i = innov.conexion(3, neat.SALIDAS[0])
    assert innov.division(i) == innov.division(i)
    assert innov.conexion(3, neat.SALIDAS[0]) == i


def test_el_hijo_hereda_la_estructura_del_mejor_padre():
    rng, innov = np.random.default_rng(2), neat.Innovaciones()
    a, b = neat.Genoma.minimo(1, rng, innov), neat.Genoma.minimo(2, rng, innov)
    for _ in range(50):
        a.mutar(rng, innov)
        b.mutar(rng, innov)
    h = neat.cruzar(3, a, b, rng)
    assert set(h.conexiones) == set(a.conexiones) and h.ocultas == a.ocultas


def test_exportar_c_da_lo_mismo(tmp_path):
    rng, innov = np.random.default_rng(4), neat.Innovaciones()
    g = neat.Genoma.minimo(1, rng, innov)
    for _ in range(150):
        g.mutar(rng, innov)
    codigo = neat.exportar_c(g, tmp_path / "red_neat.h").read_text(encoding="utf-8")
    cuerpo = codigo[codigo.index("{", codigo.index("neat_predecir")) + 1:codigo.rindex("}")]
    cuerpo = re.sub(r"(\d)f\b", r"\1", cuerpo).replace("const float ", "").replace("tanhf", "np.tanh").replace(";", "")
    for x in np.random.default_rng(5).normal(size=(5, neat.N_ENT)):
        salida = [0.0] * 7
        exec("\n".join(l.strip() for l in cuerpo.splitlines() if l.strip()), {"np": np, "x": x, "salida": salida})
        assert np.allclose(salida, g.activar(x[None])[0], atol=1e-4)


def test_poblacion_reproduce_y_especia():
    pob = neat.Poblacion(40, semilla=0)
    for g in pob.genomas:
        g.fitness = float(np.random.default_rng(g.id).random())
    pob.reproducir()
    assert pob.gen == 1 and len(pob.genomas) == 40 and len(pob.especies) >= 1


def test_choca_con_medio_segundo_equivocado():
    p = {"gesto": np.zeros(20, np.int8), "estable": np.ones(20, bool)}
    pred = np.zeros((3, 20), int)
    pred[1, 5:9] = 1                     # 4 ventanas mal: no choca
    pred[2, 5:10] = 1                    # 5 ventanas mal: choca en la 10
    assert list(pista.choques(pred, p)) == [20, 20, 10]


def test_las_transiciones_no_cuentan():
    p = {"gesto": np.zeros(10, np.int8), "estable": np.r_[np.zeros(6, bool), np.ones(4, bool)]}
    pred = np.ones((1, 10), int)
    pred[0, 6:] = 0
    assert pista.choques(pred, p)[0] == 10
