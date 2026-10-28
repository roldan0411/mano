"""Procesamiento de la señal y modelo de electrodos (pruebas unitarias, G4)."""

import numpy as np

import ia_mano as ia
import simulador as sm
from gemelo import datos_emg as de
from gemelo.motor import gemelo2


def test_diez_electrodos_y_cuarenta_caracteristicas():
    assert ia.NUM_CANALES == 10 and ia.NUM_CARACTERISTICAS == 40
    assert ia.CANALES[:5] == ["F1", "F2", "F3", "F4", "F5"] and ia.CANALES[5:] == ["E1", "E2", "E3", "E4", "E5"]
    assert len(sm.ELECTRODOS_NOMINAL) == 10


def test_caracteristicas_senoidal():
    n = np.arange(ia.VENTANA)
    A, f = 500.0, 50.0
    x = 2048 + A * np.sin(2 * np.pi * f * n / ia.FS + 0.37)
    F = ia.caracteristicas_lote(np.repeat(x[:, None], 10, 1)[None])[0]
    assert abs(F[0] / (2 * A / np.pi) - 1) < 0.01
    assert abs(F[1] / (A / np.sqrt(2)) - 1) < 0.01
    assert abs(F[2] - 2 * f * ia.VENTANA / ia.FS) <= 1
    assert np.allclose(F[:4], F[36:40])                      # los 10 canales iguales dan lo mismo


def test_vectorizada_igual_a_referencia():
    rng = np.random.default_rng(0)
    V = np.round(2048 + rng.normal(0, 200, (50, 200, 10)))
    L = ia.caracteristicas_lote(V)
    R = np.array([ia.caracteristicas_ventana(*[V[i, :, c] for c in range(10)]) for i in range(50)])
    assert np.allclose(L, R, rtol=1e-12, atol=1e-9)


def test_continua_no_cambia_nada():
    rng = np.random.default_rng(1)
    V = np.round(2048 + rng.normal(0, 100, (20, 200, 10)))
    assert np.allclose(ia.caracteristicas_lote(V), ia.caracteristicas_lote(V + 500))


def test_contexto_repite_al_empezar_y_ordena_del_mas_viejo():
    Fc = np.arange(5, dtype=np.float32)[:, None] * np.ones((1, 40), np.float32)
    X = gemelo2.contexto(Fc, np.arange(5, dtype=np.int8))
    assert X.shape == (5, 120)
    assert X[0, 0] == X[0, 40] == X[0, 80] == 0             # primera ventana: se repite
    assert (X[1, 0], X[1, 40], X[1, 80]) == (0, 0, 1)
    assert (X[4, 0], X[4, 40], X[4, 80]) == (2, 3, 4)       # t-2, t-1, t


def test_calibracion_relativa():
    F = np.abs(np.random.default_rng(2).normal(10, 2, (30, 40)))
    cal = gemelo2.calibracion(F)
    assert np.allclose(gemelo2.calibrar(F, cal).mean(0), 0, atol=1e-5)


def test_acople_normalizado_y_selectivo():
    W = sm.acople(sm.ELECTRODOS_NOMINAL)
    assert W.shape == (10, 10)
    assert np.allclose(W.max(1), 1)
    assert all((np.argmax(W[c]) < 5) == (c < 5) for c in range(10))   # cada electrodo, su grupo
    W2 = sm.acople(sm.ELECTRODOS_NOMINAL, np.full(10, 3.0))
    assert np.all(W2 < W + 1e-12)                                     # corrido a lo largo del brazo capta menos


def test_cadena_de_adquisicion_da_cuentas_de_12_bits():
    rng = np.random.default_rng(3)
    cond = de.condiciones_persona(1, rng)
    p = gemelo2.Persona(cond, rng)
    ses = p.colocar(rng)
    cuentas, sat = gemelo2.sensor_por_etapas(de._fuentes(p, "puno", 1.0, 0.0, rng), ses, cond, gemelo2.Amplificador(), rng, np.zeros(10))
    assert cuentas.shape[1] == 10 and cuentas.min() >= 0 and cuentas.max() <= 4095 and not sat
