"""
NEAT: neuroevolución que hace crecer la red
===========================================

Stanley, K. O. & Miikkulainen, R. (2002). Evolving Neural Networks through
Augmenting Topologies. Evolutionary Computation 10(2), 99-127.

Nadie entrena las redes con gradiente ni con datos etiquetados: nacen casi
vacías (cada salida conectada a la neurona de sesgo y a 4 entradas al azar)
y la evolución hace el resto:

  mutar      cambia pesos, agrega una conexión o parte una conexión en dos
             con una neurona nueva en el medio: la red CRECE
  cruzar     dos padres se combinan alineando sus genes por número de
             innovación (el "historial" de cada conexión)
  especies   las redes se agrupan por parecido; cada especie compite sobre
             todo dentro de sí misma, así una idea nueva (todavía mala)
             tiene tiempo de madurar antes de enfrentarse a las campeonas
  selección  las especies que más aguantan en la pista tienen más hijos;
             las que llevan muchas generaciones sin mejorar se extinguen

Redes hacia adelante (sin ciclos). Nodos: 0-23 entradas, 24 sesgo,
25-31 salidas (los 7 gestos), 32 en adelante neuronas ocultas (tanh).
"""

import copy

import numpy as np

import ia_mano as ia

N_ENT, N_SAL = 24, ia.NUM_GESTOS
SESGO = N_ENT
SALIDAS = list(range(N_ENT + 1, N_ENT + 1 + N_SAL))
PRIMER_OCULTO = N_ENT + 1 + N_SAL
NOMBRES_ENTRADAS = [f"{c} t{'' if k == 2 else f'-{2 - k}'}" for k in range(3) for c in ia.NOMBRES_CARAC]

P_PESOS, P_CONEXION, P_NEURONA, P_INTERRUPTOR = 0.8, 0.3, 0.08, 0.02
C_GENES, C_PESOS = 1.0, 0.4
ESPECIES_META, ESTANCADA, ELITE_MIN = 10, 20, 5
ENTRADAS_INICIALES, SIGMA, POR_PESO = 4, 0.5, 0.2    # cada peso muta con probabilidad 0,2 (mutarlos todos a la vez destruye lo aprendido)


class Innovaciones:
    """El mismo cambio estructural recibe el mismo número en toda la población (así se pueden cruzar)."""

    def __init__(self):
        self.conexiones, self.divisiones = {}, {}
        self.siguiente, self.siguiente_nodo = 0, PRIMER_OCULTO

    def conexion(self, a, b):
        if (a, b) not in self.conexiones:
            self.conexiones[(a, b)] = self.siguiente
            self.siguiente += 1
        return self.conexiones[(a, b)]

    def division(self, innov):
        if innov not in self.divisiones:
            self.divisiones[innov] = self.siguiente_nodo
            self.siguiente_nodo += 1
        return self.divisiones[innov]


class Genoma:
    def __init__(self, gid):
        self.id = gid
        self.conexiones = {}            # innovación -> [origen, destino, peso, activa]
        self.ocultas = set()
        self.fitness, self.especie, self.padres = 0.0, None, ()
        self._plan = None

    @classmethod
    def minimo(cls, gid, rng, innov):
        g = cls(gid)
        for s in SALIDAS:
            for e in [SESGO, *rng.choice(N_ENT, ENTRADAS_INICIALES, replace=False)]:
                g.conexiones[innov.conexion(int(e), s)] = [int(e), s, float(rng.normal(0, 1)), True]
        return g

    def hijo(self, gid):
        h = Genoma(gid)
        h.conexiones = copy.deepcopy(self.conexiones)
        h.ocultas = set(self.ocultas)
        h.padres = (self.id,)
        return h

    # ---------- tamaño ----------
    @property
    def n_conexiones(self):
        return sum(c[3] for c in self.conexiones.values())

    # ---------- evaluar (vectorizado en el tiempo) ----------
    def plan(self):
        """Orden topológico de las neuronas y sus entradas activas."""
        if self._plan is None:
            entra = {}
            for a, b, w, act in self.conexiones.values():
                if act:
                    entra.setdefault(b, []).append((a, w))
            hechos, orden, pendientes = set(range(N_ENT + 1)), [], set(self.ocultas) | set(SALIDAS)
            while pendientes:
                listos = [n for n in pendientes if all(a in hechos or a not in pendientes for a, _ in entra.get(n, []))]
                listos = listos or [min(pendientes)]
                for n in sorted(listos):
                    orden.append((n, entra.get(n, [])))
                    hechos.add(n)
                    pendientes.discard(n)
            self._plan = orden
        return self._plan

    def activar(self, X, todas=False):
        """X (T, 24) -> logits (T, 7); con todas=True también el valor de cada neurona."""
        v = {i: X[:, i] for i in range(N_ENT)}
        v[SESGO] = np.ones(len(X), np.float32)
        cero = np.zeros(len(X), np.float32)
        for n, entradas in self.plan():
            s = cero
            for a, w in entradas:
                s = s + w * v.get(a, cero)
            v[n] = np.tanh(s) if n >= PRIMER_OCULTO else s
        y = np.stack([v[s] for s in SALIDAS], 1)
        return (y, v) if todas else y

    # ---------- mutar ----------
    def _alcanza(self, desde, hasta):
        sig = {}
        for a, b, _, act in self.conexiones.values():
            sig.setdefault(a, []).append(b)
        pila, vistos = [desde], set()
        while pila:
            n = pila.pop()
            if n == hasta:
                return True
            if n not in vistos:
                vistos.add(n)
                pila += sig.get(n, [])
        return False

    def mutar(self, rng, innov):
        if rng.random() < P_PESOS:
            for c in self.conexiones.values():
                if rng.random() > POR_PESO:
                    continue
                if rng.random() < 0.1:
                    c[2] = float(rng.normal(0, 1))
                else:
                    c[2] = float(np.clip(c[2] + rng.normal(0, SIGMA), -8, 8))
        if rng.random() < P_CONEXION:
            origenes = list(range(N_ENT + 1)) + sorted(self.ocultas)
            destinos = sorted(self.ocultas) + SALIDAS
            for _ in range(20):
                a, b = int(rng.choice(origenes)), int(rng.choice(destinos))
                if a == b or self._alcanza(b, a):
                    continue
                i = innov.conexion(a, b)
                if i in self.conexiones:
                    if not self.conexiones[i][3]:
                        self.conexiones[i][3] = True
                        break
                    continue
                self.conexiones[i] = [a, b, float(rng.normal(0, 1)), True]
                break
        if rng.random() < P_NEURONA:
            activas = [i for i, c in self.conexiones.items() if c[3]]
            if activas:
                i = int(rng.choice(activas))
                a, b, w, _ = self.conexiones[i]
                n = innov.division(i)
                if n not in self.ocultas:
                    self.conexiones[i][3] = False
                    self.ocultas.add(n)
                    self.conexiones[innov.conexion(a, n)] = [a, n, 1.0, True]
                    self.conexiones[innov.conexion(n, b)] = [n, b, w, True]
        if rng.random() < P_INTERRUPTOR and self.conexiones:
            c = self.conexiones[int(rng.choice(list(self.conexiones)))]
            c[3] = not c[3]
        self._plan = None
        return self


def cruzar(gid, a, b, rng):
    """a es el padre con más fitness: la estructura sale de él; los genes compartidos se sortean."""
    h = Genoma(gid)
    for i, c in a.conexiones.items():
        gen = list(b.conexiones[i] if i in b.conexiones and rng.random() < 0.5 else c)
        if i in b.conexiones and not (c[3] and b.conexiones[i][3]):
            gen[3] = rng.random() > 0.75                          # si un padre la tenía apagada, suele quedar apagada
        h.conexiones[i] = gen
    h.ocultas = set(a.ocultas)
    h.padres = (a.id, b.id)
    return h


def distancia(a, b):
    ia_, ib = set(a.conexiones), set(b.conexiones)
    comunes = ia_ & ib
    distintos = len(ia_ ^ ib)
    n = max(len(ia_), len(ib))
    n = n if n > 20 else 1
    w = np.mean([abs(a.conexiones[i][2] - b.conexiones[i][2]) for i in comunes]) if comunes else 0.0
    return C_GENES * distintos / n + C_PESOS * w


class Poblacion:
    def __init__(self, n=150, semilla=None):
        self.n = n
        self.rng = np.random.default_rng(semilla)
        self.innov = Innovaciones()
        self.siguiente_id = 0
        self.genomas = [Genoma.minimo(self._id(), self.rng, self.innov) for _ in range(n)]
        self.especies = {}              # id -> {"rep", "miembros", "mejor", "sin_mejora", "nacio"}
        self.siguiente_especie, self.umbral = 1, 3.0
        self.gen = 0
        self.historial = []             # una fila por generación (para las curvas)
        self.eventos = []
        self.campeon, self.campeon_metricas, self.n_campeones = None, None, 0
        self.especiar()

    def _id(self):
        self.siguiente_id += 1
        return self.siguiente_id

    def anotar(self, texto):
        self.eventos = (self.eventos + [{"gen": self.gen, "texto": texto}])[-150:]

    def especiar(self):
        for e in self.especies.values():
            e["miembros"] = []
        for g in self.genomas:
            for sid, e in self.especies.items():
                if distancia(g, e["rep"]) < self.umbral:
                    e["miembros"].append(g)
                    g.especie = sid
                    break
            else:
                sid = self.siguiente_especie
                self.siguiente_especie += 1
                self.especies[sid] = {"rep": g, "miembros": [g], "mejor": 0.0, "sin_mejora": 0, "nacio": self.gen}
                g.especie = sid
        vacias = [sid for sid, e in self.especies.items() if not e["miembros"]]
        for sid in vacias:
            del self.especies[sid]
        n = len(self.especies)                                     # el umbral se ajusta para tener ~10 especies
        self.umbral = float(np.clip(self.umbral + (0.1 if n > ESPECIES_META else -0.1 if n < ESPECIES_META else 0), 0.3, 8))

    def reproducir(self):
        """Con los fitness ya puestos: la siguiente generación."""
        rng = self.rng
        for sid, e in self.especies.items():
            e["miembros"].sort(key=lambda g: -g.fitness)
            mejor = e["miembros"][0].fitness
            if mejor > e["mejor"] + 1e-6:
                e["mejor"], e["sin_mejora"] = mejor, 0
            else:
                e["sin_mejora"] += 1
        ranking = sorted(self.especies, key=lambda s: -self.especies[s]["mejor"])
        for sid in list(self.especies):
            if self.especies[sid]["sin_mejora"] > ESTANCADA and sid not in ranking[:2] and len(self.especies) > 2:
                self.anotar(f"Se extingue la especie {sid}: {ESTANCADA} generaciones sin mejorar")
                del self.especies[sid]
        medias = {sid: np.mean([g.fitness for g in e["miembros"]]) for sid, e in self.especies.items()}
        total = sum(medias.values()) or 1.0
        cupos = {sid: m / total * self.n for sid, m in medias.items()}
        enteros = {sid: int(c) for sid, c in cupos.items()}
        for sid in sorted(cupos, key=lambda s: -(cupos[s] - enteros[s]))[:self.n - sum(enteros.values())]:
            enteros[sid] += 1
        nuevos = []
        for sid, cupo in enteros.items():
            m = self.especies[sid]["miembros"]
            if cupo <= 0:
                continue
            if len(m) >= ELITE_MIN or sid == ranking[0]:
                elite = m[0].hijo(self._id())                      # la mejor de la especie pasa intacta
                elite.padres = (m[0].id,)
                nuevos.append(elite)
                cupo -= 1
            padres = m[:max(1, int(np.ceil(len(m) * 0.3)))]
            for _ in range(cupo):
                a = padres[int(rng.integers(len(padres)))]
                if len(padres) > 1 and rng.random() < 0.75:
                    b = padres[int(rng.integers(len(padres)))]
                    a, b = (a, b) if a.fitness >= b.fitness else (b, a)
                    h = cruzar(self._id(), a, b, rng)
                else:
                    h = a.hijo(self._id())
                nuevos.append(h.mutar(rng, self.innov))
        for e in self.especies.values():                           # representante: un miembro al azar de la generación vieja
            e["rep"] = e["miembros"][int(rng.integers(len(e["miembros"])))]
        self.genomas = nuevos[:self.n]
        self.gen += 1
        self.especiar()


# ------------------------------------------------------------------
# Para el ESP32: la red evolucionada como código C, neurona por neurona
# ------------------------------------------------------------------
def exportar_c(g, ruta, comentario=""):
    def num(w):
        return f"{w:.6f}f"

    lineas = ["// Red NEAT evolucionada por el gemelo de la mano (neuroevolución, sin gradiente).",
              f"// {comentario}" if comentario else "//",
              "// Entradas: 8 características calibradas × 3 ventanas (las mismas 24 de modelo_red_calibrado.h).",
              "// Salida: 7 puntajes (reposo, abierta, puño, pinza, señalar, flexión, extensión); gana el mayor.",
              "#pragma once", "#include <math.h>", "",
              f"#define NEAT_ENTRADAS {N_ENT}", f"#define NEAT_SALIDAS {N_SAL}", "",
              "static inline void neat_predecir(const float *x, float *salida) {"]
    nombre = {i: f"x[{i}]" for i in range(N_ENT)}
    nombre[SESGO] = "1.0f"
    for n, entradas in g.plan():
        suma = " + ".join(f"{num(w)} * {nombre[a]}" for a, w in entradas if a in nombre) or "0.0f"
        if n >= PRIMER_OCULTO:
            nombre[n] = f"n{n}"
            lineas.append(f"    const float n{n} = tanhf({suma});")
        else:
            lineas.append(f"    salida[{SALIDAS.index(n)}] = {suma};")
    lineas.append("}")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta
