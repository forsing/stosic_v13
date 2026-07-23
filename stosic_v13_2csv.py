from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v13_2csv.py — 7-node krug (K=7 / prilagodjenje 7/39) — Transport structure invariance (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/transport_structure.pvs
  — 4 ground metrike; transport_invariant(P) = ∀k P(optimal_plan(k))
  — thm_vertex_invariant, thm_monge_invariant
  — thm_rank_not_invariant (rank samo spectral)

Mapiranje na 7/39:
  4 cost matrice na {1..39}:
    0 spectral Euclidean: ||x_i-x_j||² (ko-pojavljivanje)
    1 spectral var-weight: Σ_d (x_id-x_jd)²/(v_d+ε)
    2 control Fisher-ish: (√ν_i−√ν_j)²
    3 control Euclidean: (i−j)²
  za A→B: 4× min-cost matching; Π[i,j] += broj metrika koje daju ivicu (i,j)
  skor[j] = Σ_{i∈last} Π(i,j)   (strukturno slagane ivice)
  next = top 7; bez randoma; stop ako uzastopni/AP
"""

from typing import List

import numpy as np

from stosic_v1_2csv import CSV_LOTO, CSV_PLUS, EPS, MAX_NUM, N_PICK, load_draws
from stosic_v2_2csv import top7_from_freq
from stosic_v8_2csv import cooccurrence_features, cost_matrix
from stosic_v9_2csv import optimal_matching_support
from stosic_v10_2csv import is_degenerate


def cost_var_weighted(feats: np.ndarray) -> np.ndarray:
    v = np.var(feats, axis=0) + EPS
    xs = feats / np.sqrt(v)
    return cost_matrix(xs)


def cost_fisher_control(nu: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.clip(nu, EPS, None))
    return (s[:, None] - s[None, :]) ** 2


def cost_label_euclid() -> np.ndarray:
    idx = np.arange(MAX_NUM, dtype=np.float64)
    return (idx[:, None] - idx[None, :]) ** 2


def four_costs(draws: np.ndarray) -> List[np.ndarray]:
    feats = cooccurrence_features(draws)
    nu = np.zeros(MAX_NUM, dtype=np.float64)
    for d in draws:
        for n in d:
            nu[int(n) - 1] += 1.0
    nu = nu / nu.sum()
    return [
        cost_matrix(feats),
        cost_var_weighted(feats),
        cost_fisher_control(nu),
        cost_label_euclid(),
    ]


def accumulate_invariant_support(draws: np.ndarray, costs: List[np.ndarray]) -> np.ndarray:
    Pi = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    for t in range(len(draws) - 1):
        src = [int(n) - 1 for n in draws[t]]
        tgt = [int(n) - 1 for n in draws[t + 1]]
        for C in costs:
            for i, j in optimal_matching_support(src, tgt, C):
                Pi[i, j] += 1.0
    return Pi


def predict_next(draws: np.ndarray) -> List[int]:
    costs = four_costs(draws)
    Pi = accumulate_invariant_support(draws, costs)
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    for n in draws[-1]:
        skor += Pi[int(n) - 1, :]
    combo = top7_from_freq(skor)
    if is_degenerate(combo):
        nu = np.zeros(MAX_NUM, dtype=np.float64)
        for d in draws:
            for n in d:
                nu[int(n) - 1] += 1.0
        combo = top7_from_freq(nu)
    return combo


def main():
    next_loto = predict_next(load_draws(CSV_LOTO))
    next_loto_plus = predict_next(load_draws(CSV_PLUS))
    if is_degenerate(next_loto):
        raise SystemExit("degenerisan next_loto (uzastopni/AP) — zaustavljen pre ispisa")
    if is_degenerate(next_loto_plus):
        raise SystemExit("degenerisan next_loto_plus (uzastopni/AP) — zaustavljen pre ispisa")
    print("next_loto:      ", next_loto)
    print("next_loto_plus: ", next_loto_plus)


if __name__ == "__main__":
    main()



"""
next_loto:       [6, x, 16, y, 27, z, 39]
next_loto_plus:  [3, x, 9, y, 18, z, 21]
"""



"""
v13: transport_structure — 4 metrike, Π = slagane matching ivice.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
