"""
Test suite – confronta CFS (Section 4) con le formulazioni ILP (Section 3)
e con il brute-force esatto su piccole istanze.
"""
from kpc_solver import KPCInstance, solve_ilp1, solve_ilp2, solve_ilp3, CFSSolver


# -----------------------------------------------------------------------
# Brute-force reference (esegue 2^n confronti, usabile solo per n piccolo)
# -----------------------------------------------------------------------
def brute_force(n, c, profits, weights, edges):
    edge_set = {(min(u,v), max(u,v)) for u,v in edges}
    best = 0
    for mask in range(1 << n):
        sol = [i+1 for i in range(n) if mask & (1 << i)]
        if sum(weights[i-1] for i in sol) > c:
            continue
        if any((min(u,v), max(u,v)) in edge_set for u in sol for v in sol if u != v):
            continue
        best = max(best, sum(profits[i-1] for i in sol))
    return best


# -----------------------------------------------------------------------
# Test 1 – esempio del paper (Section 6.3)
# -----------------------------------------------------------------------
def test_paper_example():
    print("--- Esempio del paper (Section 6.3) ---")
    n, c = 7, 8
    profits = [3, 2, 3, 4, 3, 5, 4]
    weights = [1, 1, 2, 3, 3, 6, 5]
    edges   = [(1,2),(2,4),(3,4),(5,6),(6,7)]

    instance = KPCInstance(n, c, profits, weights, edges)

    res_cfs  = CFSSolver(instance).solve()
    res_ilp1 = solve_ilp1(instance)

    print(f"  CFS  → valore={res_cfs['optimal_value']}, "
          f"sol={res_cfs['solution']}, nodi={res_cfs['nodes_explored']}, "
          f"t={res_cfs['elapsed_time']:.5f}s")
    print(f"  ILP1 → valore={res_ilp1['optimal_value']}, "
          f"sol={res_ilp1['solution']}, t={res_ilp1['elapsed_time']:.5f}s")

    assert res_cfs['optimal_value']  == 10, f"CFS: atteso 10, ottenuto {res_cfs['optimal_value']}"
    assert res_ilp1['optimal_value'] == 10
    print("  OK\n")


# -----------------------------------------------------------------------
# Test 2 – 10 istanze casuali (n=15) confronto CFS vs brute-force vs ILP1
# -----------------------------------------------------------------------
def test_random_instances():
    import random
    print("--- Istanze casuali (n=15, c=30, densità=0.3) ---")
    random.seed(42)

    for trial in range(10):
        n, c = 15, 30
        profits = [random.randint(5, 25) for _ in range(n)]
        weights = [random.randint(3, 15) for _ in range(n)]
        edges = [(i, j)
                 for i in range(1, n+1)
                 for j in range(i+1, n+1)
                 if random.random() < 0.3]

        bf   = brute_force(n, c, profits, weights, edges)
        inst = KPCInstance(n, c, profits, weights, edges)
        cfs  = CFSSolver(inst).solve()
        ilp1 = solve_ilp1(inst)

        status = "OK" if cfs['optimal_value'] == bf else "ERRORE"
        print(f"  Trial {trial+1:2d}: BF={bf:3d} | CFS={cfs['optimal_value']:3d} "
              f"(nodi={cfs['nodes_explored']:4d}, t={cfs['elapsed_time']:.4f}s) "
              f"| ILP1={ilp1['optimal_value']:3d}  [{status}]")

        assert cfs['optimal_value'] == bf, \
            f"Trial {trial+1}: CFS={cfs['optimal_value']}, atteso={bf}"

    print("  Tutti i trial superati!\n")


# -----------------------------------------------------------------------
# Test 3 – KP Lookup Table (Section 5.1.1)
# -----------------------------------------------------------------------
def test_kp_lookup_table():
    print("--- Test KP Lookup Table (Section 5.1.1) ---")
    from kpc_solver.bounds import KPLookupTable
    from kpc_solver import KPCInstance

    n, c = 7, 8
    profits = [3, 2, 3, 4, 3, 5, 4]
    weights = [1, 1, 2, 3, 3, 6, 5]
    edges   = [(1,2),(2,4),(3,4),(5,6),(6,7)]

    instance = KPCInstance(n, c, profits, weights, edges)
    lookup = KPLookupTable(instance)

    # Verifica i valori specifici della Tabella 1 del paper:
    # Oggetto 7: peso 5, profitto 4 (indice ordinato 7)
    # s=0..4 deve essere 0, s=5..8 deve essere 4
    for s in range(5):
        assert lookup.table[7][s] == 0, f"Atteso table[7][{s}] == 0, ottenuto {lookup.table[7][s]}"
    for s in range(5, 9):
        assert lookup.table[7][s] == 4, f"Atteso table[7][{s}] == 4, ottenuto {lookup.table[7][s]}"

    # Oggetto 1: (indice ordinato 1)
    # s=8 deve essere 12 (oggetti 1, 2, 3, 4 con peso 1+1+2+3=7 <= 8, profitti 3+2+3+4=12)
    assert lookup.table[1][8] == 12, f"Atteso table[1][8] == 12, ottenuto {lookup.table[1][8]}"
    
    # Oggetto 6: peso 6, profitto 5 (indice ordinato 6)
    # s=8: gli oggetti 6, 7 non possono essere entrambi scelti (peso totale 11 > 8), quindi max(5, 4) = 5
    assert lookup.table[6][8] == 5, f"Atteso table[6][8] == 5, ottenuto {lookup.table[6][8]}"

    print("  I valori della KP Lookup Table sono corretti!\n")


# -----------------------------------------------------------------------
# Test 4 – Martello–Toth Upper Bound (Section 5.1.2)
# -----------------------------------------------------------------------
def test_martello_toth():
    print("--- Test Martello–Toth Upper Bound (Section 5.1.2) ---")
    from kpc_solver.bounds import compute_ub_mt
    from kpc_solver import KPCInstance

    n, c = 7, 8
    profits = [3, 2, 3, 4, 3, 5, 4]
    weights = [1, 1, 2, 3, 3, 6, 5]
    edges   = [(1,2),(2,4),(3,4),(5,6),(6,7)]

    instance = KPCInstance(n, c, profits, weights, edges)
    
    # Per il nodo radice (V_hat = [1, 2, 3, 4, 5, 6, 7], c_hat = 8):
    V_hat = [1, 2, 3, 4, 5, 6, 7]
    ub_mt = compute_ub_mt(V_hat, 8, instance)
    print(f"  UB_MT per nodo radice: {ub_mt}")
    assert ub_mt == 12, f"Atteso UB_MT == 12, ottenuto {ub_mt}"

    # Se c_hat = 4, gli oggetti che ci stanno da soli sono {1, 2, 3, 4, 5}
    # In V_hat = [1, 2, 3, 4, 5], c_hat = 4
    # Trova critico:
    # 1: wt 1, cum_w 1
    # 2: wt 1, cum_w 2
    # 3: wt 2, cum_w 4
    # 4: wt 3, cum_w 7 > 4, quindi critico è 4 (t=4).
    # cum_w = 4, cum_p = 8.
    # c_bar = 4 - 4 = 0.
    # w_t = 3, p_t = 4.
    # ub0 (exclude t): cum_p + c_bar * p_next // w_next = 8 + 0 = 8.
    # ub1 (include t): cum_p + p_t - ceil((w_t - c_bar) * p_prev / w_prev)
    #  p_prev (item 3) = 3, w_prev = 2
    #  loss = ceil(3 * 3 / 2) = ceil(4.5) = 5
    #  ub1 = 8 + 4 - 5 = 7.
    #  max(8, 7) = 8.
    ub_mt_4 = compute_ub_mt([1, 2, 3, 4, 5], 4, instance)
    print(f"  UB_MT per c_hat=4: {ub_mt_4}")
    assert ub_mt_4 == 8, f"Atteso UB_MT == 8, ottenuto {ub_mt_4}"

    print("  Martello-Toth upper bound values are correct!\n")


# -----------------------------------------------------------------------
# Test 5 – MCKP Lookup Table (Section 5.2.1)
# -----------------------------------------------------------------------
def test_mckp_lookup_table():
    print("--- Test MCKP Lookup Table (Section 5.2.1) ---")
    from kpc_solver.bounds import MCKPLookupTable
    from kpc_solver import KPCInstance

    n, c = 7, 8
    profits = [3, 2, 3, 4, 3, 5, 4]
    weights = [1, 1, 2, 3, 3, 6, 5]
    edges   = [(1,2),(2,4),(3,4),(5,6),(6,7)]

    instance = KPCInstance(n, c, profits, weights, edges)
    lookup = MCKPLookupTable(instance)

    # Verifica i valori specifici della Tabella 2 del paper per s=8:
    # Oggetto 7: deve essere 4
    # Oggetto 6: deve essere 5
    # Oggetto 5: deve essere 7
    # Oggetto 4: deve essere 8
    # Oggetto 3: deve essere 8
    # Oggetto 2: deve essere 9
    # Oggetto 1: deve essere 10
    
    expected_s8 = {
        7: 4,
        6: 5,
        5: 7,
        4: 8,
        3: 8,
        2: 9,
        1: 10
    }
    for item, val in expected_s8.items():
        assert lookup.table[item][8] == val, f"Atteso table[{item}][8] == {val}, ottenuto {lookup.table[item][8]}"

    print("  I valori della MCKP Lookup Table sono corretti!\n")


# -----------------------------------------------------------------------
# Test 6 – Partition-based Upper Bound UB_P (Section 5.2.2)
# -----------------------------------------------------------------------
def test_ub_p():
    print("--- Test Partition-based Upper Bound UB_P (Section 5.2.2) ---")
    from kpc_solver.bounds import compute_ub_p
    from kpc_solver.partition import partition_no_budget
    from kpc_solver import KPCInstance

    n, c = 7, 8
    profits = [3, 2, 3, 4, 3, 5, 4]
    weights = [1, 1, 2, 3, 3, 6, 5]
    edges   = [(1,2),(2,4),(3,4),(5,6),(6,7)]

    instance = KPCInstance(n, c, profits, weights, edges)
    # V_hat = [1, 2, 3, 4, 5, 6, 7]
    V_hat = [1, 2, 3, 4, 5, 6, 7]
    cliques = partition_no_budget(V_hat, instance.neighbors)
    
    # Per la partizione: C1={1,2}, C2={3,4}, C3={5,6}, C4={7}
    # Il paper indica che UB_P(V_hat) per c_hat=8 è 65/6 (circa 10.833)
    ub = compute_ub_p(V_hat, 8, cliques, instance)
    assert abs(ub - 65/6) < 1e-6, f"Atteso 65/6 (10.8333), ottenuto {ub}"
    print("  Il calcolo di UB_P è corretto!\n")


if __name__ == "__main__":
    test_kp_lookup_table()
    test_martello_toth()
    test_mckp_lookup_table()
    test_ub_p()
    test_paper_example()
    test_random_instances()




