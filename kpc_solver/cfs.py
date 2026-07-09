"""
Section 4 – Combinatorial Branch-and-Bound (CFS).

The CFS algorithm explores a DFS tree.  At each node:
  - I_hat : current partial independent set (items fixed to 1)
  - V_hat : candidate set = anti-neighbourhood of all items in I_hat,
            filtered to items whose weight fits in the residual capacity
  - PARTITION(V_hat) → B (branching set), P (pruned set)
  - Branch on each b in B, creating one child per element:
        I_hat_new = I_hat ∪ {b}
        V_hat_new = V_hat \\ ({b} ∪ B_before_b ∪ N(b))
    where B_before_b = elements of B branched on before b (avoids duplicates).
  - Recursion stops when B = ∅.

Initial lower bound: a greedy feasible solution (items in sorted order, skip conflicts).
"""

import time
from .partition import partition_with_budget, partition_no_budget
from .bounds import KPLookupTable, compute_ub_mt, MCKPLookupTable, check_mckp_closed_form, compute_ub_p











# ---------------------------------------------------------------------------
# Greedy heuristic – used to get an initial lower bound for PARTITION
# ---------------------------------------------------------------------------


def _greedy_lb(instance):
    """
    Simple greedy: add items in sorted order (nonincreasing p/w),
    skipping those that exceed the remaining capacity or conflict with
    already selected items.
    Returns (solution_set, profit).
    """
    sol = set()
    w_used = 0
    p_total = 0
    for i in range(1, instance.n + 1):
        # Verifica che il peso dell'oggetto corrente non superi la capacità residua dello zaino
        if instance.weights[i] <= instance.c - w_used: 
            # Se l'intersezione è vuota, significa che l'oggetto corrente non è in conflitto 
            # con nessun oggetto già messo nello zaino
            if not (sol & instance.neighbors[i]):   # no conflict
                # Inserimento e aggiornamento
                sol.add(i)
                w_used  += instance.weights[i]
                p_total += instance.profits[i]
    return sol, p_total


# ---------------------------------------------------------------------------
# CFS Solver
# ---------------------------------------------------------------------------

class CFSSolver:
    """
    Branch-and-Bound solver implementing Section 4 of Coniglio et al. (2021).

    Usage:
        solver = CFSSolver(instance)
        result = solver.solve()
    """

    def __init__(self, instance, time_limit_sec=600):
        self.instance = instance
        self.time_limit_sec = time_limit_sec

        # Warm-start: greedy lower bound
        # serve per potare alcuni nodi dell' albero. Se un nodo ha un LB <= LB attuale, non lo considero
        self.best_sol, self.LB = _greedy_lb(instance)

        # Precompute KP-based lookup-table (Section 5.1.1)
        self.lookup_table = KPLookupTable(instance)

        # Precompute MCKP-based lookup-table (Section 5.2.1)
        self.mckp_lookup_table = MCKPLookupTable(instance)

        self.nodes = 0 # contatore dei nodi dell'albero
        self.timeout = False # Serve all'algoritmo per capire che deve interrompere immediatamente i calcoli


    def solve(self):
        self._start = time.time() # avvio cronometro

        # Initial candidate set: all items that fit within capacity, sorted order
        # crea l'insieme iniziale dei candidati eleggibili ( escludendo quelli che superano la capacita' massima)
        V_hat = [i for i in range(1, self.instance.n + 1)
                 if self.instance.weights[i] <= self.instance.c]
        '''
            set() -> insieme vuoto (I_hat)
            V_hat -> lista di candidati disponibili 
            0, 0 -> profitto e peso iniziali accumulati fino ad ora
        '''
        self._search(set(), V_hat, 0, 0)

        elapsed = time.time() - self._start
        # travaso ri-traducendo gli oggetti nei loro identificativi originali
        sol_orig = {self.instance.sorted_to_orig[v] for v in self.best_sol}
        return {
            "optimal_value":    self.LB,
            "solution":         sorted(sol_orig),
            "nodes_explored":   self.nodes, # numero totale di nodi dell'albero che l'algoritmo ha dovuto esaminare
            "elapsed_time":     elapsed,
            "timeout":          self.timeout,
        }

    # ------------------------------------------------------------------
    def _search(self, I_hat, V_hat, p_I_hat, w_I_hat):
        # Time-limit check (every 500 nodes)
        self.nodes += 1 # aggiorna contatore nodi visitati 
        if self.nodes % 500 == 0: # ad ogni 500 nodi visitati, controlla se e' stato superato il limite di tempo
            if time.time() - self._start > self.time_limit_sec:
                self.timeout = True
                return

        # Update incumbent
        # se il profitto accumulato fino ad ora e' maggiore del miglior profitto trovato fin'ora, 
        # aggiorna il profitto migliore e la soluzione migliore
        if p_I_hat > self.LB:
            self.LB = p_I_hat
            self.best_sol = set(I_hat)

        # Filter V_hat by residual capacity
        c_hat = self.instance.c - w_I_hat # spazio rimanente nello zaino
        # vaporizza all'istante tutti gli oggetti che da soli pesano più dello spazio rimasto
        V_hat = [v for v in V_hat if self.instance.weights[v] <= c_hat]
        # se la lista è vuota, significa che non ci sono candidati che possono essere aggiunti allo zaino
        # quindi interrompe la ricorsione
        if not V_hat:
            return

        # ==== LIVELLO 1: Filtro a Tempo Costante (Section 5.1.1) ====
        # Algoritmo: Lookup Table KP classica. Costo: O(1).
        # Stima il massimo profitto ignorando tutti i conflitti.
        ub_l1 = self.lookup_table.get_ub_l1(V_hat, c_hat)
        if p_I_hat + ub_l1 <= self.LB:
            return  # Prune

        # ==== LIVELLO 2: Filtro Lineare Continuo (Section 5.1.2) ====
        # Algoritmo: Martello-Toth. Costo: O(|V_hat|).
        # Trova l'oggetto critico e valuta i due scenari con rilassamento continuo.
        ub_mt = compute_ub_mt(V_hat, c_hat, self.instance)
        if p_I_hat + ub_mt <= self.LB:
            return  # Prune

        # ==== LIVELLO 3: Filtro sui Conflitti MCKP-based (Section 5.2) ====
        # Costruisce la partizione in cricche P(V_hat) per i due sotto-filtri.
        cliques_V_hat = partition_no_budget(V_hat, self.instance.neighbors)

        # -- Fase 3a: Scorciatoia in Forma Chiusa --
        # Prende l'oggetto di profitto massimo (peso minimo in caso di parità) di ogni cricca.
        # Se la loro somma di pesi rientra in c_hat → il bound è esatto e disponibile subito.
        val_cf = check_mckp_closed_form(cliques_V_hat, c_hat, self.instance)
        if val_cf is not None:
            if p_I_hat + val_cf <= self.LB:
                return  # Prune con bound esatto della forma chiusa
        else:
            # -- Fase 3b: Lookup Table MCKP (fallback) --
            # La forma chiusa non era ammissibile: usiamo prima UBL2 (precomputed in O(1)).
            ub_l2 = self.mckp_lookup_table.get_ub_l2(V_hat, c_hat)
            if p_I_hat + ub_l2 <= self.LB:
                return  # Prune

            # Se UBL2 fallisce, calcoliamo il bound basato sulla partizione UB_P (Sezione 5.2.2)
            ub_p = compute_ub_p(V_hat, c_hat, cliques_V_hat, self.instance)
            if p_I_hat + int(ub_p) <= self.LB:
                return  # Prune





        # ==== PARTITION (Section 4.1) ====
        # Se tutti i livelli di pruning falliscono, si procede con il branching.
        B, P = partition_with_budget(
            V_hat,
            self.instance.neighbors,
            self.instance.profits,
            self.LB,
            p_I_hat,
        )

        # Se B è vuoto non è possibile migliorare l'incumbent da questo nodo
        if not B:
            return

        # ------ n-ary branching on B -----------------------------------
        # Per ogni oggetto b dentro l'insieme di branching B
        for pos, b in enumerate(B): 
            if self.timeout:
                return

            I_new = I_hat | {b} # Crea un nuovo zaino aggiungendo l'oggetto corrente
            # Aggiorna i contatori
            p_new = p_I_hat + self.instance.profits[b]
            w_new = w_I_hat + self.instance.weights[b]

            # Candidate set for child node:
            #   keep items from V_hat except:
            #     - b itself
            #     - B[0..pos-1]  (already branched → avoid duplicate subtrees)
            #     - neighbours of b  (conflict with b)

            #contiene gli oggetti di B che abbiamo già esaminato nei giri precedenti di questo ciclo for
            B_prev = set(B[:pos])
            
            V_new = [
                v for v in V_hat
                if v != b   # non inserire di nuovo l'oggetto corrente
                and v not in B_prev # non inserire oggetti già esaminati
                and v not in self.instance.neighbors[b] # non inserire oggetti in conflitto
            ]
            # l'algoritmo lancia la chiamata ricorsiva
            self._search(I_new, V_new, p_new, w_new)
