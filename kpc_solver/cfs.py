r"""
CFS – Combinatorial Branch-and-Bound for the Knapsack Problem with Conflicts.

Follows Sections 4, 6.1, and 6.2 of Coniglio et al. (2021).

§6.1 Preprocessing (run before B&B):
  1. Greedy heuristic: items in nonincreasing p/w order, run n times forcing
     each item once; keep best feasible solution found → initial LB.
  2. Diving heuristic: iteratively solve LP relaxation of ILP2, fix the first
     fractional item (if feasible add it, otherwise exclude it), until no
     fractional items remain → candidate feasible solution.
  3. LB = best of the two heuristics.
  4. Compute KP lookup-table and MCKP lookup-table.
  5. Pegging:
       (22) eliminate i  if  LB >= UB( N̄(i) , c - wi ) + pi
                   where N̄(i) = V \ {i} \ conflict_neighbors(i)
       (23) force   i  if  LB >= UB( V \ {i} , c )
     Repeat until no change.  Forced items go into I_root; their
     conflict-neighbors are removed from V.

§6.2 B&B node pruning order:
  1. UBL2  (MCKP lookup-table)
  2. UB_MT (Martello-Toth)
  3. UB_P  (partition-based, after running PARTITION without budget check)
  If all fail → PARTITION with budget check → branching set B.
"""

import time
from ortools.linear_solver import pywraplp
from .partition import partition_with_budget, partition_no_budget
from .bounds import (KPLookupTable, compute_ub_mt,
                     MCKPLookupTable, check_mckp_closed_form, compute_ub_p)


# ---------------------------------------------------------------------------
# Module-level greedy heuristic (also used standalone before the class exists)
# ---------------------------------------------------------------------------

def _greedy_lb(instance, forced_item=None):
    """
    Greedy heuristic: scan items in nonincreasing p/w order (index 1..n),
    adding each if it fits and has no conflict with already-chosen items.

    If forced_item is given (Section 6.1 – greedy with forcing), that item
    is pre-inserted before the greedy scan.

    Returns (solution_set_of_sorted_indices, total_profit).
    """
    sol    = set()
    w_used = 0
    p_total = 0

    if forced_item is not None:
        if instance.weights[forced_item] <= instance.c:
            sol.add(forced_item)
            w_used  += instance.weights[forced_item]
            p_total += instance.profits[forced_item]

    for i in range(1, instance.n + 1):
        if i == forced_item:
            continue
        if instance.weights[i] <= instance.c - w_used:
            if not (sol & instance.neighbors[i]):   # no conflict
                sol.add(i)
                w_used  += instance.weights[i]
                p_total += instance.profits[i]

    return sol, p_total


# ---------------------------------------------------------------------------
# CFS Solver
# ---------------------------------------------------------------------------

class CFSSolver:
    """
    Branch-and-Bound solver implementing Sections 4, 6.1, and 6.2 of
    Coniglio et al. (2021).

    Usage:
        solver = CFSSolver(instance)
        result = solver.solve()
    """

    def __init__(self, instance, time_limit_sec, external_lb=0, external_sol=None):
        self.instance        = instance
        self.time_limit_sec  = time_limit_sec
        self.nodes           = 0
        self.timeout         = False
        self._start          = time.time()  # avviato qui: usato da _timed_out per il timeout

        # ── Step 1: plain greedy to get a baseline LB ─────────────────────
        self.best_sol, self.LB = _greedy_lb(instance)

        # ── Step 2: precompute lookup tables (needed for pegging too) ──────
        self.lookup_table      = KPLookupTable(instance)
        self.mckp_lookup_table = MCKPLookupTable(instance)

        # ── Step 3: Section 6.1 heuristics ────────────────────────────────

        # Heuristic 1 – greedy with forcing (n runs)
        if not self._timed_out():
            sol_g, p_g = self._greedy_heuristic_with_forcing()
            if p_g > self.LB:
                self.LB, self.best_sol = p_g, sol_g

        # Heuristic 2 – LP-relaxation diving
        if not self._timed_out():
            sol_d, p_d = self._diving_heuristic()
            if p_d > self.LB:
                self.LB, self.best_sol = p_d, sol_d

        # ── Step 4: Section 6.1 pegging ───────────────────────────────────
        self.I_root, self.V_active = self._run_pegging()

    # ── Helper timeout ─────────────────────────────────────────────────────

    def _timed_out(self):
        """Restituisce True se il budget di tempo è esaurito."""
        if time.time() - self._start > self.time_limit_sec:
            self.timeout = True
        return self.timeout

    # ── Heuristic 1 ───────────────────────────────────────────────────────

    def _greedy_heuristic_with_forcing(self):
        """Run _greedy_lb n times, each time forcing a different item."""
        best_sol, best_p = set(), 0
        for k in range(1, self.instance.n + 1):
            if self._timed_out():
                break
            sol, p = _greedy_lb(self.instance, forced_item=k)
            if p > best_p:
                best_p, best_sol = p, sol
        return best_sol, best_p

    # ── Heuristic 2 ───────────────────────────────────────────────────────

    def _diving_heuristic(self):
        """
        Diving heuristic guided by the LP relaxation of ILP2 (§6.1).

        At each step:
          - Solve LP with current fixings.
          - Find first (lowest sorted index = highest p/w) fractional item j.
          - If j fits and doesn't conflict with already-fixed items → fix x_j=1,
            exclude all neighbours of j.
          - Else → fix x_j=0.
          - Stop when LP solution is integer.
        """
        n, weights, profits = self.instance.n, self.instance.weights, self.instance.profits # definisce le variabili di ILP2, i coefficienti, i vincoli
        neighbors, c = self.instance.neighbors, self.instance.c

        # fixed_1: oggetti che l'euristica ha deciso di includere definitivamente (x_i = 1)
        # fixed_0: oggetti che l'euristica ha deciso di escludere definitivamente (x_i = 0)
        fixed_1, fixed_0 = set(), set()
        sol = set()

        # Il ciclo itera al massimo n+1 volte: una per ogni item da "immergere" nel relax LP.
        for _ in range(n + 1): 
            if self._timed_out(): 
                break
            lp = pywraplp.Solver.CreateSolver('GLOP') # il solutore LP (senza vincoli di interezza)
            if not lp:
                break
            
            # IMPOSTA IL TIMEOUT RESIDUO IN MILLISECONDI PER GLOP
            time_left_ms = int(max(0, self.time_limit_sec - (time.time() - self._start)) * 1000)
            if time_left_ms <= 0:
                self.timeout = True
                break
            lp.SetTimeLimit(time_left_ms)

            x = {}
            for i in range(1, n + 1):
                if   i in fixed_1: x[i] = lp.NumVar(1.0, 1.0, f'x{i}') #se l'item è in fixed_1, viene fissato a 1 (incluso)
                elif i in fixed_0: x[i] = lp.NumVar(0.0, 0.0, f'x{i}') #se l'item è in fixed_0, viene fissato a 0 (escluso)
                else:              x[i] = lp.NumVar(0.0, 1.0, f'x{i}') #altrimenti è libero (0,1)

            obj = lp.Objective() # massimizzazione del profitto totale
            for i in range(1, n + 1):
                obj.SetCoefficient(x[i], float(profits[i])) # Coefficiente di profitto
            obj.SetMaximization() # indica che vogliamo massimizzare

            cap = lp.Constraint(-lp.infinity(), float(c)) # Vincolo di capacità
            for i in range(1, n + 1):
                cap.SetCoefficient(x[i], float(weights[i])) # Coefficiente di peso

            for i in range(1, n + 1): # Vincoli di vicinato
                Ni = neighbors[i] # vicini dell'oggetto i
                if Ni:
                    ct = lp.Constraint(-lp.infinity(), float(len(Ni))) # Vincolo di vicinato
                    ct.SetCoefficient(x[i], float(len(Ni))) # Coefficiente di vicinato per x[i]
                    for j in Ni:
                        ct.SetCoefficient(x[j], 1.0) # Coefficiente di vicinato per x[j]

            if lp.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE): # se il solutore non trova una soluzione ottima o ammissibile
                break

            frac = [i for i in range(1, n + 1) # lista di item con soluzione frazionaria
                    if 1e-5 < x[i].solution_value() < 1.0 - 1e-5]

            if not frac: # se non ci sono item con soluzione frazionaria
                sol = {i for i in range(1, n + 1) if x[i].solution_value() > 0.5} # lista di item con soluzione intera
                break

            j = min(frac)   # first fractional item (highest p/w ratio)
            w_used     = sum(weights[i] for i in fixed_1) # peso totale degli item inclusi
            fits       = (w_used + weights[j] <= c) # se l'item j entra nel knapsack
            no_conflict = all(nb not in fixed_1 for nb in neighbors[j]) # se l'item j non ha conflitti con gli item inclusi

            if fits and no_conflict: # se l'item j entra nel knapsack e non ha conflitti con gli item inclusi
                fixed_1.add(j) # aggiunge j a fixed_1
                fixed_0.update(neighbors[j]) # aggiunge i vicini di j a fixed_0
            else:
                fixed_0.add(j) # altrimenti aggiunge j a fixed_0

        return sol, sum(profits[i] for i in sol) # restituisce la soluzione intera e il profitto totale

    # ── Pegging ───────────────────────────────────────────────────────────

    def _tightest_ub(self, V_sub, c_sub):
        """
        Return the tightest upper bound over V_sub with capacity c_sub,
        using all available bounds (UBL1, UB_MT, UBL2, UB_P).
        """
        if not V_sub or c_sub <= 0:
            return 0

        # Se siamo già in timeout, restituisci un bound lasco sicuro (infinito/molto alto) 
        # per evitare che il pegging prenda decisioni errate o perda tempo
        if self._timed_out():
            return sum(self.instance.profits)

        ub = self.lookup_table.get_ub_l1(V_sub, c_sub)
        if self._timed_out(): return ub
        ub = min(ub, compute_ub_mt(V_sub, c_sub, self.instance))
        if self._timed_out(): return ub
        ub = min(ub, self.mckp_lookup_table.get_ub_l2(V_sub, c_sub))
        if self._timed_out(): return ub

        cliques = partition_no_budget(V_sub, self.instance.neighbors)
        if self._timed_out(): return ub
        val_cf  = check_mckp_closed_form(cliques, c_sub, self.instance)
        ub_p    = val_cf if val_cf is not None \
                  else compute_ub_p(V_sub, c_sub, cliques, self.instance)
        ub = min(ub, int(ub_p))

        return ub

    def _run_pegging(self):
        """
        Apply pegging conditions (22) and (23) iteratively (§6.1).

        N̄(i) = anti-neighborhood of i in the conflict graph
              = V_active \\ {i} \\ conflict_neighbors(i)
              = items compatible with i (can coexist with i in a solution)

        Condition (22) – eliminate i:
            LB - p_root  >=  UB( N̄(i) ∩ V_active , c_rem - wi ) + pi
            Meaning: even taking i + the best compatible items gives ≤ LB.

        Condition (23) – force i:
            LB - p_root  >=  UB( V_active \\ {i} , c_rem )
            Meaning: any solution (extending I_root) that omits i gives ≤ LB.

        Returns (I_root, V_active).
        """
        V_active = set(range(1, self.instance.n + 1)) # Insieme degli oggetti del problema (all' inizio del preprocessing, ogni oggetto è potenzialmente "incerto" e attivo)
        I_root   = set() # Insieme di item fissati a 1 (quelli che l'algoritmo capisce essere obbligatori per la soluzione ottima)
        c_rem    = self.instance.c # Capacità residua (all' inizio uguale alla capacità totale)
        p_root   = 0     # profitto già presente in I_root (all' inizio 0)

        while True: # finchè ci sono modifiche
            if self._timed_out():  # interrompe il pegging se il budget è esaurito
                break
            changed = False # flag per indicare se ci sono state modifiche in questa iterazione
            for i in sorted(V_active): # scorre gli oggetti ancora attivi
                if self._timed_out():  # interrompe il pegging se il budget è esaurito
                    break    
                # Trivial: too heavy for remaining capacity
                if self.instance.weights[i] > c_rem: # se l'oggetto non entra nel knapsack
                    V_active.discard(i) # lo rimuove
                    changed = True
                    break # ricomincia il ciclo interno con il set aggiornato

                wi, pi = self.instance.weights[i], self.instance.profits[i] # peso e profitto dell'oggetto i

                # Condition (22): eliminate i
                # N̄(i) ∩ V_active = items in V_active compatible with i
                anti_N_i = V_active - {i} - self.instance.neighbors[i] # Insieme degli oggetti compatibili con i (escluso i)
                ub22 = self._tightest_ub(sorted(anti_N_i), c_rem - wi) # Upper bound del problema ridotto
                if self.LB - p_root >= ub22 + pi: # se anche prendendo i + tutti i compatibili non si raggiunge LB
                    V_active.discard(i)
                    changed = True
                    break

                # Condition (23): force i into the root solution
                rest = V_active - {i} # Crea un insieme temporaneo contenente tutti gli oggetti attivi tranne l'oggetto i.
                ub23 = self._tightest_ub(sorted(rest), c_rem) # Calcola l'upper bound per il sotto-problema ottenuto eliminando l'oggetto i dal set attivo.
                if self.LB - p_root >= ub23: # Se l'upper bound del sotto-problema (senza i) non è abbastanza alto da migliorare il limite attuale
                    I_root.add(i) # Inserisce forzatamente l'oggetto i nella soluzione radice (I_root).
                    V_active.discard(i) # Rimuove l'oggetto i dal set degli oggetti attivi.
                    V_active -= self.instance.neighbors[i]   # neighbours can't coexist
                    # Aggiorna lo stato della radice. Lo zaino si restringe del peso di i, e il profitto sicuro aumenta del valore di i
                    c_rem  -= wi
                    p_root += pi
                    changed = True # Segnala il successo e riavvia il ciclo per sfruttare questo massiccio sfoltimento del grafo
                    break

            if not changed:
                break # Se non ci sono state modifiche, il pre-processing è completo

        return I_root, V_active

    # ── Main entry-point ──────────────────────────────────────────────────

    def solve(self):

        # B&B starts from the pegged root: I_root items already fixed,
        # search continues over V_active with residual capacity.
        I_hat   = self.I_root
        V_hat   = sorted(self.V_active)
        p_I_hat = sum(self.instance.profits[v] for v in I_hat)
        w_I_hat = sum(self.instance.weights[v] for v in I_hat)

        self._search(I_hat, V_hat, p_I_hat, w_I_hat)

        elapsed  = time.time() - self._start
        sol_orig = {self.instance.sorted_to_orig[v] for v in self.best_sol}
        return {
            "optimal_value":  self.LB,
            "solution":       sorted(sol_orig),
            "nodes_explored": self.nodes,
            #"elapsed_time":   elapsed,
            "timeout":        self.timeout,
        }

    # ── Recursive B&B search (§6.2 node pruning order) ────────────────────

    def _search(self, I_hat, V_hat, p_I_hat, w_I_hat):
        
        if self._timed_out():
            return

        self.nodes += 1

        # Update incumbent
        if p_I_hat > self.LB:
            self.LB       = p_I_hat
            self.best_sol = set(I_hat)

        # Filter by residual capacity
        c_hat = self.instance.c - w_I_hat
        V_hat = [v for v in V_hat if self.instance.weights[v] <= c_hat]
        if not V_hat:
            return

        # §6.2 – Step 1: UBL2 (MCKP lookup-table, fastest conflict-aware bound)
        ub_l2 = self.mckp_lookup_table.get_ub_l2(V_hat, c_hat)
        if p_I_hat + ub_l2 <= self.LB:
            return

        if self._timed_out():
            return

        # §6.2 – Step 2: UB_MT (Martello-Toth)
        ub_mt = compute_ub_mt(V_hat, c_hat, self.instance)
        if p_I_hat + ub_mt <= self.LB:
            return

        # 2. Controllo prima del calcolo della clique partition (no-budget)
        if self._timed_out():
            return

        # §6.2 – Step 3: clique partition (no-budget) → UB_P
        cliques = partition_no_budget(V_hat, self.instance.neighbors)
        if self._timed_out():
            return

        val_cf = check_mckp_closed_form(cliques, c_hat, self.instance)
        if val_cf is not None:
            if p_I_hat + val_cf <= self.LB:
                return
        else:
            if self._timed_out():
                return
            ub_p = compute_ub_p(V_hat, c_hat, cliques, self.instance)
            if p_I_hat + int(ub_p) <= self.LB:
                return

        if self._timed_out(): return

        # §6.2 – All pruning failed: branch using PARTITION (with budget)
        B, _ = partition_with_budget(
            V_hat,
            self.instance.neighbors,
            self.instance.profits,
            self.LB,
            p_I_hat,
        )
        if not B:
            return

        for pos, b in enumerate(B):
            if self._timed_out():
                return

            I_new = I_hat | {b}
            p_new = p_I_hat + self.instance.profits[b]
            w_new = w_I_hat + self.instance.weights[b]

            B_prev = set(B[:pos])
            V_new  = [
                v for v in V_hat
                if v != b
                and v not in B_prev
                and v not in self.instance.neighbors[b]
            ]
            self._search(I_new, V_new, p_new, w_new)
