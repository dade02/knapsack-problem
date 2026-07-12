"""
Greedy + Local Search heuristic for the Knapsack Problem with Conflicts (KPC).

Fase 1 – Greedy costruttivo:
    Ordina gli item per rapporto p/w decrescente (stesso criterio di CFS).
    Aggiunge ogni item se:
      - entra nella capacità residua, e
      - non conflicta con gli item già inseriti.

Fase 2 – Local Search 1-opt (swap):
    Per ogni item x in soluzione e ogni item y fuori dalla soluzione:
      Se si può rimuovere x e inserire y migliorando il profitto totale e
      rispettando capacità e conflitti → effettua lo scambio.
    Ripete finché nessun miglioramento è trovato.

Fase 3 – Multi-start:
    Ripete fasi 1-2 con diverse permutazioni casuali degli item
    (controllate da un seed) e restituisce la soluzione migliore trovata.

La classe espone la stessa interfaccia di CFSSolver:
    result = GreedyLocalSearchSolver(instance).solve()
    result = {
        "optimal_value": ...,   # valore della soluzione (NON garantito ottimo)
        "solution":      [...],   # Lista ordinata di indici degli item inclusi nella soluzione finale
        "elapsed_time":  ...,     # Tempo totale impiegato (secondi)
        "restarts":      ...,   # numero di restart effettuati
    }
"""

import time
import random


class GreedyLocalSearchSolver:
    """
    Euristico Greedy + Local Search per il KPC.

    Parametri
    ---------
    instance      : KPCInstance
    n_restarts    : numero di multi-start con ordinamenti casuali (default 20)
    seed          : seme per la riproducibilità
    time_limit_sec: limite di tempo totale in secondi (default 30)
    """

    def __init__(self, instance, n_restarts: int = 20,
                 seed: int = 42, time_limit_sec: float = 30.0):
        self.instance       = instance
        self.n_restarts     = n_restarts
        self.seed           = seed
        self.time_limit_sec = time_limit_sec

    # ── Fase 1: greedy costruttivo ────────────────────────────────────────

    def _greedy(self, order):
        """
        Dato un ordine degli item (lista di indici 1..n), costruisce
        una soluzione greedy ammissibile.
        Restituisce (set_of_items, total_profit, total_weight).
        """
        inst      = self.instance
        sol       = set()
        w_used    = 0
        p_total   = 0

        for i in order:
            # verifica se l'item i può essere aggiunto alla soluzione corrente:
            #  - peso non supera la capacità residua
            #  - non conflitta con nessun item già presente 
            if inst.weights[i] <= inst.c - w_used and not (sol & inst.neighbors[i]):
                sol.add(i)
                w_used  += inst.weights[i]
                p_total += inst.profits[i]

        return sol, p_total, w_used

    # ── Fase 2: local search 1-opt (swap e insert) ───────────────────────

    def _local_search(self, sol, p_total, w_used):
        """
        Migliora la soluzione corrente tramite mosse:
          - INSERT: aggiunge un item non presente se ammissibile e migliora.
          - SWAP  : rimuove un item e aggiunge un altro se migliora.
        Ripete finché nessuna mossa migliora la soluzione.
        """
        inst     = self.instance
        improved = True  # flag per indicare se la soluzione è stata migliorata

        while improved:  # finché ci sono miglioramenti, continua
            improved = False # riparte sempre da false

            # ── INSERT: prova ad aggiungere item fuori dalla soluzione ──
            outside = [i for i in range(1, inst.n + 1) if i not in sol] # lista di item non presenti nella soluzione
            for y in outside:
                if (inst.weights[y] <= inst.c - w_used and not (sol & inst.neighbors[y])): 
                    # verifica se l'item y può essere aggiunto alla soluzione corrente:
                        #  - peso non supera la capacità residua
                        #  - non conflitta con nessun item già presente
                    sol.add(y)
                    w_used  += inst.weights[y]   # aggiorna il peso totale
                    p_total += inst.profits[y]   # aggiorna il profitto totale
                    improved = True               # aggiorna il flag miglioramento
                  
                    break # esci dal for y in outside e ricomincia il while

            if improved:  # se c'è stato un miglioramento, ricomincia il ciclo while (salta lo SWAP)
                continue

            # ── SWAP: rimuovi x e aggiungi y se Δp > 0 ─────────────────
            best_delta = 0 # mantiene il miglior guadagno ottenuto fino a quel momento
            best_move  = None # contiene la miglior mossa (x, y, Δp, Δw)
 
            for x in list(sol): # scorre tutti gli item x presenti nella soluzione
                px, wx = inst.profits[x], inst.weights[x] # profitto e peso dell'item x
                # capacità disponibile dopo la rimozione di x
                c_rem = inst.c - (w_used - wx) 
                # vicini di x (escluso x stesso) che sono nella sol:
                # dopo la rimozione di x, la sol senza x è sol - {x} 
                sol_minus_x = sol - {x}

                for y in range(1, inst.n + 1): # scorre tutti gli item y non presenti nella soluzione
                    if y in sol: # se y è già nella soluzione, passa al prossimo y
                        continue
                    wy, py = inst.weights[y], inst.profits[y] 
                    if wy > c_rem: # se y non entra nella capacità residua, passa al prossimo y
                        continue
                    if sol_minus_x & inst.neighbors[y]: # se y conflitta con qualche elemento di sol - {x}
                        continue
                    delta = py - px # calcola il guadagno
                    if delta > best_delta: # se il guadagno è maggiore del migliore finora trovato
                        best_delta = delta # aggiorna il miglior guadagno
                        best_move  = (x, y, py, py - px, wy - wx) # aggiorna la migliore mossa

            if best_move is not None: # se è stata trovata una mossa che migliora la soluzione
                x, y, py, delta_p, delta_w = best_move
                sol.discard(x) # rimuovi x dalla soluzione
                sol.add(y)     # aggiungi y alla soluzione
                w_used  += delta_w # aggiorna il peso totale
                p_total += delta_p # aggiorna il profitto totale
                improved = True # aggiorna il flag miglioramento

        return sol, p_total, w_used # restituisce la soluzione migliorata

    # ── Entry-point pubblico ──────────────────────────────────────────────

    def solve(self):
        inst  = self.instance
        rng   = random.Random(self.seed) # generatore di numeri casuali
        start = time.time()

        # ordine base: p/w decrescente (item già ordinati così in KPCInstance)
        base_order = list(range(1, inst.n + 1))

        best_sol   = set() # soluzione migliore trovata finora
        best_p     = 0     # profitto migliore trovato finora
        restarts   = 0     # numero di restart effettuati

        for restart in range(self.n_restarts): # itera per il numero di restart
            if time.time() - start > self.time_limit_sec: # se il tempo limite è stato superato
                break

            # Primo restart: ordine p/w puro; successivi: perturbazione casuale
            if restart == 0: # primo restart: ordine p/w puro
                order = base_order[:]
            else:
                # shuffla parzialmente: prende un sotto-campione casuale
                # come "testa" e appende il resto in ordine p/w
                order = base_order[:]
                rng.shuffle(order)

            sol, p, w = self._greedy(order) # costruzione greedy della soluzione
            sol, p, w = self._local_search(sol, p, w) # ricerca locale

            if p > best_p: # se la soluzione corrente è migliore di quella migliore trovata finora
                best_p   = p # aggiorna la soluzione migliore
                best_sol = set(sol) # aggiorna il profitto migliore

            restarts += 1

        #elapsed  = time.time() - start # tempo totale impiegato
        # Riconverti agli indici originali
        sol_orig = {inst.sorted_to_orig[v] for v in best_sol}

        return {
            "optimal_value": best_p, # Valore della migliore soluzione trovata (profitto totale)
            "solution":      sorted(sol_orig), # Lista ordinata di indici degli item inclusi nella soluzione finale
            #"elapsed_time":  elapsed, # Tempo totale impiegato (secondi)
            "restarts":      restarts, # Quanti tentativi multi-start è riuscito a completare prima del termine o del timeout
        }
