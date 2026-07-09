"""
Section 4.1 – PARTITION procedure.

Partitions the candidate set V_hat into:
  - P (pruned set): items safely coverable by the clique-partition bound
  - B (branching set): B = V_hat \\ P

Invariant maintained throughout:
    budget = LB - p(I_hat) - UBC(P) >= 0
where UBC(P) = sum over cliques C in the partition of max_{i in C} p_i   (Eq. 11)

Items in V_hat are in sorted order (nonincreasing p/w ratio).
PARTITION scans them in REVERSE order so that low-ratio items go to P first,
leaving high-ratio items in B (more likely to improve the incumbent quickly).
"""


def partition_with_budget(V_hat, neighbors, profits, LB, p_I_hat):
    """
    Returns (B, P) where:
      P  = items moved into the clique partition (pruned set)
      B  = V_hat \\ P  (branching set)

    Algorithm sketch (Section 4.1):
    Repeat until no item can be added:
      Build one clique C greedily (scanning remaining items in reverse order):
        - j joins C iff {C ∪ {j}} is a clique in G (j conflicts with all C members)
        - AND adding j does not make budget negative
        - If it would make budget negative → j is discarded (not tried for future cliques)
    """
    # Process in reverse sorted order (lowest p/w ratio first)
    # Questo serve perché le clique costruite sui nodi meno efficienti tendono a stringere il bound più velocemente.
    remaining = list(reversed(V_hat))
    P = []
    UBC_P = 0
    # : Il budget rappresenta il "divario di profitto" che dobbiamo colmare per battere il record attuale.
    budget = LB - p_I_hat   # budget = LB - p(I_hat) - UBC_P, with UBC_P = 0 initially

    while remaining:
        C = []          # current clique being built
        C_max = 0       # max profit in C  →  pi*_C
        next_remaining = []   # items that conflict with someone in C (kept for next clique)

        for j in remaining:
            # Check if j can extend C to a clique:
            # j must be in conflict with EVERY current member of C.
            if all(j in neighbors[c] for c in C):
                # Compute the delta that adding j brings to UBC
                new_C_max = max(C_max, profits[j])
                delta = new_C_max - C_max # quanto guadagneremmo in termini di "valore teorico" se aggiungessimo j alla clique C

                # Se il budget non va sotto zero
                if budget - delta >= 0: 
                    # Add j to C and to P
                    C.append(j)
                    C_max = new_C_max
                    UBC_P += delta
                    budget -= delta
                    P.append(j)
                # else: budget would go negative → discard j entirely
                # (it cannot be the max of any future clique without violating the invariant)
            else:
                # j conflicts with only SOME members of C → cannot extend this clique
                # keep j for the next clique
                next_remaining.append(j)

        if not C:
            # No item could start or extend a clique → halt
            break

        remaining = next_remaining

    # B = everything in V_hat that did not end up in P
    P_set = set(P)
    B = [v for v in V_hat if v not in P_set]
    return B, P


def partition_no_budget(V_hat, neighbors):
    """
    Crea una partizione in cricche di V_hat ignorando il vincolo di budget.
    Questa partizione serve per il closed-form check del problema MCKP.
    """
    cliques = []
    remaining = list(V_hat)

    while remaining:
        C = [] # clique corrente
        leader = remaining[0] # nodo con p/w maggiore
        C.append(leader) # lo metto dentro

        next_remaining = [] # nodi rimasti
        for j in remaining[1:]:
            # j può estendere la cricca se è in conflitto con tutti i membri attuali di C
            if all(j in neighbors[c] for c in C):
                C.append(j) # lo aggiungo alla clique
            else: 
                next_remaining.append(j) # lo tengo per la prossima clique

        cliques.append(C) # aggiungo la clique alla lista delle clique
        remaining = next_remaining # passo alla prossima clique

    return cliques

