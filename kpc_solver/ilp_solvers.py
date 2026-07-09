from ortools.linear_solver import pywraplp
import time

def solve_ilp1(instance, time_limit_sec=600):
    """
    Solves the KPC using the natural formulation (ILP1):
    Maximize sum p_i * x_i
    Subject to:
      sum w_i * x_i <= c
      x_i + x_j <= 1   for all {i, j} in E (m constraints)
    """
    # Inizializza il motore di calcolo SCIP (uno dei solver di programmazione lineare intera open-source più potenti al mondo)
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("SCIP solver could not be initialized.")
    
    # Imposta il tempo massimo di calcolo (ms)
    solver.SetTimeLimit(int(time_limit_sec * 1000))
    
    # 1. Define binary variables x_i in sorted space (1 to n)
    x = {}
    for i in range(1, instance.n + 1):
        x[i] = solver.IntVar(0.0, 1.0, f'x_{i}')
        
    # 2. Objective Function: Maximize total profit
    objective = solver.Objective()
    for i in range(1, instance.n + 1):
        # Questa funzione "aggancia" il profitto alla variabile all'interno della formula.
        objective.SetCoefficient(x[i], float(instance.profits[i]))
    # indica che vogliamo massimizzare
    objective.SetMaximization()
    
    # 3. Capacity Constraint
    # specifica che il peso totale degli item selezionati non deve superare la capacità dello zaino
    cap_constr = solver.Constraint(-solver.infinity(), float(instance.c)) # specifico un intervallo valido
    for i in range(1, instance.n + 1):
        cap_constr.SetCoefficient(x[i], float(instance.weights[i]))
        
    # 4. Conflict Constraints (Edge-based)
    for edge in instance.edges:
        u, v = list(edge)
        # Viene creato un vincolo con limite inferiore -inf e limite superiore 1.0
        conflict_constr = solver.Constraint(-solver.infinity(), 1.0) 
        conflict_constr.SetCoefficient(x[u], 1.0)
        conflict_constr.SetCoefficient(x[v], 1.0)
        
    # 5. Solve the model
    start_time = time.time()
    status = solver.Solve()
    elapsed_time = time.time() - start_time
    
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        # "Se la variabile è stata accesa (pari a 1), metti l'oggetto i nell'insieme sol_sorted
        sol_sorted = {i for i in range(1, instance.n + 1) if x[i].solution_value() > 0.5}
        # retro-traduzione nello spazio originale
        sol_orig = {instance.sorted_to_orig[v] for v in sol_sorted}
        return {
            "optimal_value": round(solver.Objective().Value()), # il valore ottimo
            "solution": sorted(list(sol_orig)), # la soluzione ottima (lista di oggetti inclusi)
            "elapsed_time": elapsed_time, # il tempo impiegato
            "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
        }
    return {"optimal_value": 0, "solution": [], "elapsed_time": elapsed_time, "status": "FAILED"}


def solve_ilp2(instance, time_limit_sec=600):
    """
    Solves the KPC using the neighborhood formulation (ILP2):
    Maximize sum p_i * x_i
    Subject to:
      sum w_i * x_i <= c
      sum_{j in N(i)} x_j <= |N(i)| * (1 - x_i)   for all i in V (n constraints)
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("SCIP solver could not be initialized.")
        
    solver.SetTimeLimit(int(time_limit_sec * 1000))
    
    # 1. Define binary variables x_i
    x = {}
    for i in range(1, instance.n + 1):
        x[i] = solver.IntVar(0.0, 1.0, f'x_{i}')
        
    # 2. Objective Function
    objective = solver.Objective()
    for i in range(1, instance.n + 1):
        objective.SetCoefficient(x[i], float(instance.profits[i]))
    objective.SetMaximization()
    
    # 3. Capacity Constraint
    cap_constr = solver.Constraint(-solver.infinity(), float(instance.c))
    for i in range(1, instance.n + 1):
        cap_constr.SetCoefficient(x[i], float(instance.weights[i]))
        
    # 4. Neighborhood Conflict Constraints:
    # sum_{j in N(i)} x_j + |N(i)| * x_i <= |N(i)|
    for i in range(1, instance.n + 1):
        N_i = instance.neighbors[i]
        if N_i: # se non ha vicini, risparmia un vincolo    
            # limite impostato al numero di vicini dell' oggetto
            constr = solver.Constraint(-solver.infinity(), float(len(N_i)))
            # Alla variabile dell'oggetto principale $x_i$ viene assegnato come coefficiente il numero totale dei suoi nemici 
            constr.SetCoefficient(x[i], float(len(N_i)))
            for j in N_i:
                constr.SetCoefficient(x[j], 1.0) # A tutte le variabili dei suoi nemici viene assegnato coefficiente 1.0
                
    # 5. Solve the model
    start_time = time.time()
    status = solver.Solve()
    elapsed_time = time.time() - start_time
    
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        sol_sorted = {i for i in range(1, instance.n + 1) if x[i].solution_value() > 0.5}
        sol_orig = {instance.sorted_to_orig[v] for v in sol_sorted}
        return {
            "optimal_value": round(solver.Objective().Value()),
            "solution": sorted(list(sol_orig)),
            "elapsed_time": elapsed_time,
            "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
        }
    return {"optimal_value": 0, "solution": [], "elapsed_time": elapsed_time, "status": "FAILED"}


def find_edge_clique_cover(n, neighbors):
    """
    Finds a collection of cliques covering all edges in the conflict graph using a greedy heuristic.
    """

    # scorre la lista di adiacenza (neighbors) e riempie l'insieme uncovered_edges con tutti gli archi del grafo. 
    # Il controllo if u < v serve unicamente a evitare di inserire lo stesso arco due volte
    uncovered_edges = set()
    for u in range(1, n + 1):
        for v in neighbors[u]:
            if u < v:
                uncovered_edges.add(frozenset({u, v}))
                
    cliques = []
    
    while uncovered_edges:
        # Pick an arbitrary uncovered edge
        start_edge = next(iter(uncovered_edges))
        C = list(start_edge)# I due nodi di questo arco vengono estratti e inseriti nella lista C
        
        # Expand it greedily to a maximal clique
        for w in range(1, n + 1):
            if w not in C:
                if all(w in neighbors[u] for u in C): # Controlla se il candidato w è nemico di tutti gli elementi già presenti dentro la clique C
                    C.append(w) # se lo è, viene aggiunto
        
        # Alla fine del ciclo, C sarà una clique massimale.
        cliques.append(C)
        
        # Mark all edges inside the clique C as covered
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                edge = frozenset({C[i], C[j]}) # genera tutte le possibili coppie (archi interni) tra i membri della clique appena formata.
                uncovered_edges.discard(edge)  # Rimuove questi archi dall'elenco di quelli da coprire
                
    return cliques


def solve_ilp3(instance, time_limit_sec=600):
    """
    Solves the KPC using the clique-cover formulation (ILP3):
    Maximize sum p_i * x_i
    Subject to:
      sum w_i * x_i <= c
      sum_{i in C} x_i <= 1   for all C in CliqueCover(G)
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("SCIP solver could not be initialized.")
        
    solver.SetTimeLimit(int(time_limit_sec * 1000))
    
    # 1. Define binary variables x_i
    x = {}
    for i in range(1, instance.n + 1):
        x[i] = solver.IntVar(0.0, 1.0, f'x_{i}')
        
    # 2. Objective Function
    objective = solver.Objective()
    for i in range(1, instance.n + 1):
        objective.SetCoefficient(x[i], float(instance.profits[i]))
    objective.SetMaximization()
    
    # 3. Capacity Constraint
    cap_constr = solver.Constraint(-solver.infinity(), float(instance.c))
    for i in range(1, instance.n + 1):
        cap_constr.SetCoefficient(x[i], float(instance.weights[i]))
        
    # 4. Clique-based Conflict Constraints
    cliques = find_edge_clique_cover(instance.n, instance.neighbors)
    for C in cliques:
        # Per ogni clique, crea un unico vincolo geometrico. Imposta il limite inferiore a -inf e il limite superiore a 1.0
        clique_constr = solver.Constraint(-solver.infinity(), 1.0)
        for i in C:
            # Questo ciclo interno scorre tutti gli oggetti $i$ che fanno parte della clique corrente $C$ e assegna a ciascuno di essi un coefficiente pari a 1.0
            clique_constr.SetCoefficient(x[i], 1.0)
            
    # 5. Solve the model
    start_time = time.time()
    status = solver.Solve()
    elapsed_time = time.time() - start_time
    
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        sol_sorted = {i for i in range(1, instance.n + 1) if x[i].solution_value() > 0.5}
        sol_orig = {instance.sorted_to_orig[v] for v in sol_sorted}
        return {
            "optimal_value": round(solver.Objective().Value()),
            "solution": sorted(list(sol_orig)),
            "elapsed_time": elapsed_time,
            "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
            "cliques_count": len(cliques)
        }
    return {"optimal_value": 0, "solution": [], "elapsed_time": elapsed_time, "status": "FAILED"}
