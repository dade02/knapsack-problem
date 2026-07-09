class KPCInstance:
    def __init__(self, n, c, profits, weights, edges):
        """
        n: number of items
        c: capacity
        profits: list or dict of profits (1-indexed or 0-indexed)
        weights: list or dict of weights (1-indexed or 0-indexed)
        edges: list of tuples (u, v) representing conflicts (1-indexed or 0-indexed)
        """
        self.n = n
        self.c = c
        
        # Convert lists to dictionaries if needed ( altrimenti faccio copia del dizionario)
        self.raw_profits = {i: profits[i-1] for i in range(1, n+1)} if isinstance(profits, list) else dict(profits)
        self.raw_weights = {i: weights[i-1] for i in range(1, n+1)} if isinstance(weights, list) else dict(weights)
        self.raw_edges = set()
        for u, v in edges:
            self.raw_edges.add(frozenset({u, v})) # insieme non ordinato ed immutabile degli edges
            
        # Sort items in nonincreasing order of profit-over-weight ratio: p_i/w_i >= p_j/w_j
        # We use a stable sort and order by profit/weight descending, using the index as a secondary key.
        items = list(range(1, n+1))
        # Python di base ordina in modo crescente, per ordinare in modo decrescente basta mettere un - prima della metrica di ordinamento 
        # in caso di pareggio uso l'indice come criterio secondario
        items.sort(key=lambda idx: (-self.raw_profits[idx] / self.raw_weights[idx], idx))
        
        # Mapping between sorted space (1 to n) and original space (1 to n)
        self.sorted_to_orig = {sorted_idx: orig_idx for sorted_idx, orig_idx in enumerate(items, 1)}
        self.orig_to_sorted = {orig_idx: sorted_idx for sorted_idx, orig_idx in enumerate(items, 1)}
        
        # Define sorted profits, weights, and edges
        self.profits = {sorted_idx: self.raw_profits[orig_idx] for sorted_idx, orig_idx in self.sorted_to_orig.items()}
        self.weights = {sorted_idx: self.raw_weights[orig_idx] for sorted_idx, orig_idx in self.sorted_to_orig.items()}
        
        self.edges = set()
        for edge in self.raw_edges:
            u, v = list(edge)
            self.edges.add(frozenset({self.orig_to_sorted[u], self.orig_to_sorted[v]}))
            
        # Build neighbors and anti-neighbors (compatible items) in sorted space

        # costruisce conflitti bidirezionali (l'items i è in conflitto con i suoi neighbors e viceversa )
        self.neighbors = {i: set() for i in range(1, n+1)}
        for edge in self.edges:
            u, v = list(edge)
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)
        
        # contiene tutti gli oggetti con cui l'oggetto $i$ può convivere pacificamente nello zaino senza 
        # violare alcun vincolo di conflitto
        self.anti_neighbors = {}
        for i in range(1, n+1):
            self.anti_neighbors[i] = set(j for j in range(1, n+1) if j != i and j not in self.neighbors[i])
