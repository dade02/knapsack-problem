"""
Section 5 – Bounding procedures for pruning individual nodes.
"""

class KPLookupTable:
    """
    KP-based lookup-table for computing UBL1 in O(1) time at each search node.
    As described in Section 5.1.1 of Coniglio et al. (2021).
    """

    def __init__(self, instance):
        self.instance = instance
        self.table = self._precompute()

    def _precompute(self):
        """
        Precomputes the lookup table f(j, s) using the classical Dynamic Programming
        algorithm for the Knapsack Problem:
        - f(j, s) is the optimal value considering items from {j, ..., n} and capacity s.
          f(j, s) := max{ f(j+1, s), f(j+1, s - w_j) + p_j }
        - f(n+1, s) = 0 for all s >= 0.
        - f(j, s) = -inf for s < 0.
        """
        n = self.instance.n
        c = self.instance.c
        profits = self.instance.profits
        weights = self.instance.weights

        # Rows: 1 to n+1 (use n+2 to be safe for 1-based indexing and j=n+1)
        # Cols: 0 to c
        f = [[0] * (c + 1) for _ in range(n + 2)]

        #  l'algoritmo riempie la tabella partendo dall'ultimo oggetto e risalendo fino al primo 
        for j in range(n, 0, -1):
            # estraggo peso e profitto dell'oggetto corrente
            w_j = weights[j]
            p_j = profits[j]
            # per ogni peso possibile,calcolo il profitto massimo che si può ottenere
            for s in range(c + 1):
                # Se decido di non prenderlo, il profitto massimo per lo zaino con capacità s rimane esattamente 
                # quello che era già stato calcolato per gli oggetti successivi
                val_exclude = f[j + 1][s]
                # se c' è  spazio
                if s >= w_j:
                    # Se decido di prenderlo, il profitto totale sarà il profitto di j più il profitto massimo che si poteva 
                    # ottenere con la capacità residua (cioè s meno il peso di j)
                    val_include = f[j + 1][s - w_j] + p_j
                else:
                    # impossible inserirlo
                    val_include = -float('inf')
                # Scelgo il massimo tra non prenderlo e prenderlo
                f[j][s] = max(val_exclude, val_include)
        return f

    def get_ub_l1(self, V_hat, c_hat):
        """
        Returns the KP-based lookup-table upper bound UBL1(V_hat) in O(1):
          UBL1(V_hat) := f(alpha_1(V_hat), c(V_hat))
        """
        # se non ci sono candidati o la capacità è zero, restituisce 0
        if not V_hat or c_hat <= 0:
            return 0

        # alpha_1(V_hat) è il primo elemento in V_hat (che è ordinata per indice crescente)
        alpha_1 = V_hat[0]
        # Safeguard the capacity value s ( controllo)
        s = max(0, min(c_hat, self.instance.c))
        # restituisce direttamente il valore massimo memorizzato
        return self.table[alpha_1][s]


def compute_ub_mt(V_hat, c_hat, instance):
    """
    Computes the Martello-Toth upper bound UBMT(V_hat) in O(|V_hat|) time.
    As described in Section 5.1.2.
    """
    # se non ci sono candidati o la capacità è zero, restituisce 0
    if not V_hat or c_hat <= 0:
        return 0

    weights = instance.weights
    profits = instance.profits

    # Trova l'oggetto critico t in V_hat
    cum_w = 0
    cum_p = 0
    critical_idx = -1
    # Nel momento esatto in cui l'oggetto corrente fa sforare lo zaino (cum_w + w_v > c_hat), il codice si ferma
    for idx, v in enumerate(V_hat):
        w_v = weights[v]
        if cum_w + w_v > c_hat:
            critical_idx = idx
            break
        cum_w += w_v
        cum_p += profits[v]

    # Se tutti gli oggetti entrano nello zaino, il profitto ottimo del rilassamento è cum_p
    if critical_idx == -1:
        return cum_p

    # Altrimenti, l'oggetto critico t è all'indice critical_idx
    v_t = V_hat[critical_idx]
    w_t = weights[v_t]
    p_t = profits[v_t]
    c_bar = c_hat - cum_w

    # Caso 0: L'oggetto critico t è ESCLUSO dallo zaino
    # Consideriamo gli oggetti da 1 a t-1 (il cui profitto è cum_p) e riempiamo lo spazio residuo con t+1
    # Verifichiamo se t+1 esiste in V_hat
    if critical_idx + 1 < len(V_hat):
        v_next = V_hat[critical_idx + 1]
        w_next = weights[v_next]
        p_next = profits[v_next]
        # Contributo continuo del prossimo elemento: floor(c_bar * p_next / w_next)
        ub0 = cum_p + int(c_bar * p_next // w_next)
    else:
        ub0 = cum_p

    # Caso 1: L'oggetto critico t è INCLUSO nello zaino
    # Dobbiamo liberare almeno (w_t - c_bar) di peso dagli elementi precedenti {1, ..., t-1}
    # L'elemento più vicino a t (quindi con efficienza peggiore tra i selezionati) è t-1 (critical_idx - 1)
    # Poiché V_hat[0] ha peso <= c_hat, critical_idx è sempre >= 1, quindi critical_idx - 1 è sempre valido
    v_prev = V_hat[critical_idx - 1]
    w_prev = weights[v_prev]
    p_prev = profits[v_prev]

    # Minima perdita di profitto: ceil((w_t - c_bar) * p_prev / w_prev)
    num = (w_t - c_bar) * p_prev
    loss = (num + w_prev - 1) // w_prev
    ub1 = cum_p + p_t - loss

    return max(ub0, ub1)


def greedy_clique_partition(V_subset, neighbors, order):
    """
    Partiziona i vertici in V_subset in cliques in modo greedy.
    order può essere:
      - 'normal': esamina gli oggetti nell'ordine ordinato di V_subset (rapporto p/w decrescente)
      - 'reverse': esamina gli oggetti in ordine invertito di V_subset (rapporto p/w crescente)
    """
    if order == 'normal':
        remaining = list(V_subset)
        # Ribaltiamo la lista e partiamo dagli oggetti peggiori
    else:
        remaining = list(reversed(V_subset))
        
    cliques = [] # Lista di clique che verranno generate
    while remaining:
        C = [] # Lista di clique corrente
        next_remaining = [] # Lista di oggetti che verranno analizzati successivamente
        for v in remaining:
            # Verifica se v è in conflitto con tutti i membri attuali di C (ossia forma un clique)
            if all(v in neighbors[c] for c in C):
                C.append(v)
            else:
                # Se non lo è, lo aggiungiamo alla lista degli oggetti da analizzare successivamente
                next_remaining.append(v)
        cliques.append(C) # Aggiunge la clique trovata alla lista di clique
        remaining = next_remaining # Sostituisce la lista degli oggetti rimasti con quella aggiornata
    return cliques


def solve_mckp_dp(cliques, c, profits, weights):
    """
    Risolve il rilassamento MCKP per una lista di cliques usando la programmazione dinamica.
    Ritorna un vettore g di lunghezza c + 1 dove g[s] è il profitto massimo per capacità s.
    """
    h = len(cliques) # Numero di cliques
    g = [0] * (c + 1) # Vettore di profitti massimi per ogni capacità
    
    for l in range(1, h + 1): # Scansiona le cliques una alla volta
        prev_g = list(g) #  scatta una "fotografia" dello stato precedente. 
        # Rappresenta i profitti ottimi calcolati usando le clique fino alla l- 1
        C_l = cliques[l - 1] # estrae la lista degli oggetti che fanno parte della clique corrente che stiamo esaminando
        for s in range(c + 1): # Scorre tutte le capacità possibili
            val_exclude = prev_g[s] # profitto massimo ottenibile senza includere alcun oggetto della clique corrente
            val_include = -float('inf') # inizializzazione a meno infinito per il profitto massimo ottenibile 
                                        # includendo un oggetto della clique corrente
            for item in C_l: # Scorre tutti gli oggetti che fanno parte della clique corrente (per trovare il più redditizio)
                w_i = weights[item]
                p_i = profits[item]
                if w_i <= s: # Se l'oggetto corrente entra nello zaino
                    # Aggiorna il profitto massimo ottenibile includendo un oggetto della clique corrente
                    val_include = max(val_include, prev_g[s - w_i] + p_i) 
            g[s] = max(val_exclude, val_include) # Memorizza il risultato dell'oggetto più redditizio della clique
    return g


class MCKPLookupTable:
    """
    MCKP-based lookup-table per calcolare UBL2 in O(1) tempo per nodo di ricerca.
    Come descritto nella sezione 5.2.1 di Coniglio et al. (2021).
    """

    def __init__(self, instance):
        self.instance = instance
        self.table = self._precompute()

    #  Avvia il metodo privato che costruirà la tabella $O(n^2 c)$ prima che la ricerca del Branch-and-Bound abbia inizio.
    def _precompute(self):
        """
        Precalcola la tabella di lookup per ogni oggetto j e ogni capacità s.
        Prende il minimo (ossia il bound superiore più stretto) tra i bound generati da due
        partizioni greedy distinte (in ordine diretto e invertito).
        """
        n = self.instance.n
        c = self.instance.c
        profits = self.instance.profits
        weights = self.instance.weights
        neighbors = self.instance.neighbors

        # table[j][s] memorizza UBL2 per l'oggetto j e capacità s.
        # j va da 1 a n. Usiamo n+2 righe per sicurezza.
        table = [[0] * (c + 1) for _ in range(n + 2)]

        for j in range(1, n + 1):
            V_check = list(range(j, n + 1)) # super-insieme da analizzare partendo dall'oggetto corrente j
            
            # Partizione 1: ordine normale (rapporto profitto/peso decrescente)
            cliques1 = greedy_clique_partition(V_check, neighbors, 'normal')
            g1 = solve_mckp_dp(cliques1, c, profits, weights)

            # Partizione 2: ordine invertito (rapporto profitto/peso crescente)
            cliques2 = greedy_clique_partition(V_check, neighbors, 'reverse')
            g2 = solve_mckp_dp(cliques2, c, profits, weights)

            # Selezioniamo il minimo tra i due bound superiori (quello più stringente)
            for s in range(c + 1):
                table[j][s] = min(g1[s], g2[s])
                
        return table

    def get_ub_l2(self, V_hat, c_hat):
        """
        Ritorna il bound superiore basato sulla lookup-table MCKP UBL2(V_hat) in O(1):
          UBL2(V_hat) := table[alpha_1(V_hat)][c(V_hat)]
        """
        # se non ci sono oggetti o lo zaino è pieno
        if not V_hat or c_hat <= 0:
            return 0
        # Estrae l'indice del primo oggetto disponibile nel nodo
        alpha_1 = V_hat[0]
        # Assicura che la capacità non superi mai quella massima
        s = max(0, min(c_hat, self.instance.c))
        # Ritorna il bound superiore memorizzato nella tabella
        return self.table[alpha_1][s]


def check_mckp_closed_form(cliques, c_hat, instance):
    """
    Verifica se il problema MCKP per la partizione in cricche fornita può essere risolto
    in forma chiusa. Se sì, ritorna il valore ottimo. Altrimenti, ritorna None.
    
    Per ogni C in P(V_hat), i(C) := arg max_{i in C} {p_i} (rompendo i pareggi a favore dell'oggetto con peso minore).
    Se la somma dei pesi di tali oggetti è <= c_hat, la soluzione è ammissibile e ottima per il rilassamento MCKP.

    prende l'oggetto più ricco di ogni gruppo e vede se entra nello zaino
    """
    total_w = 0 # peso totale degli oggetti più ricchi di ogni gruppo
    total_p = 0 # profitto totale degli oggetti più ricchi di ogni gruppo
    profits = instance.profits
    weights = instance.weights

    for C in cliques:
        # Trova l'oggetto più ricco di ogni gruppo
        best_item = -1
        best_p = -1
        best_w = float('inf')
        for i in C:
            p_i = profits[i]
            w_i = weights[i]
            if p_i > best_p: # Se l'oggetto corrente è più ricco del migliore finora trovato
                best_item = i
                best_p = p_i
                best_w = w_i
            elif p_i == best_p: # Se l'oggetto corrente ha lo stesso profitto del migliore finora trovato
                if w_i < best_w: # Se l'oggetto corrente ha un peso minore del migliore finora trovato
                    best_item = i
                    best_w = w_i
        if best_item != -1: # Se c'è almeno un oggetto nel gruppo
            total_w += best_w
            total_p += best_p

    if total_w <= c_hat: # Se il peso totale degli oggetti più ricchi di ogni gruppo è minore o uguale alla capacità
        return total_p # Ritorna il profitto totale degli oggetti più ricchi di ogni gruppo
    return None


def compute_ub_p(V_hat, c_hat, cliques, instance):
    """
    Calcola il bound basato sulla partizione UB_P(V_hat) (Sezione 5.2.2).
    Trova l'oggetto MCKP-critico usando l'euristica greedy duale e calcola
    le variabili duali pi e beta per ricavare il bound superiore.
    """
    profits = instance.profits
    weights = instance.weights

    # Mappa ogni oggetto all'indice della sua clique
    clique_map = {}
    for idx, C in enumerate(cliques):
        for item in C:
            clique_map[item] = idx

    p_bar = [0] * len(cliques) # Manterrà traccia del profitto del "campione attuale" di quella clique
    w_bar = c_hat  # La capacità residua dello zaino, inizialmente pari a tutto lo spazio disponibile nel nodo
    critical_item = None # Variabile destinata a salvare l'oggetto critico che interromperà il processo.

    # Iteriamo gli oggetti in V_hat (che sono ordinati per p/w decrescente)
    for j in V_hat:
        if j not in clique_map:
            continue
        c_idx = clique_map[j] # Ricava l'indice della clique di appartenenza dell'oggetto corrente
        pj = profits[j] # profitto dell'oggetto corrente
        wj = weights[j] # peso dell'oggetto corrente

        if pj <= p_bar[c_idx]: # Se il profitto dell'oggetto corrente è minore o uguale al profitto del "campione attuale" di quella clique
            continue
        
        # Non consumiamo tutto il peso dell'oggetto, ma solo la frazione di peso strettamente necessaria 
        # a coprire la differenza di profitto rispetto al vecchio oggetto che avevamo selezionato in quella clique
        reduction = (pj - p_bar[c_idx]) * wj / pj if pj > 0 else 0

        if reduction <= w_bar: # Se la riduzione è minore o uguale alla capacità residua
            p_bar[c_idx] = pj # Aggiorna il profitto del campione attuale
            w_bar -= reduction # Sottrae la riduzione dalla capacità residua
        else:
            critical_item = j # Salva l'oggetto critico
            break # Interrompe il ciclo

    if critical_item is None:
        # Se non troviamo un oggetto critico, usiamo la somma dei massimi profitti dei clique.
        total_max_p = 0
        for C in cliques: # itera su tutte le clique
            if C: # se la clique non è vuota
                total_max_p += max(profits[i] for i in C) # aggiunge il profitto del clique più ricco
        return total_max_p

    # Se l'oggetto critico è stato trovato, l'algoritmo calcola la variabile duale globale 
    # come il rapporto profitto/peso di questo oggetto
    beta_bar = profits[critical_item] / weights[critical_item] if weights[critical_item] > 0 else 0

    # Calcola la somma dei pi_bar per ciascuna clique
    pi_sum = 0
    for C in cliques: # itera su tutte le clique
        max_val = 0 # Profitto massimo di un clique
        for i in C: # itera su tutti gli oggetti del clique
            val = profits[i] - beta_bar * weights[i] # Calcola il profitto marginale dell'oggetto
            if val > max_val: # Se il profitto marginale dell'oggetto corrente è maggiore del profitto marginale massimo finora trovato
                max_val = val # Aggiorna il profitto marginale massimo
        pi_sum += max_val # Somma dei profitti marginali di tutti i clique

    ub_p = pi_sum + beta_bar * c_hat # Bound superiore basato sulla partizione UB_P(V_hat)
    return ub_p # Ritorna il bound superiore
    




