# Relazione sulle Scelte Progettuali: Risolutori per il Knapsack Problem con Conflitti (KPC)

## Indice
1. [Introduzione al Problema (KPC)](#1-introduzione-al-problema-kpc)
2. [Risolutore Esatto Combinatorio: Combinatorial Branch-and-Bound (CFS)](#2-risolutore-esatto-combinatorio-combinatorial-branch-and-bound-cfs)
   - [Schema di Branching n-ario (Branching and Pruned Sets)](#schema-di-branching-n-ario-branching-and-pruned-sets)
   - [Fase di Preprocessing](#fase-di-preprocessing)
   - [Gerarchia dei Bound per il Pruning dei Nodi](#gerarchia-dei-bound-per-il-pruning-dei-nodi)
   - [Procedura di PARTITION con Budget](#procedura-di-partition-con-budget)
3. [Risolutore Euristico: Greedy + Local Search](#3-risolutore-euristico-greedy--local-search)
   - [Greedy Costruttivo](#greedy-costruttivo)
   - [Ricerca Locale 1-opt (Insert & Swap)](#ricerca-locale-1-opt-insert--swap)
   - [Framework Multi-start](#framework-multi-start)
4. [Considerazioni Teoriche e Complessità Strutturale](#4-considerazioni-teoriche-e-complessità-strutturale)
   - [4.1 Caratterizzazione delle Istanze: Classi C, R e Indici di Capacità (1...10)](#41-caratterizzazione-delle-istanze-classi-c-r-e-indici-di-capacità)
5. [Analisi Sperimentale e Risultati Computazionali](#5-analisi-sperimentale-e-risultati-computazionali)
   - [5.1 Efficienza Computazionale del CFS (Esatto)](#51-efficienza-computazionale-del-cfs-esatto)
   - [5.2 Qualità della Soluzione dell'Euristica (Greedy + Local Search)](#52-qualit%C3%A0-della-soluzione-delleuristica-greedy--local-search)
   - [5.3 Profilo di Performance (Analisi di `performance_profile_split.png`)](#53-profilo-di-performance-analisi-di-performance_profile_splitpng)

---

## 1. Introduzione al Problema (KPC)
Il **Knapsack Problem con Conflitti (KPC)** è una generalizzazione del classico problema dello zaino (*Knapsack Problem*, KP) in cui le relazioni di incompatibilità tra gli oggetti sono modellate tramite un **grafo di conflitto** $G = (V, E)$. 

Dato un insieme di oggetti $V = \{1, 2, \dots, n\}$, ognuno associato a un profitto $p_i > 0$ e a un peso $w_i > 0$, e data una capacità dello zaino $c > 0$:
* L'obiettivo è selezionare un sottoinsieme di oggetti che massimizzi il profitto totale.
* La somma dei pesi degli oggetti selezionati non deve superare la capacità $c$ dello zaino.
* **Vincolo aggiuntivo (Conflitti)**: Se due oggetti $i$ e $j$ sono connessi da un arco nel grafo di conflitto $G$ (cioè $\{i, j\} \in E$), essi **non possono** essere inseriti contemporaneamente nello zaino.

---

## 2. Risolutore Esatto Combinatorio: Combinatorial Branch-and-Bound (CFS)
Il risolutore **CFS** implementa un algoritmo di Branch-and-Bound che non fa uso di motori di programmazione lineare (come SCIP o CPLEX) all'interno dell'albero di ricerca per valutare l'ottimalità, ma si basa interamente su bound combinatoriali calcolabili in tempo polinomiale o tramite lookup table precalcolate.

### Schema di Branching n-ario (Branching and Pruned Sets)
A differenza del classico branching binario ($x_i = 1$ vs $x_i = 0$), le scelte progettuali del CFS si basano su uno schema di branching n-ario.
In ogni nodo dell'albero, indichiamo con $\hat{I}$ la soluzione parziale (oggetti già fissati a 1) e con $\hat{V}$ l'insieme degli oggetti candidati che non hanno conflitti con gli elementi di $\hat{I}$ e che rientrano nella capacità residua dello zaino $c(\hat{V}) = c - w(\hat{I})$.

L'insieme dei candidati $\hat{V}$ viene partizionato in due insiemi tramite la procedura `PARTITION`:
1. **Pruned Set ($P$)**: Gli oggetti che non possono portare a un miglioramento del Lower Bound corrente (incumbent), anche se aggiunti in modo ottimale.
2. **Branching Set ($B = \hat{V} \setminus P$)**: Gli oggetti su cui effettuare il branching.

Per ciascun elemento $b \in B$, viene generato un nodo figlio inserendo $b$ nella soluzione parziale $\hat{I}$, eliminando i suoi vicini di conflitto da $\hat{V}$ e aggiornando la capacità residua. 

### Fase di Preprocessing
Prima di avviare la ricerca, il CFS esegue una fase di ottimizzazione preliminare sulla radice:
1. **Lower Bound (LB) Iniziale**: Viene calcolato usando due euristiche primali:
   * *Greedy con forcing*: Si esegue un'euristica greedy $n$ volte, forzando l'inserimento dell'oggetto $k$-esimo a ogni iterazione e completando la selezione in base all'ordine di efficienza decrescente $p_i/w_i$.
   * *LP Diving* (`_diving_heuristic`): Si risolve ripetutamente il rilassamento lineare (utilizzando il risolutore GLOP orchestrato da Google OR-Tools) fissando le variabili frazionarie a 1 (se compatibili) o a 0, fino ad ottenere una soluzione intera ammissibile.
2. **Precalcolo delle Lookup Table**:
   * **KPLookupTable**: Calcola in $O(n \cdot c)$ tramite programmazione dinamica il valore ottimo dello zaino classico (senza conflitti) per ogni sottoinsieme di oggetti $\{j, \dots, n\}$ e per ogni capacità $s \in [0, c]$.
   * **MCKPLookupTable**: Calcola in $O(n^2 \cdot c)$ i bound superiori basati sul Multiple-Choice Knapsack Problem (MCKP), in cui gli oggetti sono divisi in cliques disgiunte.
3. **Pegging (Fissaggio delle variabili alla radice)** (`_run_pegging`):
   Sfruttando il Lower Bound $LB$ e i bound superiori calcolati, per ciascun nodo $i$ si verifica se:
   * *Eliminazione*: Se il Lower Bound corrente è maior o uguale al bound superiore massimo ottenibile includendo $i$ e oggetti compatibili con $i$ (anti-neighborhood $\bar{N}(i)$), allora $i$ non può far parte della soluzione ottima e viene eliminato ($x_i = 0$).
   * *Forzamento*: Se escludendo $i$ il bound superiore massimo non riesce a superare il Lower Bound, allora $i$ deve far parte della soluzione ottima ed è forzato alla radice ($x_i = 1$). I suoi nemici sono eliminati.
   Questo processo viene eseguito ricorsivamente fino a convergenza, riducendo le dimensioni del problema attivo.

### Gerarchia dei Bound per il Pruning dei Nodi
In ogni nodo dell'albero di ricerca, si tenta di fare il *pruning* (taglio) del nodo valutando una gerarchia di bound ordinati dal più veloce al più computazionalmente costoso. Il nodo viene potato se $p(\hat{I}) + \text{Bound} \leq LB$.

1. **UBL2 (MCKP-based lookup table)**: Si interroga la lookup table MCKP in tempo $O(1)$ usando il primo indice attivo di $\hat{V}$ e la capacità residua. Trattandosi di un'operazione $O(1)$, viene valutata per prima.
2. **UB_MT (Martello-Toth Bound)**: È un bound continuo specifico per il problema dello zaino (KP), calcolato in tempo $O(|\hat{V}|)$ ordinando gli oggetti per $p/w$ decrescente.
3. **UB_P (Partition-based Bound)**:
   * Viene calcolata una partizione in cliques di $\hat{V}$ senza vincolo di budget via `partition_no_budget`.
   * Si effettua un *closed-form check* (`check_mckp_closed_form`): se per ciascuna clique la somma dei pesi degli elementi a massimo profitto rientra nella capacità residua dello zaino, tale valore è l'ottimo esatto del rilassamento MCKP ed è usato come bound.
   * Altrimenti, si risolve un'approssimazione del duale dell'MCKP tramite `compute_ub_p`, trovando l'elemento critico e calcolando le variabili duali per ricavare il bound superiore $UB_P$.

### Procedura di PARTITION con Budget
Se nessun bound della gerarchia consente di potare il nodo, viene eseguito il branching. Per ridurre il numero di figli, la procedura `PARTITION` costruisce in modo euristico il pruned set $P$ più grande possibile, aumentando il numero di nodi potati all'interno di ciascun livello e mantenendo l'invariante:

$$\text{budget} = LB - p(\hat{I}) - UBC(P) \geq 0$$

dove $UBC(P)$ è la somma dei profitti massimi delle clique create all'interno di $P$. 
* La procedura scansiona gli oggetti in **ordine inverso di efficienza** ($p/w$ crescente). Questo fa sì che gli oggetti meno efficienti (che difficilmente portano a un superamento del Lower Bound) vengano inseriti per primi in $P$ (quinto potati), lasciando nel branching set $B$ solo gli elementi più promettenti (con alto rapporto $p/w$).
* Se l'inserimento di un oggetto comporterebbe un budget negativo, l'oggetto viene rimosso dal processo e inserito nel branching set $B$.

---

## 3. Risolutore Euristico: Greedy + Local Search
Per problemi di grandi dimensioni, o in contesti in cui è richiesto un tempo di calcolo limitato (pochi millisecondi), è stato sviluppato un risolutore euristico multi-start strutturato in tre fases.

### Greedy Costruttivo
Gli oggetti vengono esaminati secondo un determinato ordinamento. Un oggetto viene aggiunto alla soluzione corrente se e solo se:
1. Il suo peso è inferiore o uguale alla capacità residua dello zaino.
2. Non ha archi di conflitto nel grafo $G$ con nessun oggetto già selezionato nella soluzione.

### Ricerca Locale 1-opt (Insert & Swap)
A partire dalla soluzione greedy, l'algoritmo esegue una ricerca locale fino al raggiungimento di un ottimo locale (nessun miglioramento possibile):
* **INSERT**: Cerca oggetti non selezionati che possono essere inseriti direttamente nello zaino senza violare i vincoli di capacità e di conflitto.
* **SWAP**: Cerca coppie di nodi $(x, y)$, con $x$ attualmente nella soluzione e $y$ fuori dalla soluzione, tali che rimuovendo $x$ e inserendo $y$ si rispetti la capacità dello zaino, non si creino conflitti con il resto della soluzione, e si ottenga un incremento netto del profitto totale ($p_y - p_x > 0$). Se esistono più mosse migliorative, viene scelta quella con il massimo incremento di profitto ($\text{best-improvement}$).

### Framework Multi-start
Per evitare che l'euristica costruttiva rimanga intrappolata in un unico bacino di attrazione, viene utilizzato un approccio multi-start con perturbazioni controllate da un seed:
* Al primo tentativo, gli oggetti vengono ordinati in modo puramente decrescente per rapporto $p_i/w_i$ (ordine di efficienza).
* Nei tentativi successivi, l'ordine degli oggetti viene parzialmente rimescolato (shuffle casuale di un sotto-campione), mantenendo però una parziale stima dell'efficienza.
* L'algoritmo memorizza la migliore soluzione complessiva trovata tra tutti i restart entro il limite di tempo specificato.

---

## 4. Considerazioni Teoriche e Complessità Strutturale

Le scelte progettuali evidenziano che la complessità del KPC non dipende solo dalla dimensione del problema ($n$), ma dall'interazione intima tra la topologia del grafo e la capacità della risorsa:

1. **La Divergenza di Costo Computazionale al variare della Densità ($d$)**:
   * **Bassa densità (es. $d = 0.1$)**: Il grafo dei conflitti ha pochissimi archi. Il rilassamento dello zaino classico (senza conflitti) diventa **estremamente stretto (tight)**, poiché la quasi totale assenza di archi invalida raramente le combinazioni ottime classiche. Tuttavia, i bound basati su clique perdono significato operativo (le clique estratte contengono 1 o al massimo 2 elementi).
   * **Alta densità (es. $d = 0.8$)**: Il grafo è quasi completo, dominato da estesi insiemi di conflitti. L'albero di ricerca si restringe moltissimo (pochissimi nodi esplorati), ma ciascun nodo diventa computazionalmente "pesantissimo". L'algoritmo spende enormi risorse CPU dentro ogni singolo nodo per calcolare la partizione in clique in **`partition_no_budget()`**, valutare **`compute_ub_p()`** e gestire la complessa topologia del grafo.
   * **Densità medie (es. $d \approx 0.4$)**: Rappresentano la vera "zona di pericolo" o transizione di fase. Sia i vincoli dello zaino sia i conflitti si sovrappongono con intensità simile, annullando l'efficacia di pruning dei bound e massimizzando il numero di nodi esplorati e i timeout di calcolo.
2. **Effetto del Moltiplicatore di Capacità ($c$)**:
   * **Zaino Stretto ($C1$)**: Riduce lo spazio delle soluzioni ammissibili fin dal branching iniziale, portando a tempi minimi e pochissimi nodi.
   * **Zaino Lasco ($C10$)**: Il vincolo di peso perde efficacia, trasformando il problema nella ricerca del *Maximum Weight Independent Set* (MWIS). I bound basati su KP falliscono e il numero di istanze risolte crolla verticalmente a causa dell'esplosione combinatoria. Inoltre, il setup iniziale delle lookup table subisce un rallentamento proporzionale a $c$ ($O(n \cdot c)$ e $O(n^2 \cdot c)$).
3. **Diving Heuristic vs Pegging nel Preprocessing**:
   * La **Diving Heuristic** (`_diving_heuristic`) è un'euristica che risolve ricorsivamente rilassamenti lineari tramite il solutore `GLOP` per trovare un $LB$ di partenza molto forte. Ha un costo iniziale medio-alto ma riduce l'esplorazione successiva.
   * Il **Pegging** (`_run_pegging`) è un meccanismo matematico rigoroso che, sfruttando il $LB$ trovato dalla diving e gli Upper Bound (`_tightest_ub`), riduce in modo permanente le variabili attive (fissandole a 1 o eliminandole). Questo taglia l'albero di ricerca sul nascere, a fronte di un investimento temporale significativo durante la fase di preprocessing.

### 4.1 Caratterizzazione delle Istanze: Classi C, R e Indici di Capacità 

Per comprendere appieno le prestazioni del solutore esatto CFS e dell'euristica, è fondamentale analizzare la struttura matematica delle istanze del benchmark. Queste vengono generate combinando due parametri cruciali: la **correlazione dei dati (Classi C e R)** e il **moltiplicatore di capacità dello zaino (da 1 a 10)**.

#### 1. Classi C (Correlated) vs Classi R (Random)
La distinzione tra queste due classi definisce la struttura interna dei coefficienti della funzione obiettivo (profitti $p_i$) rispetto ai vincoli di peso ($w_i$).

*   **Classe C (Istanze Correlate / Strongly Correlated)**: 
    In queste istanze, il profitto di un oggetto è fortemente legato al suo peso (solitamente $p_i = w_i + \text{costante}$). 
    *   **Impatto sul Solutore:** Rappresentano le istanze matematicamente più difficili per i bound dello zaino. Poiché tutti gli oggetti hanno un rapporto di efficienza $\frac{p_i}{w_i} \approx 1$, l'ordinamento greedy basato sull'efficienza perde quasi tutto il suo potere discriminante. I bound combinatoriali faticano a potare i rami dell'albero perché quasi tutte le combinazioni di oggetti ammissibili sembrano ugualmente promettenti.
*   **Classe R (Istanze Random / Uncorrelated)**: 
    I profitti e i pesi degli oggetti sono generati in modo indipendente da distribuzioni uniformi disgiunte.
    *   **Impatto sul Solutore:** Sono istanze strutturalmente più semplici. L'elevata variabilità del rapporto $\frac{p_i}{w_i}$ permette al solutore e alle euristiche di identificare immediatamente gli oggetti "dominanti" e di scartare quelli palesemente inefficienti, accelerando drammaticamente il processo di potatura (*pruning*).




## 5. Analisi Sperimentale e Risultati Computazionali
I risolutori sono stati messi a confronto su un benchmark strutturato composto da un dataset principale (`main`) e un dataset altamente sparso (`sparse`), entrambi derivati da istanze di Bin Packing e grafi randomici.

### 5.1 Efficienza Computazionale del CFS (Esatto)

Nel **Dataset Principale (`main`)**:
* **Percentuale di successo**: Il CFS ha chiuso all'ottimalità **492 istanze su 574** (**85.71%**) entro il time limit di 600 secondi.
* **Sforzo computazionale**: Ha registrato un tempo medio di esecuzione globale di **65.94 secondi**, con una media di **46.432 nodi** esplorati.

#### 5.2. Analisi delle Performance e Spiegazioni Ingegneristiche per Classe

L'interazione tra la capacità dello zaino, la correlazione dei dati e la densità del grafo dei conflitti determina la reale complexity computazionale del problema. Di seguito viene presentata l'analisi dettagliata dei risultati e delle relative spiegazioni ingegneristiche:

##### A. Classi C1 e R1 (Capacità Stretta)
*   **Risultati nel Benchmark:** Tempo di calcolo minimo, numero di nodi esplorati irrisorio, tasso di successo del $100\%$ senza alcun timeout.
*   **Spiegazione:** Essendo lo zaino piccolissimo, l'albero di ricerca viene potato quasi interamente per **violazione della capacità residua** prima ancora che l'algoritmo debba elaborare la complessità dei conflitti. I nodi figli vengono troncati sul nascere perché gli oggetti non entrano fisicamente nel contenitore. 
    *   *Nelle istanze R1*, l'ordinamento per efficienza individua immediatamente i 2-3 oggetti migliori e satura lo zaino.
    *   *Nelle istanze C1*, anche se l'ordinamento è meno efficace a causa della correlazione, la stringenza del peso azzera comunque l'esplorazione combinatoria.

##### B. Classi C10 e R10 (Capacità Lasca) — Scenario ad Alta Densità ($d = 0.8$)
*   **Risultati nel Benchmark:** Successo quasi totale (32/32, 31/31), ma con tempi di esecuzione superiori alle classi strette (131s e 136s). Il numero di nodi è sorprendentemente ridotto (8.534 e 6.048).
*   **Spiegazione:** La densità $d=0.8$ rende il grafo di conflitto il vincolo dominante. Il risolutore esplora pochissimi nodi perché ogni scelta di branching innesca un effetto domino di incompatibilità che elimina immediatamente gran parte dei candidati residui. L'aumento del tempo, nonostante l'esiguo numero di nodi, è dovuto alla complessità intrinseca nel calcolo dei bound UB_P (Partition-based) su grafi quasi completi, che richiedono più tempo di CPU per ogni singola istanza esaminata.

##### C. Classi C10 e R10 (Capacità Lasca) — Scenario a Bassa Densità ($d = 0.1$, es. Dataset `sparse`)
*   **Risultati nel Benchmark :** Performance deficitarie (solo 8/32 risolte per C10-0.1). Numero di nodi esplorati massivo (459.348 per C10-0.1, 311.634 per R10-0.1)
*   **Spiegazione :**In questo scenario, né il vincolo di peso (troppo lasco) né quello di conflitto (troppo sparso) sono in grado di potare l'albero. Il solutore è costretto a una ricerca esaustiva:

   1. Assenza di Pruning: Senza conflitti densi, il risolutore non può sfruttare le cricche e non può forzare variabili tramite Pegging aggressivo.

   2. Esplosione Combinatoria: L'albero di ricerca cresce in profondità e ampiezza, portando alla saturazione del tempo limite (timeout) prima che il bound superiore possa dimostrare l'ottimalità della soluzione. Qui la memoria e il tempo di calcolo vengono consumati dall'esplorazione di rami non promettenti che il bound combinatorio non riesce a escludere.

##### D. Classi Medie (C3, R3) — La Zona di Transizione ($d \approx 0.4$)
*   **Risultati nel Benchmark:** Tempi e nodi intermedi che mostrano la difficoltà crescente passando dalla densità minima a quella media (es. C3-0.1 con 25k nodi $\to$ C3-0.4 con 240k nodi).
*   **Spiegazione:** Rappresenta la zona di transizione di fase.Il grafo a $d=0.4$ non è abbastanza denso per permettere un pruning strutturale immediato (come a $d=0.8$), ma è abbastanza complesso da rendere i bound (UB_MT, UB_P) meno precisi rispetto al caso sparso.L'algoritmo si trova nel "peggior scenario" euristico: un'alta incertezza nella scelta dei rami che genera un albero di ricerca ampio. L'aumento dei nodi tra $d=0.1$ e $d=0.4$ nelle classi C3 conferma che la complessità del KPC raggiunge il suo picco quando il vincolo di conflitto è sufficientemente articolato da creare ambiguità, ma non abbastanza da "autolimitare" la ricerca

---
---

Nel **Dataset Altamente Sparso (`sparse`)**:
* Il CFS ha mostrato forti colli di bottiglia, risolvendo solo **11 istanze correlate (Classe C) su 48** (Tempo medio: 257.44s) e **8 istanze random (Classe R) su 48** (Tempo medio: 242.91s).
* **Il paradosso: Nodi esiguiti e tempi elevati**: Esaminando i record risolti all'ottimo nel dataset sparso, si nota un comportamento peculiare: il tempo medio schizza a **$170 \text{ - } 240$ secondi**, ma il numero di nodi esplorati è incredibilmente basso (**$\approx 50 \text{ - } 70$ nodi**, con casi limite pari a 1 o 2 nodi). 
  Questo fenomeno documenta l'impatto asimmetrico e distruttivo della combinazione tra elevata capacità dello zaino $c$, scarsa densità del grafo e l'azione combinata di bound e preprocessing:
  
  1. **Costo delle Lookup Table**: Il precalcolo in programmazione dinamica di `KPLookupTable` e `MCKPLookupTable` richiede una complessità pari a $O(n \cdot c)$ e $O(n^2 \cdot c)$. Con parametri di capacità molto ampi, il solutore spende quasi l'intero tempo iniziale alla radice (Nodo 0) solo per allocare e popolare le matrici in memoria.
  2. **Efficacia Strutturale del Pegging**: Poiché il grafo è quasi privo di conflitti, l'Upper Bound classico fornito dalle lookup table è estremamente preciso ("tight"). Quando il meccanismo di *Pegging* (`_run_pegging`) interroga questi bound alla radice, riesce a calcolare con precisione matematica assoluta il valore ottimo di quasi tutte le variabili del problema. Di conseguenza, la quasi totalità delle variabili viene forzata a 1 o azzerata prima del Branch-and-Bound, riducendo l'albero di ricerca successivo a pochissimi nodi o persino a 1 solo nodo (la radice).
  3. **Overhead dei Loop del Preprocessing**: Sebbene l'albero finale sia microscopico, il filtraggio alla radice richiede tempo elevato. Mancando l' "effetto domino" tipico dei grafi densi (dove inserire un oggetto elimina istantaneamente decine di nemici), il Pegging deve scansionare le variabili ricorsivamente in modo quasi isolato, allungando i tempi dei cicli di riduzione.
  4. **Inefficienza della Diving Heuristic**: Parallelamente, la *Diving Heuristic* spende tempo prezioso per invocare ripetutamente il solutore lineare `GLOP` per trovare un Lower Bound primale iniziale. Nei problemi sparsi questo sforzo è ridondante, in quanto un semplice approccio greedy combinatorio avrebbe estratto una soluzione di valore quasi identico con un frazione del costo computazionale.

  In sintesi, il solutore risolve quasi l'intera istanza prima di avviare il branching vero e proprio, spostando l'intero onere computazionale sulla fase di setup e dimostrazione matematica del Nodo 0.

##### L'Effetto a "U" della Densità nel Dataset Sparso (Transizione di Fase)

L'analisi dei tempi di calcolo sul dataset `sparse` rivela un comportamento apparentemente paradossale: il solutore CFS converge rapidamente a densità infinitesime ($d = 0.001$) e torna a convergere a densità medio-alte ($d \geq 0.1$), ma va sistematicamente in **timeout (600s) nella "zona grigia" intermedia ($d = 0.01$ e $d = 0.05$)**. Questo fenomeno descrive una classica **transizione di fase combinatoria**, determinata dal modo in cui i vincoli di peso e i vincoli di conflitto si annullano a vicenda:

1. **Densità Estrema ($d = 0.001$) — Il Dominio dello Zaino**
   * A questa densità il grafo è quasi totalmente privo di archi.
   * Il problema collassa matematicamente sulla struttura del *Knapsack Problem* (KP) classico.
   * Di conseguenza, i bound basati sulla programmazione dinamica (`KPLookupTable`) risultano straordinariamente precisi.
   * Il meccanismo di *Pegging* alla radice, sfruttando questi bound perfetti, riesce a forzare quasi il 100% delle variabili già al Nodo 0, risolvendo l'istanza prima ancora di avviare il branching vero e proprio.

2. **La Zona Grigia ($d = 0.01$ e $d = 0.05$) — Il Limbo Combinatorio**
   Rappresenta il punto di massima frustrazione algoritmica, in cui entrambi i modelli teorici di potatura falliscono contemporaneamente:
   * *Fallimento dello Zaino:* L'introduzione anche solo dell'1% o del 5% di conflitti allontana l'ottimo reale dall'ottimo dello zaino classico. I bound della `KPLookupTable` diventano troppo larghi e ottimistici, azzerando il potere di filtraggio del Pegging alla radice.
   * *Fallimento del Grafo:* Il grafo è comunque troppo sparso per generare strutture connesse rilevanti. Le cricche estratte hanno dimensioni minuscole (1 o 2 nodi), privando la gerarchia dei bound basata su cricche (`UBL2` e `UB_P`) di qualsiasi capacità di pruning.
   * *Risultato:* Il solutore si ritrova a dover gestire una capacità non indifferente di variabili libere prive di vincoli dominanti, causando l'esplosione geometrica dell'albero di ricerca e il conseguente timeout.

3. **Densità Medio-Alte ($d \geq 0.1$) — Il Dominio del Grafo**
   Quando la densità supera la soglia critica dello $0.1$, i vincoli di conflitto iniziano a dominare e a strutturare positivamente il problema:
   * *Effetto Domino nel Branching:* Impostare una variabile a 1 comporta l'eliminazione immediata di una massa critica di oggetti vicini incompatibili, riducendo spontaneamente la larghezza dell'albero di ricerca a ogni livello.
   * *Inasprimento dei Bound:* Il grafo inizia a ospitare cricche dense e di grandi dimensioni. Poiché in una cricca può essere selezionato al massimo un oggetto, i bound duali (`UB_P`) e la lookup table MCKP (`UBL2`) tornano a essere estremamente stringenti, consentendo al solutore di potare interi rami con estrema facilità.

---

### 5.3 Qualità della Soluzione dell'Euristica (Greedy + Local Search)
L'euristica multi-start si è rivelata fulminea, completando l'intero benchmark con un tempo medio per istanza di appena **0.08 secondi**.
Prendendo come riferimento le **511 istanze totali** risolte all'ottimo dal CFS per misurare la qualità dell'approssimazione primale, emergono i seguenti dati:
* **Ottimi esatti scoperti**: L'euristica ha intercettato l'ottimo globale nel **$47.06\%$** delle istanze per la **Classe C** (120 su 255) e nel **$43.75\%$** per la **Classe R** (112 su 256).
* **Gap percentuale medio**: Calcolato sulle singole istanze, si attesta a un eccellente **$2.60\%$** per la Classe C e **$3.46\%$** per la Classe R.
* **Gap worst-case (Massimo)**: Il picco massimo di scostamento dall'ottimo è stato del **$22.20\%$** (Classe C) e del **$20.85\%$** (Classe R).

---

### 5.4 Profilo di Performance (Analisi di `performance_profile_split.png`)
Il grafico del Profilo di Performance a due sotto-pannelli (*subplots*) visualizza la funzione di distribuzione cumulativa della qualità dei risultati euristici al variare della tolleranza del gap $\theta$:

* **Subplot 1 (Classe C - Istanze Correlate)**: La curva mostra una crescita rapida ma graduale. Circa il $47\%$ delle istanze viene risolto con un gap di $\theta = 0\%$ (ottimo esatto). La curva raggiunge una stabilità quasi totale ($>95\%$ delle istanze risolte con successo) per un gap di tolleranza di circa il $15\%$, a dimostrazione del fatto che le istanze correlate, pur essendo estremamente difficili per il solutore esatto, vengono approssimate con eccezionale precisione dall'euristica primale.
* **Subplot 2 (Classe R - Istanze Random)**: La curva mostra un andamento simile ma con una pendenza leggermente più dolce nella fascia centrale. L'ottimo esatto viene intercettato nel $43.75\%$ dei casi. L'efficienza dell'euristica si stabilizza oltre il $90\%$ dei successi per gap inferiori al $12.5\%$, confermando l'affidabilità della ricerca locale basata su swap ed insert anche in assenza di correlazione diretta tra pesi e profitti.