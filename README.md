# Solver per il Knapsack Problem con Conflitti (KPC)

Questo progetto implementa diversi risolutori per il problema del Knapsack Problem con Conflitti (KPC), cioè una generalizzazione del classico knapsack in cui alcuni oggetti non possono essere selezionati contemporaneamente.

Il repository include:

- un risolutore esatto basato su branch-and-bound combinatorio (CFS)
- formulazioni ILP tramite OR-Tools
- un risolutore euristico greedy + local search multi-start
- script per l’esecuzione di esperimenti e la generazione di report

## Obiettivo del progetto

Dato un insieme di oggetti con:

- profitto $p_i$
- peso $w_i$
- capacità dello zaino $c$
- un grafo di conflitti tra oggetti

si vuole trovare un sottoinsieme di oggetti ammissibile che massimizzi il profitto totale senza violare:

- il vincolo di capacità
- i vincoli di incompatibilità

## Struttura del repository

- [kpc_solver](kpc_solver): implementazione principale dei solver
  - [kpc_solver/cfs.py](kpc_solver/cfs.py): risolutore esatto CFS
  - [kpc_solver/heuristic.py](kpc_solver/heuristic.py): euristica greedy + local search
  - [kpc_solver/ilp_solvers.py](kpc_solver/ilp_solvers.py): formulazioni ILP
  - [kpc_solver/instance.py](kpc_solver/instance.py): rappresentazione delle istanze
  - [kpc_solver/bounds.py](kpc_solver/bounds.py): upper bounds e lookup tables
  - [kpc_solver/partition.py](kpc_solver/partition.py): procedure di partizione
- [Instances](Instances): istanze di benchmark usate per gli esperimenti
- [run_all_experiments.py](run_all_experiments.py): esecuzione degli esperimenti su tutte le istanze
- [test_cfs.py](test_cfs.py): test di validazione e esempi di uso
- [report_scelte_progettuali.md](report_scelte_progettuali.md) e [report_tabelle_finali.md](report_tabelle_finali.md): report descrittivi

## Requisiti

Il progetto richiede Python 3.9+ e le seguenti dipendenze:

```bash
pip install numpy matplotlib ortools pytest
```

## Installazione

```bash
cd /percorso/del/progetto
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy matplotlib ortools pytest
```

## Esecuzione rapida

### Eseguire i test

```bash
pytest -q test_cfs.py
```

### Eseguire gli esperimenti completi

```bash
python run_all_experiments.py --timeout 600 --workers 4
```

Questo script:

- carica le istanze presenti in [Instances](Instances)
- esegue sia il solver esatto sia l’euristica
- salva i risultati in [risultati_heuristic_cfs.csv](risultati_heuristic_cfs.csv)
- genera report e grafici collegati

## Esempio di utilizzo in Python

```python
from kpc_solver import KPCInstance, CFSSolver, GreedyLocalSearchSolver

n = 7
c = 8
profits = [3, 2, 3, 4, 3, 5, 4]
weights = [1, 1, 2, 3, 3, 6, 5]
edges = [(1, 2), (2, 4), (3, 4), (5, 6), (6, 7)]

instance = KPCInstance(n, c, profits, weights, edges)

cfs_res = CFSSolver(instance).solve()
heur_res = GreedyLocalSearchSolver(instance).solve()

print(cfs_res["optimal_value"])
print(heur_res["optimal_value"])
```

## Note

- I file di input nelle cartelle [Instances/KPCG_instances](Instances/KPCG_instances) e [Instances/sparse_corr](Instances/sparse_corr) / [Instances/sparse_rand](Instances/sparse_rand) sono usati come benchmark.
- Il solver esatto è progettato per essere robusto su istanze di media dimensione, mentre l’euristica è pensata per ottenere soluzioni veloci anche su istanze più grandi.

## Referenze interne

Il progetto include anche documentazione tecnica e risultati sperimentali nei file:

- [paper_text.txt](paper_text.txt)
- [report_scelte_progettuali.md](report_scelte_progettuali.md)
- [report_tabelle_finali.md](report_tabelle_finali.md)
