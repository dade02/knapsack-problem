import os
import time
import csv
import argparse
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from kpc_solver import KPCInstance, CFSSolver, GreedyLocalSearchSolver

MAX_PER_GROUP = 3  # numero massimo di istanze per combinazione di parametri

# Regex/String parsing for dat file
def load_instance_from_dat(file_path):
    n = None # numero di nodi
    c = None # capacità del knapsack
    profits = {} # profitto associato a ogni nodo
    weights = {} # peso associato a ogni nodo
    edges = [] # lista di archi (u,v)
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    lines = content.split('\n')
    mode = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): # salta righe vuote o commenti
            continue
        
        if line.startswith('param n :='): # leggo il numero di nodi
            parts = line.split(':=')
            n = int(parts[1].replace(';', '').strip())
            continue
        elif line.startswith('param c :='): # leggo la capacità del knapsack
            parts = line.split(':=')
            c = int(parts[1].replace(';', '').strip())
            continue
        elif line.startswith('param : V : p w :='): # leggo i profitto e i pesi
            mode = 'V'
            continue
        elif line.startswith('set E :='): # leggo gli archi
            mode = 'E'
            continue
        elif line == ';': # finisce la lettura dei dati del blocco
            mode = None
            continue
            
        if mode == 'V':
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0]) + 1  # Convert to 1-indexed
                p = int(parts[1]) 
                w = int(parts[2])
                profits[idx] = p
                weights[idx] = w
        elif mode == 'E':
            parts = line.split()
            if len(parts) >= 2:
                u = int(parts[0]) + 1
                v = int(parts[1]) + 1
                edges.append((u, v))
                
    return n, c, profits, weights, edges

def get_all_instances():
    """Finds all instance files and parses their properties from name."""
    instances = []
    
    # 1. Main Testbed (KPCG_instances)
    # Folders: C1, C3, C10, R1, R3, R10
    # Files: BPPC_[type]_0_[id].txt_[density]
    main_dir = "Instances/KPCG_instances"
    if os.path.exists(main_dir):
        for folder in ["C1", "C3", "C10", "R1", "R3", "R10"]: # itero sulle cartelle che contengono le istanze
            folder_path = os.path.join(main_dir, folder) # percorso della cartella
            if not os.path.exists(folder_path):
                continue
            class_letter = folder[0] # 'C' or 'R' --> profitti correlati o random
            multiplier = int(folder[1:]) # 1, 3, 10 --> moltiplicatore capacità zaino
            for fname in os.listdir(folder_path):
                # We expect names like BPPC_1_0_1.txt_0.1
                if fname.startswith("BPPC_"):
                    # "BPPC_1_0_1.txt_0.1".split("_") → ['BPPC','1','0','1.txt','0.1']
                    parts = fname.split("_")
                    if len(parts) >= 5:
                        try:
                            inst_type = int(parts[1])  # tipo di istanza
                            inst_id   = int(parts[3].replace(".txt", ""))  # id dell'istanza
                            density   = float(parts[4])  # densità dell'istanza

                            instances.append({
                                "testbed":    "main",
                                "class":      class_letter,
                                "multiplier": multiplier,
                                "type":       inst_type,
                                "id":         inst_id,
                                "density":    density,
                                "file_path":  os.path.join(folder_path, fname),
                                "key":        f"main/{folder}/{fname}"
                            })
                        except (ValueError, IndexError):
                            pass  # salta file con nome non valido

    # 2. Very Sparse Testbed (sparse_corr, sparse_rand)
    # Folders: sparse_corr (class C), sparse_rand (class R)
    # Files: test_[n]_[c]_r[density]-[id].dat
    for folder, class_letter in [("Instances/sparse_corr", "C"), ("Instances/sparse_rand", "R")]:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".dat"):
                    # Format: test_500_1000_r0.001-0.dat
                    parts = fname.replace(".dat", "").split("_") 
                    if len(parts) >= 4:
                        n = int(parts[1])  # numero di nodi
                        c = int(parts[2])  # capacità del knapsack
                        # last part is r0.001-0
                        last_part = parts[3]
                        subparts = last_part.split("-")
                        if len(subparts) == 2:
                            density = float(subparts[0][1:]) # strip 'r' --> densità
                            inst_id = int(subparts[1]) # id istanza 
                            
                            instances.append({
                                "testbed": "sparse",
                                "class": class_letter,
                                "n": n,
                                "c": c,
                                "density": density,
                                "id": inst_id,
                                "file_path": os.path.join(folder, fname),
                                "key": f"sparse/{class_letter}/{fname}"
                            })
                            
    return instances


# ---------------------------------------------------------------------------
# Worker: viene eseguito in un processo separato
# ---------------------------------------------------------------------------

def _solve_instance(inst, timeout_sec):
    """
    Carica e risolve una singola istanza con CFS e Heuristic.
    Restituisce un dict con i risultati, oppure None in caso di errore.
    Deve essere una funzione top-level (non metodo) per essere picklable.
    """
    try:
        fpath = inst["file_path"]
        n, c, profits, weights, edges = load_instance_from_dat(fpath)
        instance = KPCInstance(n, c, profits, weights, edges)

        # CFS
        t0_cfs = time.time()
        cfs_solver = CFSSolver(instance, time_limit_sec=timeout_sec)
        cfs_res = cfs_solver.solve()
        cfs_time = time.time() - t0_cfs

        # Heuristic
        t0_heur = time.time()
        heur_solver = GreedyLocalSearchSolver(instance, time_limit_sec=timeout_sec)
        heur_res = heur_solver.solve()
        heur_time = time.time() - t0_heur

        return {
            "inst":         inst,
            "n":            n,
            "c":            c,
            "cfs_opt":      cfs_res["optimal_value"],
            "cfs_time":     cfs_time,
            "cfs_nodes":    cfs_res["nodes_explored"],
            "cfs_timeout":  cfs_res["timeout"],
            "heur_opt":     heur_res["optimal_value"],
            "heur_time":    heur_time,
            "heur_restarts": heur_res["restarts"],
            "error":        None,
        }
    except Exception as e:
        return {
            "inst":  inst,
            "error": str(e),
        }



def _group_key(inst):
    """Chiave per raggruppare le istanze per combinazione di parametri."""
    if inst["testbed"] == "main":
        # raggruppa per class/multiplier/type/density
        return (inst["testbed"], inst["class"], inst["multiplier"], inst["type"], inst["density"])
    else:
        # raggruppa per class/n/c/density
        return (inst["testbed"], inst["class"], inst["n"], inst["c"], inst["density"])

def select_instances(all_instances, max_per_group=MAX_PER_GROUP):
    """Seleziona al massimo max_per_group istanze per ogni combinazione di parametri."""
    from collections import defaultdict
    groups = defaultdict(list)
    for inst in all_instances:
        groups[_group_key(inst)].append(inst)
    selected = []
    for key, group in groups.items():
        selected.extend(group[:max_per_group])
    return selected

def main():
    parser = argparse.ArgumentParser(description="Run KPC Solver experiments on all instances.") # imposta le variabili per i default
    parser.add_argument("--timeout", type=float, default=50.0, help="Timeout in seconds per solver call.") # imposta il timeout per solver call
    parser.add_argument("--csv", type=str, default="risultati_heuristic_cfs.csv", help="CSV file to store/resume results.") # imposta il file csv per salvare i risultati
    parser.add_argument("--workers", type=int, default=4,
                        help="Numero di processi paralleli (default: 4).")
    args = parser.parse_args()

    # Load existing results if they exist (checkpointing)
    solved_keys = set()
    if os.path.exists(args.csv):  # se il file csv esiste
        try:
            with open(args.csv, 'r') as f: # apri il file csv in lettura
                reader = csv.DictReader(f) # crea un lettore di csv
                for row in reader: # itera sulle righe
                    solved_keys.add(row["key"]) # aggiunge la chiave all'insieme dei risolti
            print(f"Rilevati {len(solved_keys)} risultati già salvati in {args.csv}. Verrà eseguito il resume.")
        except Exception as e:
            print(f"Avviso: Errore durante la lettura del CSV ({e}). Il file potrebbe essere incompleto.")
    else:
        # Write CSV header
        with open(args.csv, 'w', newline='') as f: # apri il file csv in scrittura
            writer = csv.writer(f) # crea un scrittore di csv
            writer.writerow([
                "key", "testbed", "class", "multiplier", "type", "n", "c", "density", "id",
                "cfs_opt", "cfs_time", "cfs_nodes", "cfs_timeout",
                "heur_opt", "heur_time", "heur_restarts"
            ])

    all_instances = get_all_instances() # ottieni tutte le istanze
    # Sort to make it deterministic
    all_instances.sort(key=lambda x: x["key"]) # ordina le istanze per chiave

    # Limita a MAX_PER_GROUP istanze per combinazione di parametri
    all_instances = select_instances(all_instances, max_per_group=MAX_PER_GROUP) # seleziona le istanze
    all_instances.sort(key=lambda x: x["key"])  # ri-ordina dopo la selezione

    total_count = len(all_instances) # numero totale di istanze
    print(f"Trovate {total_count} istanze totali (max {MAX_PER_GROUP} per combinazione di parametri).")

    # esclude i file per cui è già stato trovato un risultato
    to_solve = [inst for inst in all_instances if inst["key"] not in solved_keys] # istanze da risolvere

    print(f"Ci sono {len(to_solve)} istanze rimanenti da risolvere.")

    if not to_solve:
        print("Tutte le istanze selezionate sono già state risolte. Generazione dei report...")
        generate_reports(args.csv)
        return

    count = 0          # contatore di istanze completate
    t_start = time.time()
    csv_lock = threading.Lock()  # protegge le scritture sul file CSV

    print(f"Avvio esecuzione parallela con {args.workers} processi.")

    # spawn evita il deadlock fork+ortools (ortools usa thread C++ interni
    # che non sopravvivono a fork su Linux)
    mp_ctx = multiprocessing.get_context('spawn')

    with open(args.csv, 'a', newline='') as f:
        writer = csv.writer(f)

        # Crea il pool di processi paralleli
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as executor:
            # Backpressure: al massimo workers*2 future in volo contemporaneamente.
            # Evita di accodare migliaia di job tutti insieme.
            # with garantisce che, al termine del calcolo o in caso di errore, 
            # tutti i processi figli vengano spenti e puliti automaticamente

            from concurrent.futures import wait, FIRST_COMPLETED
            max_pending = args.workers * 2 # Calcola la soglia massima di compiti da tenere attivi contemporaneamente
            pending: set = set() # Insieme delle istanze in attesa di essere risolte
            inst_iter = iter(to_solve) # Iteratore sulle istanze da risolvere
            exhausted = False # Flag per indicare se tutte le istanze sono state inviate al pool

            def _submit_next():
                """Sottomette la prossima istanza se lo slot è disponibile."""
                nonlocal exhausted
                inst_next = next(inst_iter, None) # Prende la prossima istanza o None se non ci sono più istanze
                if inst_next is None: # Se non ci sono più istanze
                    exhausted = True 
                    return
                # delega la funzione _solve_instance a uno dei processi in background e restituisce immediatamente un oggetto Future
                fut = executor.submit(_solve_instance, inst_next, args.timeout) 
                # aggiungo il future al set pending
                pending.add(fut)

            # Riempie la coda iniziale
            while len(pending) < max_pending and not exhausted:
                _submit_next()

            while pending: # Finche ci sono istanze in attesa di essere risolte o in fase di calcolo
                # La funzione wait congela l'esecuzione fino a quando almeno un processo figlio non termina il suo calcolo
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # done --> future terminate in quel preciso instante
                # _ --> future ancora da terminare
                for future in done:
                    count += 1 # incrementa il contatore di istanze completate
                    pending.discard(future) # rimuove il future completato dal set pending
                    res = future.result()  # dizionario restituito da _solve_instance ( non bloccante)

                    # Riempie lo slot liberato se ci sono ancora istanze da calcolare
                    if not exhausted:
                        _submit_next()

                    # estrae i metadati dell' istanza dal dizionario restituito da _solve_instance
                    inst = res["inst"]
                    key  = inst["key"]

                    if res["error"] is not None: # se c'è stato un errore
                        print(f"[{count}/{len(to_solve)}] ERRORE su {key}: {res['error']}")
                        continue
                    
                    # estrae i dati dal dizionario restituitos
                    n            = res["n"]
                    c            = res["c"]
                    cfs_opt      = res["cfs_opt"]
                    cfs_time     = res["cfs_time"]
                    cfs_nodes    = res["cfs_nodes"]
                    cfs_timeout  = res["cfs_timeout"]
                    heur_opt     = res["heur_opt"]
                    heur_time    = res["heur_time"]
                    heur_restarts = res["heur_restarts"]

                    # Scrittura CSV protetta da lock (il file è aperto solo nel main process)
                    with csv_lock:
                        writer.writerow([
                            key, inst["testbed"], inst["class"],
                            inst.get("multiplier", ""), inst.get("type", ""),
                            n, c, inst["density"], inst["id"],
                            cfs_opt, f"{cfs_time:.4f}", cfs_nodes, int(cfs_timeout),
                            heur_opt, f"{heur_time:.4f}", heur_restarts
                        ])
                        f.flush()

                    print(f"[{count}/{len(to_solve)}] {key} → "
                        f"CFS: {cfs_opt} ({cfs_time:.3f}s, {'TIMEOUT' if cfs_timeout else 'OK'}), "
                        f"Heur: {heur_opt} ({heur_time:.3f}s, {heur_restarts} restarts)")

    print(f"\nEsecuzione completata in {time.time() - t_start:.2f} secondi.")
    generate_reports(args.csv)  # genera i report

def generate_reports(csv_path):
    """Generates Markdown reports and statistics based on the CSV results."""
    print("\n" + "="*50)
    print("GENERAZIONE REPORT COMPLESSIVO DEI RISULTATI")
    print("="*50)

    rows = [] # lista di righe del csv
    with open(csv_path, 'r') as f: # apre il file csv in lettura
        reader = csv.DictReader(f) # crea un lettore di csv
        for row in reader: # itera sulle righe
            rows.append(row) # aggiunge la riga alla lista

    if not rows:
        print("Nessun dato presente nel CSV.")
        return

    # Classify into Main and Sparse
    main_rows = [r for r in rows if r["testbed"] == "main"] # righe del main testbed
    sparse_rows = [r for r in rows if r["testbed"] == "sparse"] # righe del sparse testbed

    # ─── REPORT MAIN TESTBED ─────────────────────────────────────────────
    if main_rows:
        print("\n### 1. Riepilogo Istanze - Main Testbed")
        print("Vengono contate le istanze risolte all'ottimo (CFS) entro il timeout.")
        print("| Classe/Multiplier | Tot. Eseguite | CFS Solved | Heur Avg Value |")
        print("|---|---|---|---|")

        groups = ["C1", "C3", "C10", "R1", "R3", "R10"]
        for g in groups:
            class_letter = g[0] # lettera della classe
            mult = g[1:] # moltiplicatore
            g_rows = [r for r in main_rows if r["class"] == class_letter and r["multiplier"] == mult] # righe del gruppo

            tot = len(g_rows) # numero totale di istanze
            if tot == 0: # se non ci sono istanze
                continue

            cfs_solved = sum(1 for r in g_rows if r["cfs_timeout"] == "0") # numero di istanze risolte dal CFS
            heur_vals = [int(r["heur_opt"]) for r in g_rows if r.get("heur_opt", "") not in ("", "N/A")] # valori dell'euristica
            heur_avg = sum(heur_vals) / len(heur_vals) if heur_vals else 0.0 # valore medio dell'euristica

            print(f"| {g} | {tot} | **{cfs_solved}** | {heur_avg:.1f} |") # stampa i risultati

    # ─── REPORT VERY SPARSE TESTBED ──────────────────────────────────────
    if sparse_rows:
        print("\n### 2. Riepilogo Istanze - Very Sparse Testbed")
        print("| Classe/Capacity | Tot. Eseguite | CFS Solved | Heur Avg Value |")
        print("|---|---|---|---|")

        sparse_groups = [("C", "1000"), ("C", "2000"), ("R", "1000"), ("R", "2000")]
        for class_letter, cap in sparse_groups:
            # Filtra le righe del dataset sparso corrispondenti a questa specifica classe e capacità dello zaino
            g_rows = [r for r in sparse_rows if r["class"] == class_letter and r["c"] == cap] 

            tot = len(g_rows) # numero totale di istanze
            if tot == 0: # se non ci sono istanze
                continue

            cfs_solved = sum(1 for r in g_rows if r["cfs_timeout"] == "0") # numero di istanze risolte dal CFS
            heur_vals = [int(r["heur_opt"]) for r in g_rows if r.get("heur_opt", "") not in ("", "N/A")] # valori dell'euristica
            heur_avg = sum(heur_vals) / len(heur_vals) if heur_vals else 0.0 # valore medio dell'euristica

            print(f"| {class_letter}{cap} | {tot} | **{cfs_solved}** | {heur_avg:.1f} |") # stampa i risultati

    # ─── STATISTICHE MEDIE ───────────────────────────────────────────────
    print("\n### 3. Statistiche Medie sui Tempi e Nodi")
    print("| Solver | Tempo Medio (s) | Extra |")
    print("|---|---|---|")

    # CFS
    # Estrae i tempi di calcolo di CFS SOLO per le istanze in cui l'algoritmo NON è andato in timeout
    cfs_ok_times = [float(r["cfs_time"]) for r in rows if r["cfs_timeout"] == "0"]
    # Calcola il tempo medio di calcolo di CFS sulle istanze risolte con successo all'ottimo
    cfs_avg_time = sum(cfs_ok_times) / len(cfs_ok_times) if cfs_ok_times else 0.0
    # Estrae il numero di nodi esplorati da CFS per le istanze completate
    cfs_nodes_list = [int(r["cfs_nodes"]) for r in rows if r["cfs_timeout"] == "0"]
    # Calcola il numero medio di nodi esplorati da CFS sulle istanze risolte con successo all'ottimo
    cfs_avg_nodes = sum(cfs_nodes_list) / len(cfs_nodes_list) if cfs_nodes_list else 0.0
    print(f"| CFS | {cfs_avg_time:.4f}s | {cfs_avg_nodes:.1f} nodi medi |")

    # Heuristic
    # Estrae i tempi di calcolo dell'euristica escludendo eventuali celle vuote
    heur_all_times = [float(r["heur_time"]) for r in rows if r.get("heur_time", "") != ""]
    # Calcola il tempo medio di esecuzione dell'euristica su tutte le istanze testate
    heur_avg_time = sum(heur_all_times) / len(heur_all_times) if heur_all_times else 0.0
    # Estrae il numero di restart eseguiti dall'euristica escludendo eventuali celle vuote
    heur_restarts_list = [int(r["heur_restarts"]) for r in rows if r.get("heur_restarts", "") != ""]
    # Calcola il numero medio di restart eseguiti dall'euristica su tutte le istanze testate
    heur_avg_restarts = sum(heur_restarts_list) / len(heur_restarts_list) if heur_restarts_list else 0.0
    print(f"| Heuristic (Greedy+LS) | {heur_avg_time:.4f}s | {heur_avg_restarts:.1f} restarts medi |")

if __name__ == "__main__":
    main()
