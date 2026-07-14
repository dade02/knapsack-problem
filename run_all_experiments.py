import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import threading
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from kpc_solver import KPCInstance, CFSSolver, GreedyLocalSearchSolver

MAX_PER_GROUP = 4  # numero massimo di istanze per combinazione di parametri

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
                            if round(density, 2) not in {0.1, 0.4, 0.8}:
                                continue

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
                            if density not in {0.001, 0.01, 0.05}:
                                continue
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


        # Heuristic
        t0_heur = time.time()
        heur_solver = GreedyLocalSearchSolver(instance, time_limit_sec=timeout_sec)
        heur_res = heur_solver.solve()
        heur_time = time.time() - t0_heur

        # Recuperiamo il valore e la soluzione 
        heur_opt = heur_res["optimal_value"]
        heur_sol = heur_res.get("solution", None)


        # CFS
        t0_cfs = time.time()
        cfs_solver = CFSSolver(instance, time_limit_sec=timeout_sec,external_lb=heur_opt, external_sol=heur_sol)
        cfs_res = cfs_solver.solve()
        cfs_time = time.time() - t0_cfs

       

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
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout in seconds per solver call.") # imposta il timeout per solver call
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
        generate_gap_performance_profile(args.csv)
        return

    count = 0          # contatore di istanze completate
    t_start = time.time()







    # ─────────────────────────────────────────────────────────────────────────
    # MODALITÀ SINGLE-CORE (workers = 1)
    # ─────────────────────────────────────────────────────────────────────────
    if args.workers == 1:
        print("Avvio esecuzione sequenziale (Single-Core).")
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for inst in to_solve:
                count += 1
                key = inst["key"]
                
                # Esegui direttamente nel processo principale senza overhead
                res = _solve_instance(inst, args.timeout)
                
                if res["error"] is not None:
                    print(f"[{count}/{len(to_solve)}] ERRORE su {key}: {res['error']}")
                    continue
                
                # Estrazione dati
                n = res["n"]
                c = res["c"]
                cfs_opt = res["cfs_opt"]
                cfs_time = res["cfs_time"]
                cfs_nodes = res["cfs_nodes"]
                cfs_timeout = res["cfs_timeout"]
                heur_opt = res["heur_opt"]
                heur_time = res["heur_time"]
                heur_restarts = res["heur_restarts"]

                # Scrittura immediata su CSV
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

    # ─────────────────────────────────────────────────────────────────────────
    # MODALITÀ MULTI-CORE (workers > 1)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        print(f"Avvio esecuzione parallela con {args.workers} processi.")
        csv_lock = threading.Lock()  # protegge le scritture sul file CSV
        
        # spawn evita il deadlock fork+ortools
        mp_ctx = multiprocessing.get_context('spawn')

        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)

            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as executor:
                from concurrent.futures import wait, FIRST_COMPLETED
                max_pending = args.workers * 2 
                pending = set() 
                inst_iter = iter(to_solve) 
                exhausted = False 

                def _submit_next():
                    nonlocal exhausted
                    inst_next = next(inst_iter, None) 
                    if inst_next is None: 
                        exhausted = True 
                        return
                    fut = executor.submit(_solve_instance, inst_next, args.timeout) 
                    pending.add(fut)

                # Riempie la coda iniziale
                while len(pending) < max_pending and not exhausted:
                    _submit_next()

                while pending: 
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        count += 1 
                        pending.discard(future) 
                        res = future.result() 

                        if not exhausted:
                            _submit_next()

                        inst = res["inst"]
                        key  = inst["key"]

                        if res["error"] is not None: 
                            print(f"[{count}/{len(to_solve)}] ERRORE su {key}: {res['error']}")
                            continue
                        
                        n            = res["n"]
                        c            = res["c"]
                        cfs_opt      = res["cfs_opt"]
                        cfs_time     = res["cfs_time"]
                        cfs_nodes    = res["cfs_nodes"]
                        cfs_timeout  = res["cfs_timeout"]
                        heur_opt     = res["heur_opt"]
                        heur_time    = res["heur_time"]
                        heur_restarts = res["heur_restarts"]

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
    generate_gap_performance_profile(args.csv) # genera il grafico

def generate_reports(csv_path, output_report_file="report_tabelle_finali.md"):
    """
    Genera le 4 tabelle di riepilogo originali (Markdown) arricchite con:
    - CFS Avg Nodes: numero medio di nodi esplorati da CFS (solo istanze risolte).
    - Avg Gap (%): la media dei gap percentuali calcolati su ogni singola istanza risolta.
    
    Caratteristiche rigorose applicate:
    1. Esclude rigorosamente le istanze in timeout dal calcolo di tutte le metriche.
    2. Applica un confronto simmetrico (paired): le medie Greedy e il Gap sono calcolati 
       esclusivamente sulle stesse istanze risolte da CFS.
    3. Calcola la riga finale di MEDIA TOTALE/CLASSE come vera media ponderata di tutte 
       le singole istanze risolte del dataset (e non come media delle righe della tabella).
    """
    import os
    import csv
    import collections

    if not os.path.exists(csv_path):
        print(f"[ERRORE] File {csv_path} non trovato.")
        return

    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Nessun dato presente nel CSV.")
        return

    # Filtri originali sui due dataset
    main_rows = [r for r in rows if r["testbed"] == "main" and round(float(r["density"]), 2) in {0.1, 0.4, 0.8}]
    sparse_rows = [r for r in rows if r["testbed"] == "sparse" and float(r["density"]) in {0.001, 0.01, 0.05}]

    report_lines = []

    def log_and_append(text=""):
        print(text)
        report_lines.append(text)

    def calc_group_stats_clean(group_rows):
        """
        Calcola le statistiche di gruppo.
        Ritorna: (solved_count, avg_heur_val, avg_heur_time, avg_cfs_val, avg_cfs_time, avg_cfs_nodes, avg_gap)
        """
        tot = len(group_rows)
        if tot == 0:
            return 0, 0.0, 0.0, None, None, None, None
        
        # Filtra solo le istanze completate con successo da CFS
        solved_rows = [r for r in group_rows if r.get("cfs_timeout") == "0"]
        cfs_solved_count = len(solved_rows)
        
        if solved_rows:
            # Medie Greedy calcolate solo sulle stesse istanze risolte da CFS
            heur_vals = [float(r["heur_opt"]) for r in solved_rows if r.get("heur_opt", "") != ""]
            heur_times = [float(r["heur_time"]) for r in solved_rows if r.get("heur_time", "") != ""]
            avg_heur_val = sum(heur_vals) / len(heur_vals) if heur_vals else 0.0
            avg_heur_time = sum(heur_times) / len(heur_times) if heur_times else 0.0
            
            # Medie CFS sulle istanze risolte
            cfs_vals = [float(r["cfs_opt"]) for r in solved_rows if r.get("cfs_opt", "") != ""]
            cfs_times = [float(r["cfs_time"]) for r in solved_rows if r.get("cfs_time", "") != ""]
            avg_cfs_val = sum(cfs_vals) / len(cfs_vals) if cfs_vals else 0.0
            avg_cfs_time = sum(cfs_times) / len(cfs_times) if cfs_times else 0.0
            
            # Nodi CFS
            node_key = "cfs_nodes" if "cfs_nodes" in solved_rows[0] else "nodes_explored"
            cfs_nodes = [float(r[node_key]) for r in solved_rows if r.get(node_key, "") != ""]
            avg_cfs_nodes = sum(cfs_nodes) / len(cfs_nodes) if cfs_nodes else 0.0
            
            # Calcolo del gap individuale per ciascuna istanza e successiva media aritmetica
            gaps = []
            for r in solved_rows:
                c_val = float(r.get("cfs_opt", 0))
                h_val = float(r.get("heur_opt", 0))
                if c_val > 0:
                    # Formula del gap individuale per singola istanza
                    gaps.append(((c_val - h_val) / c_val) * 100)
            avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
            
        else:
            # Se nessuna istanza è risolta nel gruppo
            heur_vals = [float(r["heur_opt"]) for r in group_rows if r.get("heur_opt", "") != ""]
            heur_times = [float(r["heur_time"]) for r in group_rows if r.get("heur_time", "") != ""]
            avg_heur_val = sum(heur_vals) / len(heur_vals) if heur_vals else 0.0
            avg_heur_time = sum(heur_times) / len(heur_times) if heur_times else 0.0
            
            avg_cfs_val = None
            avg_cfs_time = None
            avg_cfs_nodes = None
            avg_gap = None
        
        return cfs_solved_count, avg_heur_val, avg_heur_time, avg_cfs_val, avg_cfs_time, avg_cfs_nodes, avg_gap

    def format_row(label, tot, solved, h_v, h_t, c_v, c_t, c_n, gap, is_sparse=False):
        """Formatta la riga Markdown gestendo i valori N/D"""
        c_v_str = f"{c_v:.2f}" if c_v is not None else "N/D"
        c_t_str = f"{c_t:.4f}" if c_t is not None else "N/D"
        c_n_str = f"{int(round(c_n))}" if c_n is not None else "N/D"
        gap_str = f"{gap:.2f}%" if gap is not None else "N/D"
        
        if is_sparse:
            return f"| {label[0]} | {label[1]} | {tot} | **{solved}** | {h_v:.2f} | {h_t:.4f} | {c_v_str} | {c_t_str} | {c_n_str} | {gap_str} |"
        else:
            return f"| {label} | {tot} | **{solved}** | {h_v:.2f} | {h_t:.4f} | {c_v_str} | {c_t_str} | {c_n_str} | {gap_str} |"

    def calc_true_global_averages(target_rows):
        """
        Calcola la vera media ponderata globale basandosi unicamente sulle singole 
        istanze risolte presenti nel dataset target (coerente per tutte le metriche).
        """
        solved_instances = [r for r in target_rows if r.get("cfs_timeout") == "0"]
        solved_count = len(solved_instances)
        
        if not solved_instances:
            return solved_count, 0.0, 0.0, None, None, None, None
            
        heur_vals = [float(r["heur_opt"]) for r in solved_instances if r.get("heur_opt", "") != ""]
        heur_times = [float(r["heur_time"]) for r in solved_instances if r.get("heur_time", "") != ""]
        cfs_vals = [float(r["cfs_opt"]) for r in solved_instances if r.get("cfs_opt", "") != ""]
        cfs_times = [float(r["cfs_time"]) for r in solved_instances if r.get("cfs_time", "") != ""]
        
        node_key = "cfs_nodes" if "cfs_nodes" in solved_instances[0] else "nodes_explored"
        cfs_nodes = [float(r[node_key]) for r in solved_instances if r.get(node_key, "") != ""]
        
        avg_h_v = sum(heur_vals) / len(heur_vals) if heur_vals else 0.0
        avg_h_t = sum(heur_times) / len(heur_times) if heur_times else 0.0
        avg_c_v = sum(cfs_vals) / len(cfs_vals) if cfs_vals else 0.0
        avg_c_t = sum(cfs_times) / len(cfs_times) if cfs_times else 0.0
        avg_c_n = sum(cfs_nodes) / len(cfs_nodes) if cfs_nodes else 0.0
        
        # Calcola i gap individuali di tutte le singole istanze risolte e ne fa la media globale
        gaps = []
        for r in solved_instances:
            c_val = float(r.get("cfs_opt", 0))
            h_val = float(r.get("heur_opt", 0))
            if c_val > 0:
                gaps.append(((c_val - h_val) / c_val) * 100)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        
        return solved_count, avg_h_v, avg_h_t, avg_c_v, avg_c_t, avg_c_n, avg_gap

    log_and_append("\n" + "="*80)
    log_and_append("GENERAZIONE REPORT AGGREGATI (CFS vs GREEDY) - SOLO ISTANZE COMPLETATE")
    log_and_append("="*80)
    log_and_append("> NOTA: Tutte le medie (compresi Nodi e Gap) sono calcolate esclusivamente sulle istanze risolte entro il timeout (600s).")
    log_and_append("> Il Gap % di ogni riga e della MEDIA TOTALE è la media aritmetica dei gap individuali delle singole istanze.")

    # ─────────────────────────────────────────────────────────────────────────
    # TABELLA 1. DATASET BASE: AGGREGATO PER CLASSE E TIPO
    # ─────────────────────────────────────────────────────────────────────────
    if main_rows:
        log_and_append("\n### 1. Dataset Base - Aggregato per Classe e Tipo")
        log_and_append("| Classe/Tipo | Ist. Tot | CFS Solved | Greedy Avg Val | Greedy Avg Time (s) | CFS Avg Val | CFS Avg Time (s) | CFS Avg Nodes | Avg Gap (%) |")
        log_and_append("|---|---|---|---|---|---|---|---|---|")
        
        by_class_type = collections.defaultdict(list)
        for r in main_rows:
            key = f"{r['class']}{r['multiplier']}-{r['type']}"
            by_class_type[key].append(r)
            
        def sort_key_table1(x):
            cls = x[0]
            parts = x[1:].split('-')
            mult = int(parts[0]) if parts[0].isdigit() else 0
            typ = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return (cls, mult, typ)

        sorted_keys = sorted(by_class_type.keys(), key=sort_key_table1)
        
        for k in sorted_keys:
            g_rows = by_class_type[k]
            solved, h_v, h_t, c_v, c_t, c_n, gap = calc_group_stats_clean(g_rows)
            log_and_append(format_row(k, len(g_rows), solved, h_v, h_t, c_v, c_t, c_n, gap))
            
        # Calcolo vera media ponderata globale per il Dataset Base
        solved, h_v, h_t, c_v, c_t, c_n, gap = calc_true_global_averages(main_rows)
        log_and_append(format_row("**MEDIA TOTALE**", len(main_rows), solved, h_v, h_t, c_v, c_t, c_n, gap))

    # ─────────────────────────────────────────────────────────────────────────
    # TABELLA 2. DATASET BASE: AGGREGATO PER CLASSE E DENSITÀ
    # ─────────────────────────────────────────────────────────────────────────
    if main_rows:
        log_and_append("\n### 2. Dataset Base - Aggregato per Classe e Densità")
        log_and_append("| Classe/Densità | Ist. Tot | CFS Solved | Greedy Avg Val | Greedy Avg Time (s) | CFS Avg Val | CFS Avg Time (s) | CFS Avg Nodes | Avg Gap (%) |")
        log_and_append("|---|---|---|---|---|---|---|---|---|")
        
        by_class_dens = collections.defaultdict(list)
        for r in main_rows:
            dens_val = float(r['density']) if r['density'] else 0.0
            key = f"{r['class']}{r['multiplier']}-{dens_val:.1f}"
            by_class_dens[key].append(r)
            
        def sort_key_table2(x):
            cls = x[0]
            parts = x[1:].split('-')
            mult = int(parts[0]) if parts[0].isdigit() else 0
            dens = float(parts[1]) if len(parts) > 1 else 0.0
            return (cls, mult, dens)

        sorted_keys = sorted(by_class_dens.keys(), key=sort_key_table2)
        
        for k in sorted_keys:
            g_rows = by_class_dens[k]
            solved, h_v, h_t, c_v, c_t, c_n, gap = calc_group_stats_clean(g_rows)
            log_and_append(format_row(k, len(g_rows), solved, h_v, h_t, c_v, c_t, c_n, gap))
            
        # Calcolo vera media ponderata globale (identica alla Tabella 1)
        solved, h_v, h_t, c_v, c_t, c_n, gap = calc_true_global_averages(main_rows)
        log_and_append(format_row("**MEDIA TOTALE**", len(main_rows), solved, h_v, h_t, c_v, c_t, c_n, gap))

    # ─────────────────────────────────────────────────────────────────────────
    # TABELLA 3. DATASET SPARSE: ISTANZE CORRELATE (CLASSE C)
    # ─────────────────────────────────────────────────────────────────────────
    sparse_c = [r for r in sparse_rows if r["class"] == "C"]
    if sparse_c:
        log_and_append("\n### 3. Dataset Very Sparse - Istanze Correlate (Classe C)")
        log_and_append("| Nodi/Cap (c) | Densità | Ist. Tot | CFS Solved | Greedy Avg Val | Greedy Avg Time (s) | CFS Avg Val | CFS Avg Time (s) | CFS Avg Nodes | Avg Gap (%) |")
        log_and_append("|---|---|---|---|---|---|---|---|---|---|")
        
        by_size_dens = collections.defaultdict(list)
        for r in sparse_c:
            dens_val = float(r['density']) if r['density'] else 0.0
            key = (f"{r['n']}/{r['c']}", dens_val)
            by_size_dens[key].append(r)
            
        sorted_keys = sorted(by_size_dens.keys(), key=lambda x: (int(x[0].split('/')[0]) if '/' in x[0] else 0, int(x[0].split('/')[1]) if '/' in x[0] else 0, x[1]))
        
        for size, dens in sorted_keys:
            g_rows = by_size_dens[(size, dens)]
            solved, h_v, h_t, c_v, c_t, c_n, gap = calc_group_stats_clean(g_rows)
            log_and_append(format_row((size, dens), len(g_rows), solved, h_v, h_t, c_v, c_t, c_n, gap, is_sparse=True))
            
        # Vera media ponderata per la Classe C del dataset sparse
        solved, h_v, h_t, c_v, c_t, c_n, gap = calc_true_global_averages(sparse_c)
        c_v_str = f"{c_v:.2f}" if c_v is not None else "N/D"
        c_t_str = f"{c_t:.4f}" if c_t is not None else "N/D"
        c_n_str = f"{int(round(c_n))}" if c_n is not None else "N/D"
        gap_str = f"{gap:.2f}%" if gap is not None else "N/D"
        log_and_append(f"| **MEDIA CLASSE C** | - | {len(sparse_c)} | **{solved}** | **{h_v:.2f}** | **{h_t:.4f}** | **{c_v_str}** | **{c_t_str}** | **{c_n_str}** | **{gap_str}** |")

    # ─────────────────────────────────────────────────────────────────────────
    # TABELLA 4. DATASET SPARSE: ISTANZE RANDOM (CLASSE R)
    # ─────────────────────────────────────────────────────────────────────────
    sparse_r = [r for r in sparse_rows if r["class"] == "R"]
    if sparse_r:
        log_and_append("\n### 4. Dataset Very Sparse - Istanze Random (Classe R)")
        log_and_append("| Nodi/Cap (c) | Densità | Ist. Tot | CFS Solved | Greedy Avg Val | Greedy Avg Time (s) | CFS Avg Val | CFS Avg Time (s) | CFS Avg Nodes | Avg Gap (%) |")
        log_and_append("|---|---|---|---|---|---|---|---|---|---|")
        
        by_size_dens = collections.defaultdict(list)
        for r in sparse_r:
            dens_val = float(r['density']) if r['density'] else 0.0
            key = (f"{r['n']}/{r['c']}", dens_val)
            by_size_dens[key].append(r)
            
        sorted_keys = sorted(by_size_dens.keys(), key=lambda x: (int(x[0].split('/')[0]) if '/' in x[0] else 0, int(x[0].split('/')[1]) if '/' in x[0] else 0, x[1]))
        
        for size, dens in sorted_keys:
            g_rows = by_size_dens[(size, dens)]
            solved, h_v, h_t, c_v, c_t, c_n, gap = calc_group_stats_clean(g_rows)
            log_and_append(format_row((size, dens), len(g_rows), solved, h_v, h_t, c_v, c_t, c_n, gap, is_sparse=True))
            
        # Vera media ponderata per la Classe R del dataset sparse
        solved, h_v, h_t, c_v, c_t, c_n, gap = calc_true_global_averages(sparse_r)
        c_v_str = f"{c_v:.2f}" if c_v is not None else "N/D"
        c_t_str = f"{c_t:.4f}" if c_t is not None else "N/D"
        c_n_str = f"{int(round(c_n))}" if c_n is not None else "N/D"
        gap_str = f"{gap:.2f}%" if gap is not None else "N/D"
        log_and_append(f"| **MEDIA CLASSE R** | - | {len(sparse_r)} | **{solved}** | **{h_v:.2f}** | **{h_t:.4f}** | **{c_v_str}** | **{c_t_str}** | **{c_n_str}** | **{gap_str}** |")

    # Scrittura su file .md
    try:
        with open(output_report_file, 'w', encoding='utf-8') as f_out:
            f_out.write("\n".join(report_lines))
        print(f"\n[INFO] Report finale con le 4 tabelle salvato in Markdown su: {output_report_file}")
    except Exception as e:
        print(f"[ERRORE] Impossibile scrivere il file di report: {e}")

def generate_gap_performance_profile(csv_path, output_image_file="performance_profile_split.png"):
    """
    Legge il CSV dei risultati e genera due grafici affiancati (subplot) del
    Performance Profile focalizzato sul Gap %: uno per la Classe C e uno per la Classe R.
    """
    import os
    import csv
    
    # Forza il backend non-GUI per evitare warning GTK/Qt su Linux
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.exists(csv_path):
        print(f"[ERRORE] File CSV non trovato in: {csv_path}")
        return

    # 1. Lettura dei dati dal CSV
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 2. Filtraggio e separazione delle classi (C e R) escludendo i timeout di CFS
    # e verificando che i valori di ottimo siano convertibili e validi
    gaps_c = []
    gaps_r = []

    for r in rows:
        # Consideriamo solo le istanze risolte con successo da CFS (no timeout)
        if r.get("cfs_timeout") == "0":
            try:
                c_val = float(r.get("cfs_opt", 0))
                h_val = float(r.get("heur_opt", 0))
                
                if c_val > 0:
                    gap = ((c_val - h_val) / c_val) * 100
                    cls = r.get("class", "").strip().upper()
                    
                    if cls == "C":
                        gaps_c.append(gap)
                    elif cls == "R":
                        gaps_r.append(gap)
            except (ValueError, TypeError):
                continue  # Salta righe con dati corrotti o incompleti

    if not gaps_c and not gaps_r:
        print("[ATTENZIONE] Nessun gap valido calcolato per le classi C e R. Grafico non generato.")
        return

    # Ordiniamo i gap per il calcolo cumulativo
    gaps_c = sorted(gaps_c)
    gaps_r = sorted(gaps_r)

    # 3. Creazione del layout (1 riga, 2 colonne) con asse Y condiviso
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=300)

    # Funzione di supporto per disegnare ciascun subplot
    def plot_subplot(ax, gaps, title):
        n_instances = len(gaps)
        if n_instances == 0:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha="center", va="center", fontsize=12)
            return

        # Definizione sicura del limite X per questa classe
        max_gap = max(gaps)
        limit_x = max_gap * 1.05 if max_gap > 0 else 5.0
        x_vals = np.linspace(0.0, limit_x, 500)

        # Calcolo ordinate (frazione cumulativa)
        y_greedy = []
        for x in x_vals:
            count = sum(1 for gap in gaps if gap <= x)
            y_greedy.append((count / n_instances) * 100)

        y_cfs = [100.0] * len(x_vals)

        # Curve
        ax.plot(x_vals, y_cfs, label="CFS (Exact)", color="#d62728", linewidth=2.5)
        ax.plot(x_vals, y_greedy, label="Greedy (Heur)", color="#2ca02c", linewidth=2)

        # Stile del subplot
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Tolleranza del Gap (%) rispetto all'ottimo ($\\theta$)", fontsize=10)
        ax.set_xlim(-0.1, limit_x * 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=10)

    # 4. Disegno dei due profili
    plot_subplot(ax1, gaps_c, "Istanze Correlate (Classe C)")
    plot_subplot(ax2, gaps_r, "Istanze Random (Classe R)")

    # Etichetta asse Y comune (presente solo a sinistra grazie a sharey=True)
    ax1.set_ylabel("% di istanze risolte entro la tolleranza", fontsize=10)
    ax1.set_ylim(-5, 105)

    # Titolo della figura globale
    plt.suptitle("Performance Profile - Qualità della Soluzione (Dataset Base & Sparse)", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()

    # 5. Salvataggio su disco
    try:
        plt.savefig(output_image_file)
        plt.close()
        print(f"[OK] Grafici salvati con successo su: {output_image_file}")
        print(f"     -> Classe C analizzate: {len(gaps_c)} istanze")
        print(f"     -> Classe R analizzate: {len(gaps_r)} istanze")
    except Exception as e:
        print(f"[ERRORE] Errore durante il salvataggio del grafico: {e}")
   
if __name__ == "__main__":
    main()
