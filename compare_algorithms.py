#!/usr/bin/env python3
"""
compare_algorithms.py

Perbandingan cepat (estimasi) antara algoritma penjadwalan:
- ACO (implementasi di `scheduler.py`: ant_colony_optimization)
- ACO alternatif (dari `aco_scheduler.py`: aco_task_scheduling)
- SHC (stochastic hill climbing)
- Round Robin
- FCFS

Mode: estimasi (tidak mengeksekusi HTTP). Hasil: CSV dengan makespan tiap run.
"""

import os
import csv
import argparse
import statistics
from typing import List, Dict

# import utilities + algoritma dari project
from scheduler import load_tasks, VM, Task, VM_SPECS, ant_colony_optimization
from aco_scheduler import aco_task_scheduling
from shc_scheduler import stochastic_hill_climbing
from rr_scheduler import round_robin_scheduler
from fcfs_scheduler import fcfs_scheduler as first_come_first_serve

# python
def normalize_mapping(mapping, tasks, vms):
    """Return mapping as dict {task_id: vm_name}.

    Accepts:
    - dict: returned as-is
    - list[str]: list of vm names in task order -> convert by tasks order
    - list[int]: list of vm indices in task order -> map index->vms[index].name
    """
    if isinstance(mapping, dict):
        return mapping
    if isinstance(mapping, list):
        # list of vm names
        if all(isinstance(x, str) for x in mapping):
            return {tasks[i].id: mapping[i] for i in range(min(len(mapping), len(tasks)))}
        # list of vm indices
        if all(isinstance(x, int) for x in mapping):
            return {tasks[i].id: vms[x].name for i, x in enumerate(mapping) if 0 <= x < len(vms)}
    raise ValueError(f"Unsupported mapping format: {type(mapping)}")

def build_vms_from_specs() -> List[VM]:
    vms = []
    for name, spec in VM_SPECS.items():
        vms.append(VM(name=name, ip=spec.get("ip", "127.0.0.1"),
                      cpu_cores=spec.get("cpu", 1), ram_gb=spec.get("ram_gb", 1)))
    return vms

def estimate_makespan(mapping: Dict[int, str], tasks: List[Task], vms: List[VM]) -> float:
    vmap = {v.name: v for v in vms}
    tasks_by_id = {t.id: t for t in tasks}
    loads = {v.name: 0.0 for v in vms}
    for tid, vmname in mapping.items():
        t = tasks_by_id[tid]
        vm = vmap[vmname]
        exec_time = float(t.cpu_load) / max(1.0, vm.cpu_cores)
        loads[vmname] += exec_time
    return max(loads.values()) if loads else 0.0

# wrapper untuk menyamakan signature: (tasks, vms) -> mapping {task_id: vm_name}
def run_aco_main(tasks, vms, run_index=None):
    # ant_colony_optimization di scheduler menerima seed param
    seed = run_index
    return ant_colony_optimization(tasks, vms, seed=seed)

def run_aco_alt(tasks, vms, run_index=None):
    # aco_task_scheduling mengembalikan mapping task_id -> vm_name
    return aco_task_scheduling(tasks, vms)

def run_shc(tasks, vms, run_index=None):
    return stochastic_hill_climbing(tasks, vms)

def run_rr(tasks, vms, run_index=None):
    return round_robin_scheduler(tasks, vms)

def run_fcfs(tasks, vms, run_index=None):
    return first_come_first_serve(tasks, vms)

ALGORITHMS = {
    "ACO_main": run_aco_main,
    "ACO_alt": run_aco_alt,
    "SHC": run_shc,
    "RR": run_rr,
    "FCFS": run_fcfs
}

def compare(tasks, vms, runs: int, out_file: str):
    rows = []
    summary = {}

    for name, fn in ALGORITHMS.items():
        makespans = []
        for r in range(1, runs + 1):
            mapping_raw = fn(tasks, vms, run_index=r)
            try:
                mapping = normalize_mapping(mapping_raw, tasks, vms)
            except Exception as e:
                print(f"Warning: could not normalize mapping for {name} run {r}: {e}")
                mapping = {}

            if not mapping:
                ms = float("inf")
            else:
                ms = estimate_makespan(mapping, tasks, vms)
            makespans.append(ms)
            rows.append({"algorithm": name, "run": r, "makespan": ms})
        # compute summary
        finite = [m for m in makespans if m != float("inf")]
        summary[name] = {
            "runs": runs,
            "mean": statistics.mean(finite) if finite else float("nan"),
            "stdev": statistics.stdev(finite) if len(finite) > 1 else 0.0,
            "min": min(finite) if finite else float("nan"),
            "max": max(finite) if finite else float("nan"),
        }

    # write CSV
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "run", "makespan"])
        writer.writeheader()
        writer.writerows(rows)

    # print summary
    print(f"\nComparison complete — results written to {out_file}\n")
    for name, s in summary.items():
        print(f"{name:10s} runs={s['runs']:2d} mean={s['mean']:.4f} stdev={s['stdev']:.4f} min={s['min']:.4f} max={s['max']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Compare scheduling algorithms (estimation mode).")
    parser.add_argument("--dataset", "-d", default=os.getenv("DATASET_FILE", "dataset.txt"), help="path to dataset.txt")
    parser.add_argument("--runs", "-r", type=int, default=10, help="number of runs per algorithm")
    parser.add_argument("--out", "-o", default="compare_results.csv", help="output CSV file")
    args = parser.parse_args()

    tasks = load_tasks(args.dataset)
    vms = build_vms_from_specs()

    if not tasks:
        print("No tasks loaded; exiting.")
        return

    compare(tasks, vms, args.runs, args.out)

if __name__ == "__main__":
    main()