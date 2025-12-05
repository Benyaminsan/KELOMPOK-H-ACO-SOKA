#!/usr/bin/env python3
"""
scheduler.py

Scheduler lengkap yang menggunakan Ant Colony Optimization (ACO) untuk
menentukan pemetaan task -> VM, lalu mengeksekusi tugas secara asinkron
ke VM target, menyimpan hasil ke CSV, dan menghitung metrik.

Cara pakai:
- Siapkan file dataset.txt (satu angka index per baris, 1..10 misal)
- Siapkan file .env dengan VM IPs: VM1_IP, VM2_IP, ...
- Jalankan: python scheduler.py
"""

import asyncio
import httpx
import time
from datetime import datetime
import csv
import pandas as pd
import sys
import os
from dotenv import load_dotenv
from collections import namedtuple
import math
import random
from typing import List, Dict, Tuple
from aco_scheduler import aco_task_scheduling
from shc_scheduler import stochastic_hill_climbing
from rr_scheduler import round_robin_scheduler
import fcfs_scheduler
from fcfs_scheduler import fcfs_scheduler as first_come_first_serve

# ---------------------------
# Konfigurasi & NamedTuples
# ---------------------------
load_dotenv()

# Ganti / tambahkan VM_SPECS sesuai kebutuhan atau .env
VM_SPECS = {
    'vm1': {'ip': os.getenv("VM1_IP", "127.0.0.1"), 'cpu': 1, 'ram_gb': 1},
    'vm2': {'ip': os.getenv("VM2_IP", "127.0.0.2"), 'cpu': 2, 'ram_gb': 2},
    'vm3': {'ip': os.getenv("VM3_IP", "127.0.0.3"), 'cpu': 4, 'ram_gb': 4},
    'vm4': {'ip': os.getenv("VM4_IP", "127.0.0.4"), 'cpu': 8, 'ram_gb': 8},
}

VM_PORT = int(os.getenv("VM_PORT", "5000"))
DATASET_FILE = os.getenv("DATASET_FILE", "dataset.txt")
RESULTS_FILE = os.getenv("RESULTS_FILE", "aco_results.csv")

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# ---------------------------
# ACO Implementation
# (adapted from the ACO module you provided)
# ---------------------------
def ant_colony_optimization(tasks: List[Task], vms: List[VM],
                             iterations: int = 200,
                             ants: int = 40,
                             alpha: float = 1.0,
                             beta: float = 2.0,
                             evaporation: float = 0.5,
                             q: float = 100.0,
                             seed: int = None) -> Dict[int, str]:
    """
    Run ACO to find a mapping from task.id -> vm.name.
    Returns dict { task_id: vm_name }.
    """
    rnd = random.Random(seed)

    num_tasks = len(tasks)
    num_vms = len(vms)

    if num_tasks == 0 or num_vms == 0:
        return {}

    initial_pheromone = 0.1
    pheromone = [[initial_pheromone for _ in range(num_vms)] for _ in range(num_tasks)]
    vm_speed = [max(1.0, float(vm.cpu_cores)) for vm in vms]

    def estimate_exec_time(task_cpu_load: float, vm_speed_value: float) -> float:
        return task_cpu_load / (vm_speed_value * 1000.0)

    def fitness_of_mapping(mapping: List[int]) -> float:
        vm_loads = [0.0 for _ in range(num_vms)]
        vm_costs = [0.0 for _ in range(num_vms)]
        for t_idx, vm_idx in enumerate(mapping):
            task = tasks[t_idx]
            exec_time = estimate_exec_time(task.cpu_load, vm_speed[vm_idx])
            cost_rate = 0.001
            cost = exec_time * cost_rate
            vm_loads[vm_idx] += exec_time
            vm_costs[vm_idx] += cost

        makespan = max(vm_loads) if vm_loads else 0.0
        total_cost = sum(vm_costs)

        avg_load = sum(vm_loads) / len(vm_loads) if vm_loads else 0.0
        variance = sum((l - avg_load) ** 2 for l in vm_loads) / len(vm_loads) if vm_loads else 0.0
        imbalance = math.sqrt(variance)

        cost_w = 0.4
        makespan_w = 0.4
        imbalance_w = 0.2

        makespan_scale = 100.0
        imbalance_scale = 50.0

        fitness = (cost_w * total_cost) + (makespan_w * makespan * makespan_scale) + (imbalance_w * imbalance * imbalance_scale)
        return fitness

    def construct_solution_for_ant(rnd_local: random.Random) -> Tuple[List[int], float]:
        solution = [-1] * num_tasks
        for t_idx, task in enumerate(tasks):
            probs = []
            total_prob = 0.0
            for v_idx in range(num_vms):
                h = vm_speed[v_idx] / max(1.0, task.cpu_load)
                val = (pheromone[t_idx][v_idx] ** alpha) * (h ** beta)
                probs.append(val)
                total_prob += val

            if total_prob == 0:
                chosen = rnd_local.randrange(num_vms)
                solution[t_idx] = chosen
            else:
                r = rnd_local.random() * total_prob
                cum = 0.0
                chosen = num_vms - 1
                for v_idx, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen = v_idx
                        break
                solution[t_idx] = chosen

        fit = fitness_of_mapping(solution)
        return solution, fit

    best_solution = None
    best_fitness = float('inf')

    for iteration in range(iterations):
        ants_solutions = []
        for a in range(ants):
            sol, fit = construct_solution_for_ant(rnd)
            ants_solutions.append((sol, fit))
            if fit < best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

        # Evaporation
        for t in range(num_tasks):
            for v in range(num_vms):
                pheromone[t][v] *= (1.0 - evaporation)
                if pheromone[t][v] < 1e-6:
                    pheromone[t][v] = 1e-6

        # Deposit
        for sol, fit in ants_solutions:
            delta = q if fit <= 0 else q / fit
            for t_idx, v_idx in enumerate(sol):
                pheromone[t_idx][v_idx] += delta

    if best_solution is None:
        return {}

    assignment = { tasks[t_idx].id: vms[best_solution[t_idx]].name for t_idx in range(num_tasks) }
    return assignment

# ---------------------------
# Task loader / helper
# ---------------------------
def get_task_load(index: int) -> float:
    # Kontrol beban CPU berdasar index; sesuaikan jika perlu
    return float(index * index * 10000)

def load_tasks(dataset_path: str) -> List[Task]:
    if not os.path.exists(dataset_path):
        print(f"Error: File dataset '{dataset_path}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    tasks: List[Task] = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            val = line.strip()
            if not val:
                continue
            try:
                index = int(val)
                cpu_load = get_task_load(index)
                task_name = f"task-{index}-{i}"
                tasks.append(Task(id=i, name=task_name, index=index, cpu_load=cpu_load))
            except ValueError:
                print(f"Peringatan: Mengabaikan baris {i+1} yang tidak valid: '{val}'", file=sys.stderr)
    print(f"Berhasil memuat {len(tasks)} tugas dari {dataset_path}")
    return tasks

# ---------------------------
# Eksekusi Task Asinkron
# ---------------------------
async def execute_task_on_vm(task: Task, vm: VM, client: httpx.AsyncClient,
                            vm_semaphore: asyncio.Semaphore, results_list: list):
    """
    Mengirim request GET sederhana ke VM (endpoint /task/{index}),
    mencatat waktu tunggu/eksekusi/selesai, dan menyimpan hasil di results_list.
    """
    url = f"http://{vm.ip}:{VM_PORT}/task/{task.index}"
    task_start_time = None
    task_finish_time = None
    task_exec_time = -1.0
    task_wait_time = -1.0

    wait_start = time.monotonic()
    try:
        async with vm_semaphore:
            task_wait_time = time.monotonic() - wait_start
            print(f"[{datetime.now()}] Mengeksekusi {task.name} (id:{task.id}) -> {vm.name} ({vm.ip})")
            start_mono = time.monotonic()
            task_start_time = datetime.now()

            response = await client.get(url, timeout=300.0)
            response.raise_for_status()

            task_exec_time = time.monotonic() - start_mono
            task_finish_time = datetime.now()
            print(f"[{datetime.now()}] Selesai {task.name} di {vm.name} — exec_time: {task_exec_time:.4f}s")
    except httpx.HTTPStatusError as e:
        print(f"Error HTTP pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    except httpx.RequestError as e:
        print(f"Error Request pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error tidak diketahui pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    finally:
        if task_start_time is None:
            task_start_time = datetime.now()
        if task_finish_time is None:
            task_finish_time = datetime.now()
        results_list.append({
            "index": task.id,
            "task_name": task.name,
            "vm_assigned": vm.name,
            "start_time": task_start_time,
            "exec_time": task_exec_time,
            "finish_time": task_finish_time,
            "wait_time": task_wait_time
        })

# ---------------------------
# Results & Metrics
# ---------------------------
def write_results_to_csv(results_list: list, filename: str = RESULTS_FILE):
    if not results_list:
        print("Tidak ada hasil untuk ditulis ke CSV.", file=sys.stderr)
        return
    # Urut berdasarkan index
    results_list.sort(key=lambda x: x['index'])
    headers = ["index", "task_name", "vm_assigned", "start_time", "exec_time", "finish_time", "wait_time"]

    # Normalisasi waktu relatif terhadap earliest start untuk keterbacaan
    min_start = min(item['start_time'] for item in results_list)
    formatted = []
    for r in results_list:
        new_r = r.copy()
        new_r['start_time'] = (r['start_time'] - min_start).total_seconds()
        new_r['finish_time'] = (r['finish_time'] - min_start).total_seconds()
        formatted.append(new_r)

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(formatted)
        print(f"Hasil disimpan ke {filename}")
    except IOError as e:
        print(f"Error menulis ke CSV {filename}: {e}", file=sys.stderr)

def calculate_and_print_metrics(results_list: list, vms: List[VM], total_schedule_time: float):
    try:
        df = pd.DataFrame(results_list)
    except Exception as e:
        print(f"Error membuat DataFrame: {e}", file=sys.stderr)
        return

    if df.empty:
        print("Tidak ada data untuk dihitung metrik.")
        return

    # Pastikan tipe kolom
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['finish_time'] = pd.to_datetime(df['finish_time'])

    success_df = df[df['exec_time'] > 0].copy()
    if success_df.empty:
        print("Tidak ada tugas yang berhasil dieksekusi (exec_time <= 0).")
        return

    num_tasks = len(success_df)
    total_cpu_time = success_df['exec_time'].sum()
    total_wait_time = success_df['wait_time'].sum()
    avg_exec_time = success_df['exec_time'].mean()
    avg_wait_time = success_df['wait_time'].mean()

    min_start = success_df['start_time'].min()
    success_df['rel_start_time'] = (success_df['start_time'] - min_start).dt.total_seconds()
    success_df['rel_finish_time'] = (success_df['finish_time'] - min_start).dt.total_seconds()
    avg_start_time = success_df['rel_start_time'].mean()
    avg_finish_time = success_df['rel_finish_time'].mean()

    makespan = total_schedule_time
    throughput = num_tasks / makespan if makespan > 0 else 0.0

    vm_exec_times = success_df.groupby('vm_assigned')['exec_time'].sum()
    # Pastikan semua VM ada di index
    for vm in vms:
        if vm.name not in vm_exec_times.index:
            vm_exec_times.loc[vm.name] = 0.0
    max_load = vm_exec_times.max()
    min_load = vm_exec_times.min()
    avg_load = vm_exec_times.mean() if len(vm_exec_times) > 0 else 0.0
    imbalance_degree = (max_load - min_load) / avg_load if avg_load > 0 else 0.0

    total_cores = sum(vm.cpu_cores for vm in vms)
    total_available_cpu_time = makespan * total_cores if makespan > 0 else 0.0
    resource_utilization = total_cpu_time / total_available_cpu_time if total_available_cpu_time > 0 else 0.0

    print("\n--- METRIK ---")
    print(f"Total Tugas Selesai       : {num_tasks}")
    print(f"Makespan (Total waktu)    : {makespan:.4f} detik")
    print(f"Throughput                : {throughput:.4f} tugas/detik")
    print(f"Total CPU Time            : {total_cpu_time:.4f} detik")
    print(f"Total Wait Time           : {total_wait_time:.4f} detik")
    print(f"Average Start Time (rel)  : {avg_start_time:.4f} detik")
    print(f"Average Execution Time    : {avg_exec_time:.4f} detik")
    print(f"Average Finish Time (rel) : {avg_finish_time:.4f} detik")
    print(f"Imbalance Degree          : {imbalance_degree:.4f}")
    print(f"Resource Utilization (CPU): {resource_utilization:.4%}")

# ---------------------------
# Main routine
# ---------------------------
async def main():
    # 1. Inisialisasi VM & tasks
    vms = [VM(name, spec['ip'], spec['cpu'], spec['ram_gb']) for name, spec in VM_SPECS.items()]
    tasks = load_tasks(DATASET_FILE)
    if not tasks:
        print("Tidak ada tugas untuk dijadwalkan. Keluar.", file=sys.stderr)
        return

    # 2. Jalankan ACO untuk mendapatkan assignment
    print("\nMenjalankan Ant Colony Optimization untuk penjadwalan...")
    # Konfigurasi ACO dapat disesuaikan
    best_assignment = ant_colony_optimization(
        tasks, vms,
        iterations=int(os.getenv("ACO_ITERATIONS", "300")),
        ants=int(os.getenv("ACO_ANTS", "50")),
        alpha=float(os.getenv("ACO_ALPHA", "1.0")),
        beta=float(os.getenv("ACO_BETA", "2.0")),
        evaporation=float(os.getenv("ACO_EVAPORATION", "0.4")),
        q=float(os.getenv("ACO_Q", "120.0")),
        seed=int(os.getenv("ACO_SEED", "42"))
    )

    if not best_assignment:
        print("ACO tidak menemukan assignment. Keluar.", file=sys.stderr)
        return

    print("\nAssignment (contoh 10 pertama):")
    shown = 0
    for tid, vmn in best_assignment.items():
        print(f"  Task {tid} -> {vmn}")
        shown += 1
        if shown >= 10:
            break
    if len(best_assignment) > 10:
        print("  ...")

    # 3. Persiapkan eksekusi async
    tasks_dict = {t.id: t for t in tasks}
    vms_dict = {v.name: v for v in vms}

    # Semaphores per-VM berdasarkan cpu_cores
    vm_semaphores = {vm.name: asyncio.Semaphore(max(1, vm.cpu_cores)) for vm in vms}

    results_list = []
    all_coroutines = []

    async with httpx.AsyncClient() as client:
        for task_id, vm_name in best_assignment.items():
            # validasi mapping
            if task_id not in tasks_dict:
                print(f"Peringatan: task_id {task_id} tidak ada di tasks_dict, lewati.", file=sys.stderr)
                continue
            if vm_name not in vms_dict:
                print(f"Peringatan: vm_name {vm_name} tidak dikenal, lewati.", file=sys.stderr)
                continue

            task = tasks_dict[task_id]
            vm = vms_dict[vm_name]
            sem = vm_semaphores[vm_name]
            all_coroutines.append(execute_task_on_vm(task, vm, client, sem, results_list))

        print(f"\nMemulai eksekusi {len(all_coroutines)} tugas secara paralel...")
        start_time = time.monotonic()
        await asyncio.gather(*all_coroutines)
        end_time = time.monotonic()
        total_schedule_time = end_time - start_time
        print(f"\nSemua eksekusi selesai dalam {total_schedule_time:.4f} detik.")

    # 4. Simpan hasil & hitung metrik
    write_results_to_csv(results_list, RESULTS_FILE)
    calculate_and_print_metrics(results_list, vms, total_schedule_time)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Dibatalkan oleh user.", file=sys.stderr)
