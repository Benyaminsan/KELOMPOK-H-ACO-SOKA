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
from aco_scheduler import (
    aco_standard,
    stochastic_hill_climbing,
    round_robin,
    fcfs
)

# --- Konfigurasi Lingkungan ---

load_dotenv()

VM_SPECS = {
    'vm1': {'ip': os.getenv("VM1_IP"), 'cpu': 1, 'ram_gb': 1},
    'vm2': {'ip': os.getenv("VM2_IP"), 'cpu': 2, 'ram_gb': 2},
    'vm3': {'ip': os.getenv("VM3_IP"), 'cpu': 4, 'ram_gb': 4},
    'vm4': {'ip': os.getenv("VM4_IP"), 'cpu': 8, 'ram_gb': 4},
}

VM_PORT = 5000
DATASET_FILES = [
    'Low-High Dataset.txt',
    'Random Simple Dataset.txt',
    'Stratified Random Dataset.txt'
]
RESULTS_FILE = 'comparison_results.csv'
DETAILED_RESULTS_DIR = 'detailed_results'
ITERATIONS_PER_DATASET = 10

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Fungsi Helper & Definisi Task ---

def get_task_load(index: int):
    cpu_load = (index * index * 10000)
    return cpu_load

def load_tasks(dataset_path: str) -> list[Task]:
    if not os.path.exists(dataset_path):
        print(f"Error: File dataset '{dataset_path}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)
        
    tasks = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                index = int(line.strip())
                if not 1 <= index <= 10:
                    print(f"Peringatan: Task index {index} di baris {i+1} di luar rentang (1-10).")
                    continue
                
                cpu_load = get_task_load(index)
                task_name = f"task-{index}-{i}"
                tasks.append(Task(
                    id=i,
                    name=task_name,
                    index=index,
                    cpu_load=cpu_load,
                ))
            except ValueError:
                print(f"Peringatan: Mengabaikan baris {i+1} yang tidak valid: '{line.strip()}'")
    
    print(f"Berhasil memuat {len(tasks)} tugas dari {dataset_path}")
    return tasks

# --- Eksekutor Tugas Asinkron ---

async def execute_task_on_vm(task: Task, vm: VM, client: httpx.AsyncClient, 
                            vm_semaphore: asyncio.Semaphore, results_list: list):
    """
    Mengirim request GET ke VM yang ditugaskan, dibatasi oleh semaphore VM.
    Mencatat hasil dan waktu.
    """
    url = f"http://{vm.ip}:{VM_PORT}/task/{task.index}"
    task_start_time = None
    task_finish_time = None
    task_exec_time = -1.0
    task_wait_time = -1.0
    
    wait_start_mono = time.monotonic()
    
    try:
        async with vm_semaphore:
            # Waktu tunggu selesai, eksekusi dimulai
            task_wait_time = time.monotonic() - wait_start_mono
            
            print(f"Mengeksekusi {task.name} (idx: {task.id}) di {vm.name} (IP: {vm.ip})...")
            
            # Catat waktu mulai
            task_start_mono = time.monotonic()
            task_start_time = datetime.now()
            
            # Kirim request GET
            response = await client.get(url, timeout=300.0) # Timeout 5 menit
            response.raise_for_status()
            
            # Catat waktu selesai
            task_finish_time = datetime.now()
            task_exec_time = time.monotonic() - task_start_mono
            
            print(f"Selesai {task.name} (idx: {task.id}) di {vm.name}. Waktu: {task_exec_time:.4f}s")
            
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

# --- Fungsi Paska-Proses & Metrik ---

def write_results_to_csv(results_list: list, algorithm_name: str, dataset_name: str, iteration: int):
    """Menyimpan hasil eksekusi ke file CSV terpisah per algoritma, dataset, dan iterasi."""
    if not results_list:
        print("Tidak ada hasil untuk ditulis ke CSV.", file=sys.stderr)
        return

    # Buat direktori jika belum ada
    os.makedirs(DETAILED_RESULTS_DIR, exist_ok=True)
    
    filename = os.path.join(DETAILED_RESULTS_DIR, f"{algorithm_name}_{dataset_name}_iter{iteration}_details.csv")
    
    # Urutkan berdasarkan 'index' untuk keterbacaan
    results_list.sort(key=lambda x: x['index'])

    headers = ["index", "task_name", "vm_assigned", "start_time", "exec_time", "finish_time", "wait_time"]
    
    # Format datetime agar lebih mudah dibaca di CSV
    formatted_results = []
    min_start = min(item['start_time'] for item in results_list)
    for r in results_list:
        new_r = r.copy()
        new_r['start_time'] = (r['start_time'] - min_start).total_seconds()
        new_r['finish_time'] = (r['finish_time'] - min_start).total_seconds()
        formatted_results.append(new_r)

    formatted_results.sort(key=lambda item: item['start_time'])

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(formatted_results)
        # print(f"Data hasil eksekusi disimpan ke {filename}")
    except IOError as e:
        print(f"Error menulis ke CSV {filename}: {e}", file=sys.stderr)

def check_iteration_exists(algorithm_name: str, dataset_name: str, iteration: int) -> bool:
    """Mengecek apakah file hasil iterasi sudah ada."""
    filename = os.path.join(DETAILED_RESULTS_DIR, f"{algorithm_name}_{dataset_name}_iter{iteration}_details.csv")
    return os.path.exists(filename)

def calculate_metrics(results_list: list, vms: list[VM], total_schedule_time: float) -> dict:
    """Menghitung metrik dan mengembalikan sebagai dictionary."""
    try:
        df = pd.DataFrame(results_list)
    except pd.errors.EmptyDataError:
        print("Error: Hasil kosong, tidak ada metrik untuk dihitung.", file=sys.stderr)
        return None

    # Konversi kolom waktu
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['finish_time'] = pd.to_datetime(df['finish_time'])
    
    # Filter 'failed' tasks (exec_time < 0)
    success_df = df[df['exec_time'] > 0].copy()
    
    if success_df.empty:
        print("Tidak ada tugas yang berhasil diselesaikan. Metrik tidak dapat dihitung.")
        return None

    num_tasks = len(success_df)
    
    # Hitung metrik
    total_cpu_time = success_df['exec_time'].sum()
    total_wait_time = success_df['wait_time'].sum()
    
    avg_exec_time = success_df['exec_time'].mean()
    avg_wait_time = success_df['wait_time'].mean()
    
    # Waktu mulai & selesai relatif terhadap awal
    min_start = success_df['start_time'].min()
    success_df['rel_start_time'] = (success_df['start_time'] - min_start).dt.total_seconds()
    success_df['rel_finish_time'] = (success_df['finish_time'] - min_start).dt.total_seconds()
    
    avg_start_time = success_df['rel_start_time'].mean()
    avg_finish_time = success_df['rel_finish_time'].mean()
    
    makespan = total_schedule_time # Waktu dari eksekusi pertama hingga terakhir
    throughput = num_tasks / makespan if makespan > 0 else 0
    
    # Imbalance Degree (Degree of Imbalance)
    vm_exec_times = success_df.groupby('vm_assigned')['exec_time'].sum()
    max_load = vm_exec_times.max()
    min_load = vm_exec_times.min()
    avg_load = vm_exec_times.mean()
    imbalance_degree = (max_load - min_load) / avg_load if avg_load > 0 else 0
    
    # Resource Utilization
    total_available_cpu_time = 0
    total_cores = sum(vm.cpu_cores for vm in vms)
    total_available_cpu_time = makespan * total_cores
    resource_utilization = total_cpu_time / total_available_cpu_time if total_available_cpu_time > 0 else 0

    return {
        'total_tasks': num_tasks,
        'makespan': makespan,
        'throughput': throughput,
        'total_cpu_time': total_cpu_time,
        'total_wait_time': total_wait_time,
        'avg_start_time': avg_start_time,
        'avg_exec_time': avg_exec_time,
        'avg_finish_time': avg_finish_time,
        'avg_wait_time': avg_wait_time,
        'imbalance_degree': imbalance_degree,
        'resource_utilization': resource_utilization
    }

def print_metrics(metrics: dict, algorithm_name: str):
    """Menampilkan metrik ke console."""
    print(f"\n--- Hasil {algorithm_name} ---")
    print(f"Total Tugas Selesai       : {metrics['total_tasks']}")
    print(f"Makespan (Waktu Total)    : {metrics['makespan']:.4f} detik")
    print(f"Throughput                : {metrics['throughput']:.4f} tugas/detik")
    print(f"Total CPU Time            : {metrics['total_cpu_time']:.4f} detik")
    print(f"Total Wait Time           : {metrics['total_wait_time']:.4f} detik")
    print(f"Average Start Time (rel)  : {metrics['avg_start_time']:.4f} detik")
    print(f"Average Execution Time    : {metrics['avg_exec_time']:.4f} detik")
    print(f"Average Finish Time (rel) : {metrics['avg_finish_time']:.4f} detik")
    print(f"Average Wait Time         : {metrics['avg_wait_time']:.4f} detik")
    print(f"Imbalance Degree          : {metrics['imbalance_degree']:.4f}")
    print(f"Resource Utilization (CPU): {metrics['resource_utilization']:.4%}")

# --- Fungsi untuk Menjalankan Satu Algoritma ---

async def run_algorithm(algorithm_name: str, algorithm_func, tasks: list[Task], 
                       vms: list[VM], tasks_dict: dict, vms_dict: dict, 
                       client: httpx.AsyncClient, dataset_name: str, iteration: int, **kwargs) -> dict:
    """Menjalankan satu algoritma dan mengembalikan metriknya."""
    
    # Jalankan algoritma scheduling
    assignment = algorithm_func(tasks, vms, **kwargs)
    
    # Eksekusi tugas
    results_list = []
    vm_semaphores = {vm.name: asyncio.Semaphore(vm.cpu_cores) for vm in vms}
    
    all_task_coroutines = []
    for task_id, vm_name in assignment.items():
        task = tasks_dict[task_id]
        vm = vms_dict[vm_name]
        sem = vm_semaphores[vm_name]
        
        all_task_coroutines.append(
            execute_task_on_vm(task, vm, client, sem, results_list)
        )
    
    schedule_start_time = time.monotonic()
    await asyncio.gather(*all_task_coroutines)
    schedule_end_time = time.monotonic()
    total_schedule_time = schedule_end_time - schedule_start_time
    
    # Simpan hasil dan hitung metrik
    write_results_to_csv(results_list, algorithm_name, dataset_name, iteration)
    metrics = calculate_metrics(results_list, vms, total_schedule_time)
    
    if metrics:
        metrics['algorithm'] = algorithm_name
        metrics['dataset'] = dataset_name
        metrics['iteration'] = iteration
        return metrics
    
    return None

def calculate_average_metrics(metrics_list: list) -> dict:
    """Menghitung rata-rata metrik dari beberapa iterasi."""
    if not metrics_list:
        return None
    
    avg_metrics = {
        'algorithm': metrics_list[0]['algorithm'],
        'dataset': metrics_list[0]['dataset'],
        'total_tasks': metrics_list[0]['total_tasks'],
        'makespan': sum(m['makespan'] for m in metrics_list) / len(metrics_list),
        'throughput': sum(m['throughput'] for m in metrics_list) / len(metrics_list),
        'total_cpu_time': sum(m['total_cpu_time'] for m in metrics_list) / len(metrics_list),
        'total_wait_time': sum(m['total_wait_time'] for m in metrics_list) / len(metrics_list),
        'avg_start_time': sum(m['avg_start_time'] for m in metrics_list) / len(metrics_list),
        'avg_exec_time': sum(m['avg_exec_time'] for m in metrics_list) / len(metrics_list),
        'avg_finish_time': sum(m['avg_finish_time'] for m in metrics_list) / len(metrics_list),
        'avg_wait_time': sum(m['avg_wait_time'] for m in metrics_list) / len(metrics_list),
        'imbalance_degree': sum(m['imbalance_degree'] for m in metrics_list) / len(metrics_list),
        'resource_utilization': sum(m['resource_utilization'] for m in metrics_list) / len(metrics_list),
    }
    return avg_metrics

def write_comparison_results(all_avg_metrics: list):
    """Menulis hasil perbandingan semua algoritma ke CSV."""
    if not all_avg_metrics:
        print("Tidak ada metrik untuk dibandingkan.", file=sys.stderr)
        return
    
    headers = ['algorithm', 'dataset', 'total_tasks', 'makespan', 'throughput', 'total_cpu_time', 
               'total_wait_time', 'avg_start_time', 'avg_exec_time', 'avg_finish_time',
               'avg_wait_time', 'imbalance_degree', 'resource_utilization']
    
    try:
        with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_avg_metrics)
        print(f"\n{'='*60}")
        print(f"Hasil perbandingan rata-rata disimpan ke {RESULTS_FILE}")
        print(f"{'='*60}")
    except IOError as e:
        print(f"Error menulis hasil perbandingan: {e}", file=sys.stderr)

def write_algorithm_summary(algorithm_name: str, all_metrics: list):
    """Menulis summary untuk satu algoritma dengan semua runs dan rata-ratanya."""
    if not all_metrics:
        return
    
    filename = os.path.join(DETAILED_RESULTS_DIR, f"{algorithm_name}_summary.csv")
    
    headers = ['dataset', 'iteration', 'total_tasks', 'makespan', 'throughput', 'total_cpu_time', 
               'total_wait_time', 'avg_start_time', 'avg_exec_time', 'avg_finish_time',
               'avg_wait_time', 'imbalance_degree', 'resource_utilization']
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            # Tulis semua runs
            for m in all_metrics:
                row = {k: m[k] for k in headers}
                writer.writerow(row)
            
            # Hitung dan tulis rata-rata per dataset
            datasets = sorted(set(m['dataset'] for m in all_metrics))
            for dataset in datasets:
                dataset_metrics = [m for m in all_metrics if m['dataset'] == dataset]
                avg = calculate_average_metrics(dataset_metrics)
                avg['iteration'] = 'AVERAGE'
                row = {k: avg[k] for k in headers}
                writer.writerow(row)
        
        print(f"Summary untuk {algorithm_name} disimpan ke {filename}")
    except IOError as e:
        print(f"Error menulis summary {algorithm_name}: {e}", file=sys.stderr)

# --- Fungsi Main ---

async def main():
    # 1. Inisialisasi
    vms = [VM(name, spec['ip'], spec['cpu'], spec['ram_gb']) 
            for name, spec in VM_SPECS.items()]
    
    # 2. Daftar algoritma yang akan dijalankan
    algorithms = [
        (
            'ACO_Standard',
            aco_standard,
            {
                'num_ants': 20,
                'num_iterations': 40,
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.2,
                'pheromone_deposit': 1.0
            }
        ),
        ('Stochastic_Hill_Climbing', stochastic_hill_climbing, {'iterations': 1000}),
        ('Round_Robin', round_robin, {}),
        ('FCFS', fcfs, {})
    ]
    
    all_avg_metrics = []  # Untuk menyimpan rata-rata per algoritma per dataset
    total_runs = len(algorithms) * len(DATASET_FILES) * ITERATIONS_PER_DATASET
    current_run = 0
    
    print(f"\n{'='*60}")
    print(f"Total Runs yang akan dieksekusi: {total_runs}")
    print(f"Algoritma: {len(algorithms)}, Dataset: {len(DATASET_FILES)}, Iterasi per Dataset: {ITERATIONS_PER_DATASET}")
    print(f"{'='*60}\n")
    
    # 3. Jalankan semua algoritma secara berurutan
    async with httpx.AsyncClient() as client:
        for algo_name, algo_func, algo_kwargs in algorithms:
            print(f"\n{'='*60}")
            print(f"Algoritma: {algo_name}")
            print(f"{'='*60}")
            
            algo_metrics = []  # Semua metrics untuk algoritma ini
            
            for dataset_file in DATASET_FILES:
                dataset_name = dataset_file.replace('.txt', '').replace(' ', '_')
                
                # Load tasks untuk dataset ini
                tasks = load_tasks(dataset_file)
                if not tasks:
                    print(f"Tidak ada tugas untuk dataset {dataset_file}. Skip.", file=sys.stderr)
                    continue
                
                tasks_dict = {task.id: task for task in tasks}
                vms_dict = {vm.name: vm for vm in vms}
                
                print(f"\n  Dataset: {dataset_file} ({len(tasks)} tasks)")
                dataset_metrics = []  # Metrics untuk dataset ini
                
                for iteration in range(1, ITERATIONS_PER_DATASET + 1):
                    current_run += 1
                    
                    # Cek apakah iterasi ini sudah pernah dijalankan
                    if check_iteration_exists(algo_name, dataset_name, iteration):
                        print(f"    Iterasi {iteration}/{ITERATIONS_PER_DATASET} (Run {current_run}/{total_runs})... ⏭ (Skip - sudah ada)")
                        
                        # Baca metrics dari file yang sudah ada untuk menghitung rata-rata
                        filename = os.path.join(DETAILED_RESULTS_DIR, f"{algo_name}_{dataset_name}_iter{iteration}_details.csv")
                        try:
                            df = pd.read_csv(filename)
                            if not df.empty:
                                # Rekonstruksi metrics dari file CSV
                                makespan = df['finish_time'].max()
                                total_tasks = len(df)
                                throughput = total_tasks / makespan if makespan > 0 else 0
                                
                                metrics = {
                                    'algorithm': algo_name,
                                    'dataset': dataset_name,
                                    'iteration': iteration,
                                    'total_tasks': total_tasks,
                                    'makespan': makespan,
                                    'throughput': throughput,
                                    'total_cpu_time': df['exec_time'].sum(),
                                    'total_wait_time': df['wait_time'].sum(),
                                    'avg_start_time': df['start_time'].mean(),
                                    'avg_exec_time': df['exec_time'].mean(),
                                    'avg_finish_time': df['finish_time'].mean(),
                                    'avg_wait_time': df['wait_time'].mean(),
                                    'imbalance_degree': 0.0,  # Tidak bisa dihitung dari CSV sederhana
                                    'resource_utilization': 0.0  # Tidak bisa dihitung dari CSV sederhana
                                }
                                algo_metrics.append(metrics)
                                dataset_metrics.append(metrics)
                        except Exception as e:
                            print(f" (Warning: Tidak bisa membaca file - {e})")
                        
                        continue
                    
                    print(f"    Iterasi {iteration}/{ITERATIONS_PER_DATASET} (Run {current_run}/{total_runs})...", end=' ')
                    
                    metrics = await run_algorithm(
                        algo_name, algo_func, tasks, vms, 
                        tasks_dict, vms_dict, client, dataset_name, iteration, **algo_kwargs
                    )
                    
                    if metrics:
                        algo_metrics.append(metrics)
                        dataset_metrics.append(metrics)
                        print(f"✓ (Makespan: {metrics['makespan']:.4f}s)")
                    else:
                        print("✗")
                
                # Hitung rata-rata untuk dataset ini
                if dataset_metrics:
                    avg_metrics = calculate_average_metrics(dataset_metrics)
                    all_avg_metrics.append(avg_metrics)
                    print(f"\n  Rata-rata {algo_name} pada {dataset_file}:")
                    print(f"    Makespan: {avg_metrics['makespan']:.4f}s")
                    print(f"    Throughput: {avg_metrics['throughput']:.4f} tasks/s")
                    print(f"    Resource Utilization: {avg_metrics['resource_utilization']:.4%}")
            
            # Tulis summary untuk algoritma ini
            if algo_metrics:
                write_algorithm_summary(algo_name, algo_metrics)
    
    # 4. Tulis hasil perbandingan (kesimpulan)
    write_comparison_results(all_avg_metrics)
    
    print(f"\n{'='*60}")
    print(f"Semua eksekusi selesai! Total runs: {current_run}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
