import random
from collections import namedtuple

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])
MIN_PHEROMONE = 1e-6

# --- Helper Function ---

def calculate_estimated_makespan(solution: dict, tasks_dict: dict, vms_dict: dict) -> float:
    """
    Fungsi Biaya (Cost Function).
    Memperkirakan makespan (waktu selesai maks) untuk solusi tertentu.
    Model sederhana: makespan = max(total_beban_cpu_vm / core_vm)
    """
    vm_loads = {vm.name: 0.0 for vm in vms_dict.values()}
    
    for task_id, vm_name in solution.items():
        task = tasks_dict[task_id]
        vm = vms_dict[vm_name]
        
        # Estimasi waktu eksekusi: beban / jumlah core
        estimated_time = task.cpu_load / vm.cpu_cores
        vm_loads[vm_name] += estimated_time
        
    # Makespan adalah VM yang paling lama selesai
    return max(vm_loads.values())

# --- Standard Ant Colony Optimization (ACO) ---

def _initialize_pheromone(tasks: list[Task], vms: list[VM], initial: float = 1.0) -> dict:
    """Membangun matriks pheromone awal untuk setiap kombinasi task-VM."""
    return {task.id: {vm.name: initial for vm in vms} for task in tasks}

def _initialize_heuristic(tasks: list[Task], vms: list[VM]) -> dict:
    """Matriks heuristic: semakin besar core VM dibanding beban task semakin baik."""
    heuristic = {}
    for task in tasks:
        heuristic[task.id] = {}
        for vm in vms:
            score = vm.cpu_cores / task.cpu_load if task.cpu_load else 0.0
            heuristic[task.id][vm.name] = max(score, MIN_PHEROMONE)
    return heuristic

def _choose_vm(task_id: int, vm_names: list[str], pheromone: dict, heuristic: dict,
               alpha: float, beta: float) -> str:
    """Memilih VM dengan roulette wheel berdasarkan pheromone dan heuristic."""
    desirabilities = []
    for vm_name in vm_names:
        tau = pheromone[task_id][vm_name] ** alpha
        eta = heuristic[task_id][vm_name] ** beta
        desirabilities.append(tau * eta)

    total = sum(desirabilities)
    if total <= 0:
        return random.choice(vm_names)

    pick = random.random() * total
    cumulative = 0.0
    for vm_name, desirability in zip(vm_names, desirabilities):
        cumulative += desirability
        if pick <= cumulative:
            return vm_name
    return vm_names[-1]

def aco_standard(
    tasks: list[Task],
    vms: list[VM],
    num_ants: int = 20,
    num_iterations: int = 50,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.1,
    pheromone_deposit: float = 1.0
) -> dict:
    """
    Implementasi standar ACO untuk penjadwalan task -> VM.
    Pheromone diperbarui per iterasi menggunakan solusi terbaik iterasi.
    """
    print(f"Memulai ACO Standard Scheduling ({num_ants} semut, {num_iterations} iterasi)...")

    if not tasks or not vms:
        return {}

    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_names = [vm.name for vm in vms]

    pheromone = _initialize_pheromone(tasks, vms)
    heuristic = _initialize_heuristic(tasks, vms)

    best_solution = None
    best_cost = float('inf')

    for iteration in range(num_iterations):
        iteration_best_solution = None
        iteration_best_cost = float('inf')

        for _ in range(num_ants):
            assignment = {}
            for task in tasks:
                chosen_vm = _choose_vm(task.id, vm_names, pheromone, heuristic, alpha, beta)
                assignment[task.id] = chosen_vm

            cost = calculate_estimated_makespan(assignment, tasks_dict, vms_dict)

            if cost < iteration_best_cost:
                iteration_best_cost = cost
                iteration_best_solution = assignment

            if cost < best_cost:
                best_cost = cost
                best_solution = assignment

        # Evaporasi pheromone
        for task_id in pheromone:
            for vm_name in pheromone[task_id]:
                pheromone[task_id][vm_name] = max(
                    (1 - evaporation_rate) * pheromone[task_id][vm_name],
                    MIN_PHEROMONE
                )

        # Deposit pheromone berdasarkan solusi terbaik iterasi
        if iteration_best_solution:
            deposit = pheromone_deposit / (iteration_best_cost + MIN_PHEROMONE)
            for task_id, vm_name in iteration_best_solution.items():
                pheromone[task_id][vm_name] += deposit

        if num_iterations <= 10 or (iteration + 1) % max(1, num_iterations // 10) == 0:
            print(f"  Iterasi {iteration + 1}/{num_iterations} - Estimasi makespan terbaik: {best_cost:.2f}")

    print(f"ACO selesai. Estimasi makespan terbaik: {best_cost:.2f}")
    return best_solution if best_solution else {task.id: random.choice(vm_names) for task in tasks}

# --- Enhanced Min-Min Algorithm ---

def enhanced_min_min(tasks: list[Task], vms: list[VM]) -> dict:
    """
    Enhanced Min-Min Scheduling Algorithm (EMin-Min)
    - Menggunakan load balancing improvement dari Min-Min klasik
    """
    print("Memulai Enhanced Min-Min Scheduling...")

    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}

    unassigned_tasks = tasks.copy()
    ready_time = {vm.name: 0.0 for vm in vms}

    assignment = {}

    # Threshold: untuk memisahkan antara short dan long tasks
    avg_load = sum(t.cpu_load for t in tasks) / len(tasks)
    threshold_ratio = 0.5  # rasio tweak, bisa disesuaikan
    threshold = avg_load * threshold_ratio

    while unassigned_tasks:
        # Hitung completion time (CTij) untuk semua task-VM
        ct_matrix = {}
        for task in unassigned_tasks:
            ct_matrix[task.id] = {}
            for vm in vms:
                et = task.cpu_load / vm.cpu_cores
                ct = ready_time[vm.name] + et
                ct_matrix[task.id][vm.name] = ct

        # Tentukan task terbaik (dengan CTmin)
        task_min_ct = {}
        for task in unassigned_tasks:
            vm_best = min(ct_matrix[task.id], key=ct_matrix[task.id].get)
            ct_best = ct_matrix[task.id][vm_best]
            task_min_ct[task.id] = (vm_best, ct_best)

        # Enhanced part:
        # Pilih antara short-task-priority atau long-task-priority tergantung beban rata-rata
        short_tasks = [tid for tid, (vm, ct) in task_min_ct.items()
                       if tasks_dict[tid].cpu_load <= threshold]
        long_tasks = [tid for tid, (vm, ct) in task_min_ct.items()
                      if tasks_dict[tid].cpu_load > threshold]

        # Ambil task dengan CTmin tergantung kategori
        if short_tasks:
            # short tasks diberi prioritas lebih dulu
            selected_task_id = min(short_tasks, key=lambda tid: task_min_ct[tid][1])
        else:
            selected_task_id = min(task_min_ct.keys(), key=lambda tid: task_min_ct[tid][1])

        selected_vm, selected_ct = task_min_ct[selected_task_id]

        # Update hasil penugasan
        assignment[selected_task_id] = selected_vm
        selected_task = tasks_dict[selected_task_id]
        et = selected_task.cpu_load / vms_dict[selected_vm].cpu_cores
        ready_time[selected_vm] += et

        # Hapus dari daftar unassigned
        unassigned_tasks = [t for t in unassigned_tasks if t.id != selected_task_id]

    makespan = calculate_estimated_makespan(assignment, tasks_dict, vms_dict)
    print(f"Enhanced Min-Min selesai. Estimasi Makespan: {makespan:.2f}")

    return assignment

# --- Stochastic Hill Climbing Algorithm ---

def get_random_neighbor(solution: dict, vm_names: list) -> dict:
    """
    Membuat solusi 'tetangga' dengan memindahkan satu tugas acak
    ke VM acak yang berbeda.
    """
    new_solution = solution.copy()
    
    # Pilih tugas acak untuk dipindah
    task_id_to_move = random.choice(list(new_solution.keys()))
    current_vm = new_solution[task_id_to_move]
    
    # Pilih VM baru (pastikan berbeda)
    possible_new_vms = [vm for vm in vm_names if vm != current_vm]
    if not possible_new_vms:
        return new_solution # Terjadi jika hanya ada 1 VM
        
    new_vm = random.choice(possible_new_vms)
    
    # Pindahkan tugas
    new_solution[task_id_to_move] = new_vm
    return new_solution

def stochastic_hill_climbing(tasks: list[Task], vms: list[VM], iterations: int = 1000) -> dict:
    """Menjalankan algoritma SHC untuk menemukan solusi (penugasan) terbaik."""
    
    print(f"Memulai Stochastic Hill Climbing ({iterations} iterasi)...")
    
    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_names = list(vms_dict.keys())

    # 1. Buat Solusi Awal (Acak)
    current_solution = {}
    for task in tasks:
        current_solution[task.id] = random.choice(vm_names)
        
    current_cost = calculate_estimated_makespan(current_solution, tasks_dict, vms_dict)
    
    best_solution = current_solution
    best_cost = current_cost
    
    print(f"Estimasi Makespan Awal (Acak): {best_cost:.2f}")

    # 2. Iterasi SHC
    for i in range(iterations):
        # Buat tetangga (neighbor)
        neighbor_solution = get_random_neighbor(current_solution, vm_names)
        neighbor_cost = calculate_estimated_makespan(neighbor_solution, tasks_dict, vms_dict)
        
        # 3. Bandingkan - hanya menerima yang lebih baik
        if neighbor_cost < current_cost:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            
            # Perbarui solusi terbaik jika ditemukan
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution
                print(f"Iterasi {i}: Estimasi Makespan Baru: {best_cost:.2f}")

    print(f"SHC Selesai. Estimasi Makespan Terbaik: {best_cost:.2f}")
    return best_solution

# --- Round Robin Algorithm ---

def round_robin(tasks: list[Task], vms: list[VM]) -> dict:
    """
    Menugaskan tugas secara bergiliran ke setiap VM.
    """
    
    print("Memulai Round Robin Scheduling...")
    
    assignment = {}
    vm_names = [vm.name for vm in vms]
    
    for i, task in enumerate(tasks):
        vm_index = i % len(vm_names)
        assignment[task.id] = vm_names[vm_index]
    
    print(f"Round Robin Selesai. {len(tasks)} tugas ditugaskan ke {len(vms)} VMs.")
    return assignment

# --- First Come First Serve (FCFS) Algorithm ---

def fcfs(tasks: list[Task], vms: list[VM]) -> dict:
    """
    Menugaskan tugas secara berurutan ke VM dengan beban terendah saat ini.
    FCFS dengan load balancing sederhana.
    """
    
    print("Memulai FCFS Scheduling...")
    
    vms_dict = {vm.name: vm for vm in vms}
    assignment = {}
    vm_loads = {vm.name: 0.0 for vm in vms}
    
    # Proses tugas sesuai urutan (FCFS)
    for task in tasks:
        # Pilih VM dengan beban terendah
        min_load_vm = min(vm_loads, key=vm_loads.get)
        assignment[task.id] = min_load_vm
        
        # Update beban VM
        vm = vms_dict[min_load_vm]
        estimated_time = task.cpu_load / vm.cpu_cores
        vm_loads[min_load_vm] += estimated_time
    
    print(f"FCFS Selesai. {len(tasks)} tugas ditugaskan ke {len(vms)} VMs.")
    return assignment
