import random
import math
from collections import namedtuple

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load', 'ram_mb'])

# -----------------------------------------------------------
# Fungsi Cost: Estimasi Makespan
# -----------------------------------------------------------

def calculate_estimated_makespan(solution: dict, tasks_dict: dict, vms_dict: dict) -> float:
    vm_loads = {vm.name: 0.0 for vm in vms_dict.values()}

    for task_id, vm_name in solution.items():
        task = tasks_dict[task_id]
        vm = vms_dict[vm_name]
        estimated_time = task.cpu_load / vm.cpu_cores  
        vm_loads[vm_name] += estimated_time

    return max(vm_loads.values())


# -----------------------------------------------------------
# ACO FUNCTIONS
# -----------------------------------------------------------

def roulette_wheel_selection(probabilities):
    """Memilih index berdasarkan probabilitas (list float)."""
    r = random.random()
    cumulative = 0.0
    for index, prob in enumerate(probabilities):
        cumulative += prob
        if r <= cumulative:
            return index
    return len(probabilities) - 1  # fallback


def ant_construct_solution(tasks, vm_names, pheromone, heuristic):
    """Semut membangun solusi task → VM."""
    solution = {}

    for task_idx, task in enumerate(tasks):
        pher = pheromone[task_idx]
        heur = heuristic[task_idx]

        # Hitung probabilitas
        pher_pow = [p ** 1.2 for p in pher]
        heur_pow = [h ** 2 for h in heur]
        scores = [pher_pow[i] * heur_pow[i] for i in range(len(vm_names))]

        total = sum(scores)
        probabilities = [s / total for s in scores]

        chosen_vm_index = roulette_wheel_selection(probabilities)
        solution[task.id] = vm_names[chosen_vm_index]

    return solution


# -----------------------------------------------------------
# ACO MAIN ALGORITHM
# -----------------------------------------------------------

def aco_task_scheduling(tasks: list[Task], vms: list[VM],
                        num_ants=20, iterations=50, evaporation_rate=0.2):
    print(f"Menjalankan Ant Colony Optimization ({iterations} iterasi, {num_ants} semut)...")

    vms_dict = {vm.name: vm for vm in vms}
    tasks_dict = {task.id: task for task in tasks}
    vm_names = list(vms_dict.keys())

    num_tasks = len(tasks)
    num_vms = len(vm_names)

    # --------------------------
    # Inisialisasi Pheromone
    # --------------------------
    pheromone = [[1.0 for _ in range(num_vms)] for _ in range(num_tasks)]

    # --------------------------
    # Hitung Heuristic
    # η = 1 / execution_time
    # --------------------------
    heuristic = []
    for task in tasks:
        row = []
        for vm in vms:
            time = task.cpu_load / vm.cpu_cores
            row.append(1.0 / time)
        heuristic.append(row)

    # --------------------------
    # ACO Iterations
    # --------------------------
    best_solution = None
    best_cost = float('inf')

    for iter in range(iterations):
        ants_solutions = []
        ants_costs = []

        # Semut menghasilkan solusi
        for _ in range(num_ants):
            solution = ant_construct_solution(tasks, vm_names, pheromone, heuristic)
            cost = calculate_estimated_makespan(solution, tasks_dict, vms_dict)

            ants_solutions.append(solution)
            ants_costs.append(cost)

        # Cari solusi terbaik dalam iterasi ini
        iter_best_cost = min(ants_costs)
        iter_best_solution = ants_solutions[ants_costs.index(iter_best_cost)]

        if iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_solution = iter_best_solution
            print(f"Iterasi {iter}: Best Makespan = {best_cost:.3f}")

        # --------------------------
        # UPDATE PHEROMONE
        # --------------------------
        # Evaporasi
        for i in range(num_tasks):
            for j in range(num_vms):
                pheromone[i][j] *= (1 - evaporation_rate)

        # Deposit Pheromone oleh solusi terbaik
        deposit = 1.0 / iter_best_cost
        for task_id, vm_name in iter_best_solution.items():
            task_idx = next(i for i, t in enumerate(tasks) if t.id == task_id)
            vm_idx = vm_names.index(vm_name)
            pheromone[task_idx][vm_idx] += deposit

    print(f"\nACO Selesai. Best Makespan: {best_cost:.4f}")
    return best_solution
