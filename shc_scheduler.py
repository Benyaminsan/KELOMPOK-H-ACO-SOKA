import random
import copy

def stochastic_hill_climbing(tasks, vms, iterations=1000):
    """
    Stochastic Hill Climbing untuk mapping task -> VM.
    tasks : list of Task (namedtuple) atau list of numeric loads
    vms   : list of VM (namedtuple) — fungsi akan memakai vm indices dan vm.name di hasil
    """

    num_tasks = len(tasks)
    num_vms = len(vms)

    # --- 1. Generate solusi awal random ---
    current_solution = [random.randint(0, num_vms - 1) for _ in range(num_tasks)]
    
    def get_task_load(task_or_value):
        # support either Task (with cpu_load) or numeric load
        if hasattr(task_or_value, "cpu_load"):
            return float(task_or_value.cpu_load)
        return float(task_or_value)

    def cost(solution):
        """Cost = selisih load antar VM (semakin rata semakin baik)"""
        load = [0.0] * num_vms
        for task_idx, vm_idx in enumerate(solution):
            load[vm_idx] += get_task_load(tasks[task_idx])

        # spread lebih kecil = lebih baik
        return max(load) - min(load)

    current_cost = cost(current_solution)

    # --- 2. Iterasi perbaikan lokal ---
    for _ in range(iterations):
        candidate = copy.deepcopy(current_solution)
        t = random.randint(0, num_tasks - 1)           # pilih task acak
        candidate[t] = random.randint(0, num_vms - 1)  # pindahkan ke VM acak

        c_cost = cost(candidate)
        if c_cost < current_cost:
            current_solution = candidate
            current_cost = c_cost

    # Convert ke format mapping: { task.id: vm_name }
    mapping = {}
    for task_idx in range(num_tasks):
        task = tasks[task_idx]
        vm = vms[current_solution[task_idx]]
        # if Task has id attribute use it; otherwise use index
        key = getattr(task, "id", task_idx)
        mapping[key] = vm.name

    return mapping