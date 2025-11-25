"""
ACO-based scheduler module: aco_scheduler.py

Usage:
- Place this file next to your existing scheduler.py
- Replace `from shc_algorithm import stochastic_hill_climb` with:
    from aco_scheduler import ant_colony_optimization

Function exported:
- ant_colony_optimization(tasks, vms, iterations=200, ants=50, alpha=1.0, beta=2.0,
                          evaporation=0.5, q=100.0, seed=None)
  returns dict mapping task.id -> vm.name

The module expects the same Task and VM namedtuples as your existing code:
    VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
    Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

Notes:
- The ACO implemented here is a straightforward single-objective ACO that tries to
  minimize a fitness composed of (cost + makespan*scale + imbalance*scale).
- Heuristic uses VM "speed" estimated from cpu_cores; you can adapt to real MIPS
  if available.

This implementation is written to be clear and easy to integrate into your
asynchronous executor script.
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
from collections import defaultdict
import math
import random
from typing import List, Dict, Tuple

# Public API

def ant_colony_optimization(tasks, vms,
                             iterations: int = 200,
                             ants: int = 40,
                             alpha: float = 1.0,
                             beta: float = 2.0,
                             evaporation: float = 0.5,
                             q: float = 100.0,
                             seed: int = None) -> Dict[int, str]:
    """
    Run ACO to find a mapping from task.id -> vm.name.

    Parameters
    - tasks: list of Task namedtuple (id, name, index, cpu_load)
    - vms: list of VM namedtuple (name, ip, cpu_cores, ram_gb)
    - iterations, ants: ACO control
    - alpha: pheromone influence
    - beta: heuristic influence
    - evaporation: pheromone evaporation rate (0..1)
    - q: pheromone deposit constant
    - seed: random seed for reproducibility

    Returns
    - mapping dict { task_id: vm_name }
    """
    rnd = random.Random(seed)

    num_tasks = len(tasks)
    num_vms = len(vms)

    if num_tasks == 0 or num_vms == 0:
        return {}

    # Build index maps for convenience
    vm_index_by_name = {vm.name: i for i, vm in enumerate(vms)}

    # Initialize pheromone matrix (task x vm)
    initial_pheromone = 0.1
    pheromone = [[initial_pheromone for _ in range(num_vms)] for _ in range(num_tasks)]

    # Precompute simple heuristic: faster VM (more cores) preferred for large cpu_load
    # We'll use heuristic = vm_speed / cpu_load_estimate (so larger cpu_load favors faster VM)
    # vm_speed: proportional to cpu_cores (tweak if you have MIPS)
    vm_speed = [max(1.0, float(vm.cpu_cores)) for vm in vms]

    # Helper: estimate exec time of a task on vm (seconds) -- adjustable
    def estimate_exec_time(task_cpu_load: float, vm_speed_value: float) -> float:
        # Treat vm_speed_value as pseudo-MIPS (higher = faster)
        # Larger constant scales can be tuned to match real world
        # Make denominator non-zero
        return task_cpu_load / (vm_speed_value * 1000.0)

    # Fitness function: lower is better
    def fitness_of_mapping(mapping: List[int]) -> float:
        # mapping is list of vm indices per task index order
        vm_loads = [0.0 for _ in range(num_vms)]
        vm_costs = [0.0 for _ in range(num_vms)]

        for t_idx, vm_idx in enumerate(mapping):
            task = tasks[t_idx]
            exec_time = estimate_exec_time(task.cpu_load, vm_speed[vm_idx])
            # simplified cost ~ exec_time * cost_rate (cost_rate is assumed constant)
            cost_rate = 0.001
            cost = exec_time * cost_rate
            vm_loads[vm_idx] += exec_time
            vm_costs[vm_idx] += cost

        makespan = max(vm_loads) if vm_loads else 0.0
        total_cost = sum(vm_costs)

        # load imbalance = stddev of vm_loads
        avg_load = sum(vm_loads) / len(vm_loads) if vm_loads else 0.0
        variance = sum((l - avg_load) ** 2 for l in vm_loads) / len(vm_loads) if vm_loads else 0.0
        imbalance = math.sqrt(variance)

        # Weighted sum (tunable)
        cost_w = 0.4
        makespan_w = 0.4
        imbalance_w = 0.2

        # scale makespan to be comparable to cost (because exec_time numbers are small)
        makespan_scale = 100.0
        imbalance_scale = 50.0

        fitness = (cost_w * total_cost) + (makespan_w * makespan * makespan_scale) + (imbalance_w * imbalance * imbalance_scale)
        return fitness

    # Utility: build a solution by one ant
    def construct_solution_for_ant(rnd_local: random.Random) -> Tuple[List[int], float]:
        solution = [-1] * num_tasks
        # For each task, choose a VM based on pheromone and heuristic
        for t_idx, task in enumerate(tasks):
            # compute probabilities
            probs = []
            total_prob = 0.0
            for v_idx in range(num_vms):
                # heuristic: faster vm -> higher heuristic. Also larger tasks benefit more.
                # We use heuristic = vm_speed / task_cpu_load (so bigger tasks emphasize speed)
                h = vm_speed[v_idx] / max(1.0, task.cpu_load)
                val = (pheromone[t_idx][v_idx] ** alpha) * (h ** beta)
                probs.append(val)
                total_prob += val

            if total_prob == 0:
                # uniform choice
                chosen = rnd_local.randrange(num_vms)
                solution[t_idx] = chosen
            else:
                # roulette wheel
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

    # Main loop
    best_solution = None
    best_fitness = float('inf')

    for iteration in range(iterations):
        ants_solutions = []  # tuples (solution list, fitness)

        for a in range(ants):
            sol, fit = construct_solution_for_ant(rnd)
            ants_solutions.append((sol, fit))
            if fit < best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

        # Evaporate pheromone
        for t in range(num_tasks):
            for v in range(num_vms):
                pheromone[t][v] *= (1.0 - evaporation)
                # keep minimal pheromone to avoid zeros
                if pheromone[t][v] < 1e-6:
                    pheromone[t][v] = 1e-6

        # Deposit pheromone for each ant proportional to quality
        for sol, fit in ants_solutions:
            # avoid division by zero
            if fit <= 0:
                delta = q
            else:
                delta = q / fit
            for t_idx, v_idx in enumerate(sol):
                pheromone[t_idx][v_idx] += delta

    # Convert best_solution (list of vm indices per task-order) to mapping by task.id
    if best_solution is None:
        return {}

    assignment = { tasks[t_idx].id: vms[best_solution[t_idx]].name for t_idx in range(num_tasks) }
    return assignment


# If run standalone, provide small demo
if __name__ == '__main__':
    from collections import namedtuple
    VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb'])
    Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

    # quick demo tasks
    demo_tasks = [Task(i, f't{i}', i % 5 + 1, (i % 5 + 1) ** 2 * 1000) for i in range(10)]
    demo_vms = [VM('vm1', '127.0.0.1', 1, 1), VM('vm2', '127.0.0.2', 2, 2), VM('vm3', '127.0.0.3', 4, 4)]

    mapping = ant_colony_optimization(demo_tasks, demo_vms, iterations=100, ants=20, seed=42)
    print('Demo mapping (task_id -> vm_name):')
    for k, v in mapping.items():
        print(f'  {k} -> {v}')

# ...existing code...
# Integrated version for scheduler.py

from aco_scheduler import ant_colony_optimization

# Replace SHC call:
# best_assignment = stochastic_hill_climb(tasks, vms, SHC_ITERATIONS)

 # With ACO call:
 best_assignment = ant_colony_optimization(tasks, vms, iterations=300, ants=50, alpha=1.0, beta=2.0, evaporation=0.4, q=120.0)
 # The ACO implementation is defined above in this same file.
 # Replace SHC call:
 # best_assignment = stochastic_hill_climb(tasks, vms, SHC_ITERATIONS)

 # With ACO call (use the local function defined earlier)
 best_assignment = ant_colony_optimization(tasks, vms,
                                         iterations=300,
                                         ants=50,
                                         alpha=1.0,
                                         beta=2.0,
                                         evaporation=0.4,
                                         q=120.0)
# ...existing code...