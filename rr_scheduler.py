def round_robin_scheduler(tasks, vms):
    """
    Round Robin: Task 0 → VM 0, Task 1 → VM 1, ..., repeat.
    Return: dict { task_id: vm_name } for compatibility.
    """
    num_vms = len(vms)
    mapping = {}
    for i, _ in enumerate(tasks):
        vm_idx = i % num_vms
        task = tasks[i]
        task_id = getattr(task, "id", i)
        mapping[task_id] = vms[vm_idx].name
    return mapping
