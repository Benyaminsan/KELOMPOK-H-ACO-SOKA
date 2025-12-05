def first_come_first_serve(tasks, vms):
    """
    FCFS (First Come, First Served):
    - Task pertama masuk ke VM pertama,
    - Task kedua juga ke VM pertama,
    - dan seterusnya.
    - Jadi *semua task* dijadwalkan ke VM pertama.
    """
    vm_name = vms[0].name  # VM pertama
    
    solution = {}
    for task in tasks:
        solution[task.id] = vm_name

    return solution

fcfs_scheduler = first_come_first_serve