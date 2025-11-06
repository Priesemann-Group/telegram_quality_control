def distribute_tasks(tasks, num_workers):
    """Distribute tasks among workers as equally as possible."""
    num_tasks = len(tasks)
    base_size = num_tasks // num_workers
    remainder = num_tasks % num_workers
    
    result = []
    start = 0
    for i in range(num_workers):
        # First 'remainder' workers get one extra task
        size = base_size + (1 if i < remainder else 0)
        result.append(tasks[start:start + size])
        start += size

    return result