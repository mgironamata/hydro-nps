def task_to_device(task,device='cuda'):
    for k in task.keys():
        task[k] = task[k].to(device)
    return task

def loaded_task(task, device='cuda'):
    """Reshapes task to the right shape, and loads it onto the right device.
    
    """

    if len(task['x'].shape)==4:
        for k in task.keys():
            if task[k].device==device:
                task[k] = task[k].reshape(task[k].shape[1:])
            else:
                task[k] = task[k].reshape(task[k].shape[1:]).to(device)
    else:
        for k in task.keys():
            if task[k].device==device:
                pass
            else:
                task[k] = task[k].to(device)
    return task

def try_to_delete(var_list):
    for var in var_list:
        if var in globals():
            print(f"{var} deleted")
            del var
        else:
            print(f"{var} does not exist")