from sweep_utils import submit_slurm_job, get_compute_command, strict_format
### Jobs ###
BASE_RUN_ID = "2025_05_12"
BASE_PYTHON_COMMAND = "python nano_r1_script_vineppo_binary_simple.py" \
                      " --algorithm {algorithm}" \
                      " --run_id {run_id}"
                 
def get_grpo_job():
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'algorithm': 'grpo',
    }
    return python_command, python_kwargs

def get_vineppo_job(vineppo_k, vineppo_refinement_iterations):
    python_command = BASE_PYTHON_COMMAND + " --vineppo_k {vineppo_k} --vineppo_refinement_iterations {vineppo_refinement_iterations}"
    python_kwargs = {
        'algorithm': 'vineppo',
        'vineppo_k': vineppo_k,
        'vineppo_refinement_iterations': vineppo_refinement_iterations,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        'grpo': get_grpo_job(),
        'vineppo_1K': get_vineppo_job(1, 1),
        'vineppo_2K': get_vineppo_job(2, 1),
        'vineppo_3K': get_vineppo_job(3, 1),
        'vineppo_9K': get_vineppo_job(9, 1),
        'vineppo_3K_2iter': get_vineppo_job(3, 2),
        'vineppo_3K_3iter': get_vineppo_job(3, 3),
        'vineppo_3K_4iter': get_vineppo_job(3, 4),
    }
    
    return jobs


### Submit Jobs ###
def submit_job(job_names: str | list[str], fake_submit: bool = True, compute_type: str = "l40s"):
    jobs = get_jobs()
    if isinstance(job_names, str):
        # Handle both comma-separated strings and single job names
        job_names = [name.strip() for name in job_names.split(',')]
        
    for job_name in job_names:
        python_command, python_kwargs = jobs[job_name]
        python_kwargs['run_id'] = f'{BASE_RUN_ID}_{job_name}'
        
        formatted_python_command = strict_format(python_command, **python_kwargs)
        submit_slurm_job(formatted_python_command, fake_submit=fake_submit, compute_type=compute_type, job_name=job_name)
         
    
if __name__ == '__main__':
    import fire
    fire.Fire()
