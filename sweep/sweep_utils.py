import subprocess
import re
import os
import string
### Utils ###
def gen_script(python_command):
    script = (
        "#!/bin/bash\n"
        "source ./load_env.sh\n"
        f"{python_command}"
    )
    return script

def get_compute_command(compute_type: str = "l40s", job_name: str = 'meltingpot_default', dependency_job_id: str = None):
    if compute_type == "l40s":
        base =  'sbatch  --time=48:0:0  --gres=gpu:l40s:1 --partition=long --cpus-per-task=12 --mem=256G'
    elif compute_type == "l40s_short":
        base = 'sbatch  --time=4:0:0  --gres=gpu:l40s:1 --partition=long --cpus-per-task=12 --mem=256G'
    elif compute_type == "l40s_eval":
        base = 'sbatch  --time=4:0:0  --gres=gpu:l40s:1 --partition=long --cpus-per-task=12 --mem=256G'
    elif compute_type == "a100_12h":
        base = 'sbatch  --time=12:0:0  --gres=gpu:a100l:1 --partition=long --cpus-per-task=6 --mem=64G'
    else:
        raise ValueError(f"Invalid Compute Type: {compute_type}")
    
    if not os.path.exists('./slurmout'):
        os.makedirs('./slurmout', exist_ok=True)
        
    base += ' --output=./slurmout/slurm-%j.out'
    
    command = base 
    if job_name:
        command += f" --job-name={job_name}"
    if dependency_job_id:
        command += f" --dependency=afterany:{dependency_job_id}"
    
    command += ' sweep_temp_job.sh'
    return command 


def submit_slurm_job(python_command, fake_submit: bool = True, compute_type: str = "l40s", dependency_job_id: str = None, job_name: str = None):
    script = gen_script(python_command)
    print('-'*100)
    print('##SCRIPT##\n', script)
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    job_id = None
    if fake_submit:
        print('##FAKE SUBMIT##')
        job_id = '123456789-FAKE-JOB-ID'
    else:
        compute_command = get_compute_command(compute_type, job_name, dependency_job_id)
        result = subprocess.run(compute_command, shell=True, capture_output=True, text=True)
        print(result.stdout.strip())
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)

    print('-'*100)
    return job_id

def strict_format(template, **kwargs):
    formatter = string.Formatter()
    expected_keys = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name}
    unexpected_keys = set(kwargs) - expected_keys
    if unexpected_keys:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")
    return template.format(**kwargs)
