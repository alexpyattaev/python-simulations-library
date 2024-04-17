import os
from typing import List
import subprocess


def slurm_cores_available(partition: str = "") -> int:
    try:
        cmd = 'sinfo -h -o %C'
        if partition:
            cmd += f" --partition {partition}"
        x = subprocess.check_output(cmd.split(), shell=False)
        return int(x.decode('ASCII').strip().split('/')[-1])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def create_sbatch_file(job_name: str, args: List[str], log_dir: str, ntasks: int = 1,
                       max_time: str = "00:00:00") -> str:
    """
    Creates sbatch script file
    :param job_name:
        Name of the job
    :param log_dir:
        Log diectory name
    :param args:
        Args for srun
    :param ntasks:
        Number of tasks for the batch
    :param max_time:
        Time requested from the CPU
    :return:
        Created file name.
    """
    filename = "jobs/" + job_name + ".sh"
    argstring = " ".join(args)
    output = os.path.join(log_dir, f"{job_name}.out")
    contents = f"#!/bin/bash\n" \
               f"#SBATCH --job-name={job_name}\n" \
               f"#SBATCH --output={output}\n" \
               f"#SBATCH --ntasks={ntasks}\n" \
               f"#SBATCH --time={max_time}\n" \
               f"srun {argstring}\n"

    with open(filename, 'w+') as f:
        f.seek(0)
        f.write(contents)
        print(f"Created job file {filename}")
    return filename
