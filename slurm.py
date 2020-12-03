import os
from typing import List


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
