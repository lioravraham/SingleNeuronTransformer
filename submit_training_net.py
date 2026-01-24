
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from slurm_job import SlurmJob

print("going to submit a job to run transformer encoder")

job = SlurmJob(
    job_name="training_transformer_encoder_layer5_david",
    job_folder = "/ems/elsc-labs/london-m/lior.avraham1/SingleNeuronTransformer/training_net_job_output3",
    mem = 80000,
    run_line = """
    #!/bin/bash -l
    cd /ems/elsc-labs/london-m/lior.avraham1/SingleNeuronTransformer
    source .venv/bin/activate
    python model.py
    """,  
    run_on_GPU=True,
    new_cluster=True,
    allow_ss_cluster=False
)

job.send()

print("job sent") 