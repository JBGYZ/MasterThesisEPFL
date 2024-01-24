import os
import re
import time


# seed_init = 2
# seed_net = 3
seed_net = 1
seed_rules = 1
seed_sample = 1
m = 6
num_features = m 
num_layers = 3
ptr_nb_list = [14000,] #12000, 10000, 7000, 5000]
pte_nb = 10000

layer_nb = 6 
nhead = 4
scaleup_dim = nhead * m 
dim_feedforward = 4 * scaleup_dim


for ptr_nb in ptr_nb_list:
    script="""python main.py --pickle /to/directory/{folder_name}.pkl --optim adam --device cuda --cls_token True --batch_size 128 \
    --dataset hier1 --m {m} --num_features {num_features} --num_layers {num_layers} --ptr {ptr_nb} --pte {pte_nb} \
    --net albert --net_layers {layer_nb} --nhead {nhead} --dim_feedforward {dim_feedforward} \
    --reducer_type linear --embedding_type scaleup --scaleup_dim {scaleup_dim} \
    --epochs 501 --scheduler none --pos_encoder_type learned --lr 0.0005 \
    --zero_loss_threshold 0.005 --zero_loss_epochs 10 --seed_net {seed_net} --seed_rules {seed_rules} --seed_sample {seed_sample} """.format(m =m, 
                                                                num_features=num_features,
                                                                num_layers=num_layers,
                                                                dim_feedforward = dim_feedforward,
                                                                ptr_nb=ptr_nb, 
                                                                layer_nb =layer_nb, 
                                                                nhead = nhead, 
                                                                scaleup_dim = scaleup_dim,
                                                                seed_net = seed_net, 
                                                                seed_rules = seed_rules,
                                                                seed_sample = seed_sample,
                                                                pte_nb = pte_nb,
                                                                folder_name="{folder_name}")

    pattern = r'--(ptr|net_layers|nhead|dim_feedforward|scaleup_dim|num_features|num_layers|lr|seed_net|seed_rules|seed_sample|scheduler)\s+([\w.]+)'
    # Use re.findall to extract matches
    matches = re.findall(pattern, script)
    # Create a dictionary to store the extracted arguments and their values
    arguments = dict(matches)
    # Concatenate the extracted arguments and their values into a string
    folder_name = "_".join([f"{arg}_{value}" for arg, value in arguments.items()])

    py_script = script.format(folder_name=folder_name)

    template = \
"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --chdir /to/directory
#SBATCH --output=/to/directory/{folder_name}.out

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB 
#SBATCH --time 7:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1

date
echo "job name: $SLURM_JOB_NAME"
echo "job id: $SLURM_JOB_ID"
echo "{script}"

conda activate mlepfl

{script}

date
"""


    script_content = template.format(job_name=folder_name, folder_name=folder_name, script=py_script)

    # Write the script to a file
    with open(f"{folder_name}_script.sh", "w") as script_file:
        script_file.write(script_content)

    # Submit the job using sbatch
    os.system(f"sbatch {folder_name}_script.sh")
    time.sleep(1)
    os.system(f"rm {folder_name}_script.sh")



