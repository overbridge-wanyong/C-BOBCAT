import itertools
import collections
import glob
import os
import datetime
import subprocess
import string
import sys

def safe_makedirs(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass

UNITY_CONSTRAINTS = '#SBATCH --constraint=""'
UNITY_BASE = "/gypsum/scratch1/arighosh/catext"
GYPSUM_BASE = "/mnt/nfs/scratch1/arighosh/catext"
UNITY_PATHS = 'module load python/3.9.1\nmodule load cuda/10.2.89\ncd {}\nsource /old/home/arighosh_umass_edu/.venv/catext/bin/activate'.format(UNITY_BASE)
GYPSUM_PATHS = 'module load python3/current\ncd {}\nsource ~/linkedin/bin/activate'.format(GYPSUM_BASE)

create_dirs = ['logs/', 'slurm/', 'configs/']
for c in create_dirs:
    safe_makedirs(c)

def get_memory(combo):
    if combo['dataset']=='ednet':
        return 60000
    return 40000

def get_cpu(combo):
    if combo['dataset']=='ednet':
        return 4
    return 4

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def get_run_id():
    filename = "logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts)
    return run_id

    
def is_long(combo):
    return 'long'
#['mapt-read','mapt-math','eedi-1','eedi-3','junyi', 'ednet', 'assist2009']
save = False
fixed_params = '   '.join(['--neptune', '--cuda', '--gumbel'])
hyperparameters = [
    [('dataset',), ['ednet', 'assist2009','eedi-1']]
    #[('dataset',), ['mapt-read','mapt-math','eedi-1','eedi-3','junyi', 'ednet', 'assist2009']]
    #,[('model',), ['biirt-active', 'biirt-random', 'biirt-unbiased','biirt-biased', 'binn-active', 'binn-random', 'binn-unbiased','binn-biased']]
    ,[('model',), [ 'binn-biased', 'biirt-biased']]
    ,[('fold',), [ 1 ]]
    ,[('lamda',), [0.01, 0.03,0.1, 0.3 ]]#0.001,0.003,0.01, 0.03, 0.1
    ,[('hidden_dim'), [256]]
    ,[('lr',), [ 1e-3 ]]
    ,[('inner_lr',), [ 2e-1, 1e-1, 5e-2]]#1e-1,
    ,[('meta_lr',), [ 1e-4 ]]
    ,[('inner_loop',), [ 5 ]]
    ,[('policy_lr',), [  5e-4]]#2e-3, 2e-4
    ,[('n_query',), [2,4,8]]
    ,[('fixed_params',), [fixed_params]]
    ,[('cluster',), ['unity']]
]

def get_gpu(combo):
    if combo['model'] == 'binn-biased':
        v = '2080ti'
    elif combo['model'] == 'biirt-biased':
        v = '1080ti'
    elif combo['model'] in {'biirt-unbiased',  'binn-unbiased'}:
        v = 'm40'
    else:
        v = "titanx"
    
    if combo['cluster']=='gypsum':
        return v +'-long'
    return 'gpu-long'

    
def is_valid(combo):
    if ('random' in combo['model'] or 'active' in combo['model']) and combo['policy_lr']!=5e-4:
        return False
    if combo['lamda']!=0. and '-biased' not in combo['model']:
        return False
    return True

def get_constraints(combo):
    if combo['cluster']=='unity':
        return UNITY_CONSTRAINTS
    return ""

def get_paths(combo):
    if combo['cluster']=='unity':
        return UNITY_PATHS
    return GYPSUM_PATHS

def get_base_path(combo):
    return UNITY_BASE if combo['cluster'] =='unity' else GYPSUM_BASE

other_dependencies = {'gpu': get_gpu, 'memory': get_memory, 'n_cpu':get_cpu, 'valid':is_valid, 'long':is_long, 'constraints':get_constraints, 'paths':get_paths, 'base_path' :get_base_path}

run_id = int(get_run_id())

key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []
gpu_counts =collections.defaultdict(int)

for combo in combinations:
    # Write the scheduler scripts
    template_name = "template.sh"
    with open(template_name, 'r') as f:
        schedule_script = f.read()

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    for k, v in other_dependencies.items():
         combo[k] = v(combo)
    if not combo['valid']:
        #print(combo)
        continue
    combo['run_id'] = run_id
    gpu_counts[combo['gpu']] +=1

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))
    


    schedule_script += "\n"

    # Write schedule script
    script_name = 'configs/cat_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)
    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name +", Time Now= "+ datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" 
    with open("logs/expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1

print(gpu_counts)
# schedule jobs
excludes =  "--exclude=node128,node097,node094,node095" if combo['cluster']=='gypsum' else "--exclude="
for script in scripts:#--exclude=node078
    command = "sbatch {} {}".format(excludes,script)
    # print(command)
    print(subprocess.check_output(command, shell=True))