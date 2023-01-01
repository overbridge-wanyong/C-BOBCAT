import subprocess
import argparse
import time

def create_parser():
    parser = argparse.ArgumentParser(description='Relaunch')
    parser.add_argument('--t', type=str,
                        default='5m', help='m or s or h')
    parser.add_argument('--v', action='store_true')
    parser.add_argument('--f', action='store_true')

    params = parser.parse_args()
    return params

if __name__ == "__main__":
    params = create_parser()
    time_now = time.time()
    mapper = {'s':1., 'm':60., 'h':3600}
    base_time = time_now - float(params.t[:-1])*mapper[params.t[-1]]
    node_time = time_now - 86000.
    with open('logs/exceptions.txt', 'r') as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line.split(',') for line in lines]
    exclude_nodes = list(set([line[1] for line in lines if float(line[2])>=node_time]))
    # line = params.name+','+params.nodes+','+str(time_now)+','+config_name +'\n'
    lines = [line for line in lines if float(line[2])>=base_time]

    if params.v:
        for line in lines:
            print(line[-1])#config name
        print('Number of Failed Jobs: ', len(lines))
        print('Number of Excluded Nodes: ', len(exclude_nodes))
    exclude_nodes =  ','.join(exclude_nodes)
    for line in lines:#--exclude=node078
        command = "sbatch --exclude=%s %s" % (exclude_nodes, line[-1])
        if params.f:
            print(subprocess.check_output(command, shell=True))
        elif params.v:
            print(command)
        
        

