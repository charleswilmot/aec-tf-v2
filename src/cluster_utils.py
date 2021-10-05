import os
from paramiko import SSHClient, RSAKey, AutoAddPolicy
from getpass import getpass


def get_n_free_cpus(node):
    cpusstate = os.popen('sinfo -h --nodes {} -O cpusstate'.format(node)).read()
    cpusstate = cpusstate.split("/")
    return int(cpusstate[1])


def get_free_mem(node):
    meminfo = os.popen('sinfo -h --nodes {} -O memory,allocmem'.format(node)).read()
    memory, allocated_memory = [int(s) for s in meminfo.split() if s.isdigit()]
    return memory - allocated_memory


def get_n_free_gpus(node):
    total = os.popen("sinfo -h -p sleuths -n {} -O gres".format(node)).read()
    total = int(total.split(":")[-1])
    used = os.popen("squeue -h -w {} -O gres".format(node)).read()
    used = sum([int(x.strip()[-1]) if x.strip() != "gpu" else 1 for x in used.strip().split() if x != "(null)" and x != "N/A"])
    return total - used


def node_list_availability(node_list, min_cpus=8, min_free_mem=20000):
    for node in node_list:
        n_free_cpus = get_n_free_cpus(node)
        free_mem = get_free_mem(node)
        n_free_gpus = get_n_free_gpus(node)
        if n_free_cpus >= min_cpus and free_mem >= min_free_mem and n_free_gpus >= 1:
            print(node, end=" -> ")
            return True
    return False


def get_partition_reservation():
    # OPTION 2
    print("checking OPTION 2 ... ", end="")
    if node_list_availability(["jetski"]): # , "speedboat"
        print("free space available, sending job")
        return "sleuths", "triesch-shared"
    print("no free space")
    # OPTION 1
    print("checking OPTION 1 ... ", end="")
    if node_list_availability(["turbine"]): # , "vane"
        print("free space available, sending job")
        return "sleuths", None
    print("no free space")
    print("No space available on the cluster. Defaulting to turbine/vane OPTION 1")
    return "sleuths", None
    # print("No space available on the cluster. Defaulting to jetski/speedboat OPTION 2")
    # return "sleuths", "triesch-shared"


def get_job_name():
    return os.path.basename(os.getcwd())


def ssh_command(cmd):
    host="fias.uni-frankfurt.de"
    user="wilmot"
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.load_system_host_keys()
    PASSWORD = getpass("Please enter password\n")
    client.connect(host, username=user, password=PASSWORD)
    stdin, stdout, stderr = client.exec_command("""(
        eval "$(/home/wilmot/.software/miniconda/miniconda3/bin/conda shell.bash hook)" ;
        export COPPELIASIM_ROOT=/home/aecgroup/aecdata/Software/CoppeliaSim_Edu_V4_2_0_Ubuntu16_04 ;
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT ;
        export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT ;
        export COPPELIASIM_MODEL_PATH=/home/wilmot/Documents/code/aec-tf-v2/models/ ;
        cd /home/wilmot/Documents/code/aec-tf-v2/src ;
        {})""".format(cmd))
    for line in stdout.readlines():
        print(line, end="")
    for line in stderr.readlines():
        print(line, end="")
    print("")
