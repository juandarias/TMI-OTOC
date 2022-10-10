#! https://stackoverflow.com/questions/51139711/hpc-cluster-select-the-number-of-cpus-and-threads-in-slurm-sbatch
bash = "#!/bin/sh\n"
slurm_settings = "
#SBATCH --ntasks 6 
#SBATCH --cpus-per-task=8 
#SBATCH --ntasks-per-node=2 
#SBATCH -t 48:00:10 
#SBATCH --mem=2G 
#SBATCH --mail-user=j.d.ariasespinoza2@uva.nl 
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=\"./outlogs/PL_alpha_%A_%a.out\" 
"

modules = "
module load 2021 
module load HDF5/1.10.7-gompi-2021a 
module load GLib/2.68.2-GCCcore-10.3.0 
module load Eigen/3.3.9-GCCcore-10.3.0 
module load GSL/2.7-GCC-10.3.0 
module load Boost/1.76.0-GCC-10.3.0 
module load  OpenMPI/4.1.1-GCC-10.3.0
"

parallel_settings = """
N_JOB=\$SLURM_NTASKS
NPROC=`nproc --all`
export OMP_NUM_THREADS=\$NPROC 
"""

locations = "
LOC=\"/home/diego/Code/Cpp/TDVP/_research/\"
DATA=\$LOC/data/
cd \$LOC
"

name_script = "TMI_benchmark.in"
pl_exp = [1.2, 3.0];
svd_tol = [0.0, 1e-8, 1e-6];
size = 16;
time = 16;
J = 1.0;
Bx = -1.05;
Bz = 0.5;
steps = 320;
nSV = 2^8;

script_body = "\n";
misc = "\n\nwait\n\n"; 


for α in pl_exp, ϵ in svd_tol
    filename = "L=$(size)_t=$(time)_Bx=$(Bx)_Bz=$(Bz)_pl_exp=$(pl_exp)_NSV=$(nSV)_eps_svd=$(ϵ)";
    outfolder = "PL_alpha_$α/";
    command = "./TDVP_PL -L=$(size) -Jmax=$(J) -Bx=$(Bx) -Bz=$(Bz) -pl_exp=$(α) -eps_svd=$(ϵ) -max_Nsv=$(nSV) -t_quench=$(time) -steps_quench=$(steps) -outfile=$(filename) -outfolder=$(outfolder) -CALC_OTOC=0 &\n"
    global script_body *= command
    global misc *= "tar  --remove-files -cvf \$DATA$(outfolder)$(filename).tar.gz $(filename)*\n"
end

script_body= chop(script_body, tail=2)
script = bash* slurm_settings * modules * parallel_settings * locations * script_body * misc;


open(name_script, "w") do io
    write(io, script)
end;

run(`cat $name_script`)

run(`sbatch $name_script`)

