# TODO: if memory becomes an issue, see https://github.com/severinson/H5Sparse.jl

using DrWatson, Dates
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5, LinearAlgebra, SparseArrays
using argparser
include(srcdir()*"/MPS_conversions.jl");
include(srcdir()*"/ncon.jl");




args_dict = ArgParser(ARGS)
const folder = getParm(args_dict, "folder", "nofolder");
const file_root = getParm(args_dict, "file_root", "L=16_t=16_Bx=0.5_Bz=0.5_NSV=256_eps_svd=1e-6_state_step=");
const save_to = getParm(args_dict, "save_to", "results");
const eps_svd = getParm(args_dict, "eps_svd", 0.0);
const pl_exp = getParm(args_dict, "pl_exp", 0.5);
const steps = getParm(args_dict, "steps", 320);
const save_each = getParm(args_dict, "save_each", 1);
const step_start = getParm(args_dict, "step_start", 1);
const step_end = getParm(args_dict, "step_end", 320);
const sites = getParm(args_dict, "sites", 16);
const use_sparse = getParm(args_dict, "use_sparse", 0);



# Creates storage file
Ψ_all = h5open("$(folder)/$(save_to).h5", "w"); #creates the collection of states
create_group(Ψ_all, "mps"); #for mps
create_group(Ψ_all, "states"); #for rebuilt states

num_states = steps ÷ save_each;
#Ψ_rebuilt = zeros(ComplexF64, num_states, 2^sites);
#Ψ_rebuilt_transp = zeros(ComplexF64, 64, 2^sites);
steps_list = collect(step_start:save_each:steps);

for step_n in steps_list
    start_time = now();
    #step_n= steps_list[n];
    print("Rebuilding state of step $(step_n)")
    file_name = "$(folder)/$(file_root)_step=$(step_n).h5"
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    #HDF5.h5l_create_external(Ψ_i.filename, "mps", Ψ_all.id, "mps/step_$(step_n)", HDF5.H5P_DEFAULT,HDF5.H5P_DEFAULT)
    mps_in = read(Ψ_i["mps"]);
    #Ψ_rebuilt[n,:] = reconstructcomplexState(mps_in);
    #Ψ_rebuilt = reconstructcomplexState(mps_in);
    Ψ_all["states/step_$(step_n)"] = reconstructcomplexState(mps_in);
    t_trans = (now()-start_time)/Millisecond(1) * (1 / 60000);
    print(", Time elapsed: $(t_trans) min \n")
#    Ψ_rebuilt_transp[n,:] = reconstructcomplexState(mps_in, transpose=true);
    close(Ψ_i)
end


#Ψ_all["states/forward_quench"] = Matrix(Ψ_rebuilt);
#Ψ_all["states/transpose"] = Matrix(Ψ_rebuilt_transp);
close(Ψ_all);

