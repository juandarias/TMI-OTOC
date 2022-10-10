push!(LOAD_PATH, pwd());
using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5, LinearAlgebra, SparseArrays, Dates, Arpack, TensorOperations
include("MPS_conversions.jl");
#include("./ncon.jl");

println("Number of threads BLAS = ", BLAS.get_num_threads());
flush(stdout)

obs = h5open(datadir("debug_v3_fw.h5"), "r")


step_n = 5;
location = "/home/jdarias/TMI/L32/alpha_3.5/NSV_400"
file_name = "$(location)/single_site_forward_state_step=$(step_n)_AC.h5"
data = h5open(file_name, "r")
mps_tensors = read(data["mps"])
psi_tensors = reconstructcomplexState(mps_tensors)


args_dict = ArgParser(ARGS)
const folder = getParm(args_dict, "folder", "nofolder");
const outfile = getParm(args_dict, "outfile", "results");
const alpha = getParm(args_dict, "alpha", 0.5);
const calc_each = getParm(args_dict, "calc_each", 5);
const step_end = getParm(args_dict, "step_end", 100);
const step_start = getParm(args_dict, "step_start", 5);

# Creates storage file
rho_red = h5open("$(folder)/$(outfile).h5", "w"); #creates the collection of states
create_group(rho_red, "spectrum_AC"); #for mps
create_group(rho_red, "spectrum_BC"); #for mps
create_group(rho_red, "rho_AC"); #for rebuilt states
create_group(rho_red, "rho_BC"); #for rebuilt states

sites_out_AC = vcat(collect(8:15), collect(25:32))
sites_out_BC = vcat(collect(1:8), collect(25:32))
#l = Threads.SpinLock()

for (n, step_n) in enumerate(step_start:calc_each:step_end)
    start_time = now();
    file_name = "$(folder)/single_site_forward_state_step=$(step_n).h5"
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    mps_in = read(Ψ_i["mps"]);
    
    println("Reading tensors of step: $(step_n)")
    flush(stdout)
    tensors = MPS_tensors(mps_in);

    println("Calculating reduced density matrix AC step: $(step_n)")
    flush(stdout)
    rho_AC = density_matrix(tensors,sites_out_AC)
    
    println("Calculating SVD: $(step_n)")
    flush(stdout)
    vec, vals = eigen(rho_AC);
    rho_red["spectrum_AC/step_$(step_n)"] = vals
    

    #rho_red["rho_AC/step_$(step_n)"] = density_matrix(tensors,sites_out_AC)
    #println("Calculating reduced density matrix BC step: $(step_n)")
    #flush(stdout)
    #rho_BC = density_matrix(tensors,sites_out_BC)
    #rho_red["rho_BC/step_$(step_n)"] = rho_BC
    
    
    #F_BC=svd(rho_BC);
    #rho_red["spectrum_BC/step_$(step_n)"] = F_BC.S
    t_trans = (now()-start_time)/Millisecond(1) * (1 / 60000);
    println(", Time elapsed: $(t_trans) min \n")
    flush(stdout)
    close(Ψ_i)

end


#Ψ_all["states/forward_quench"] = Matrix(Ψ_rebuilt);
#Ψ_all["states/transpose"] = Matrix(Ψ_rebuilt_transp);
close(rho_red);


