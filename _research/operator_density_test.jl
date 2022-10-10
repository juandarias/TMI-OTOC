using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
using TensorOperations
using UnicodePlots

using dmrg_methods
using argparser

include(srcdir("observables.jl"))


##### Parameters #####
######################
args_dict = collect_args(ARGS)

## System
const L = get_param!(args_dict, "L", 10);
const alpha = get_param!(args_dict, "alpha", 2.5);
const tf = get_param!(args_dict, "tf", 5.0);
const dt = get_param!(args_dict, "dt", 0.05);

const op = get_param!(args_dict, "op", "X");
const loc_O = get_param!(args_dict, "loc_O", 1);


const start_step = get_param!(args_dict, "start_step", 1);
const end_step = get_param!(args_dict, "end_step", 10);
const t_step = get_param!(args_dict, "step", 10);

const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const Dmax = get_param!(args_dict, "Dmax", 190);
const DSVD_max = get_param!(args_dict, "DSVD", 500);
const sweep_dir = get_param!(args_dict, "sweep_dir", "left");
const compress = get_param!(args_dict, "compress", true);

const JOB_ID = get_param!(args_dict, "ID", "X");


#### Load and compress Ut ####
#################

function rebuild_Ut(address)
    Ut_h5 = h5open(address);
    tensors = Vector{Array{ComplexF64,4}}();
    for i ∈ 1:10
        push!(tensors, read(Ut_h5["Tensors/Wi_$(i)"]));
    end
    return MPO(tensors)
end


obelix_folder = "/mnt/obelix/TMI/WII/D_$(Dmax)/"
address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID)_step=$(t_step).h5"

U_t = rebuild_Ut(address);
if compress == true
    if sweep_dir == "left"
        U_t = mpo_compress(U_t; METHOD = SVD, direction = "left", final_site = 10, Dmax = DSVD_max, normalize = true);
    elseif sweep_dir == "right"
        U_t = mpo_compress(U_t; METHOD = SVD, direction = "right", final_site = 1, Dmax = DSVD_max, normalize = true);
    end
end
U_mps = cast_mps(U_t);
println("Bond dimensions of U(t) are : ", U_t.D)
println("Norm of U(t) is :", overlap(U_mps, U_mps))

#### Calculate ρₗ ####
######################
Y = im*[0 -1; 1 0];

rho_l = []
for Λ ∈ 1:L
    push!(rho_l, operator_density(U_t, Y, L, Λ, normalized = true))
end

display(scatterplot(real.(rho_l), yscale = :log10))
println()
println("ρₗ($(dt*t_step)) = ", abs.(rho_l))