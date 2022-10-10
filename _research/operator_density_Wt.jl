#### Initialization ####
########################

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using argparser
args_dict = collect_args(ARGS)
global JOB_ID = get_param!(args_dict, "ID", "r_id$(string(rand(1:100)))");


## Modules ##
using BSON
using HDF5
using TensorOperations

using dmrg_methods
include(srcdir("observables.jl"))


##### Parameters #####
######################

## System
const N = get_param!(args_dict, "N", 10);
const alpha = get_param!(args_dict, "alpha", 2.5);
const tf = get_param!(args_dict, "tf", 1.0);
const dt = get_param!(args_dict, "dt", 0.1);

const op = get_param!(args_dict, "op", "X");
const loc_O = get_param!(args_dict, "loc_O", N);

const start_step = get_param!(args_dict, "start_step", 1);
const each_step = get_param!(args_dict, "each_step", 1);
const end_step = get_param!(args_dict, "end_step", 10);

const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const compress = get_param!(args_dict, "compress", false);
const Dmax = get_param!(args_dict, "Dmax", 200);
const DSVD_max = get_param!(args_dict, "DSVD", 500);
const svd_comp = get_param!(args_dict, "svd_comp", false);
const exact = get_param!(args_dict, "exact", false);

log_message("\n ##### Parameters: $(string(args_dict))\n")

exact == true ? (out_folder = "exact") : (out_folder = "WII");


#### Methods ####
#################

function load_tensors(step::Int)
    if svd_comp == false
        input_file = "Wt_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_$(JOB_ID)_step=$(step)";
    else
        input_file = "Wt_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_$(JOB_ID)_seed_step=$(step)";
    end
#    data = h5open("$(obelix_folder)/D_$(Dmax)/$(input_file).h5", "r");
    data = h5open(datadir("$(out_folder)/D_$(Dmax)/$(input_file).h5"), "r");
    Wi = Vector{Array{ComplexF64, 4}}();
    for n ∈ 1:N
        push!(Wi, read(data["Tensors/Wi_$(n)"]))
    end
    return MPO(Wi)
end

#### Calculate densities ####
#############################

if svd_comp == false
    output_file = "rho_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_$(JOB_ID)";
else
    output_file = "rho_svd_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_$(JOB_ID)";
end

bson(datadir("$(output_file).bson"), args_dict);

h5open(datadir("$(out_folder)/results/$(output_file).h5"), "cw") do f;
    try
        create_group(f, "rho_l");
    catch
    end
end

for s ∈ start_step:each_step:end_step
    f = h5open(datadir("$(out_folder)/results/$(output_file).h5"), "cw");
    Wt = load_tensors(s);
    compress == true && (Wt, e_c = mpo_compress(Wt; METHOD = SVD, direction = "left", final_site = N, Dmax = DSVD_max));
    compress == true && (log_message("\n Compressed W(t) to Dmax = $(DSVD_max) with error $(e_c)"))

    ρl = zeros(ComplexF64, N);
    t = round(dt*s, digits = 3);
    log_message("\nCalculating ρ(t) for t = $(t) : "; color = :blue);
    ρΛ = operator_density(Wt, normalized = true)
    f["rho_l/step_$(s)"] = ρΛ
    log_message("$(abs.(ρΛ))")
    close(f);
end



