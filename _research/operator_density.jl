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
const L = get_param!(args_dict, "L", 10);
const alpha = get_param!(args_dict, "alpha", 2.5);
const tf = get_param!(args_dict, "tf", 1.0);
const dt = get_param!(args_dict, "dt", 0.1);

const op = get_param!(args_dict, "op", "X");
const loc_O = get_param!(args_dict, "loc_O", L);

const start_step = get_param!(args_dict, "start_step", 1);
const each_step = get_param!(args_dict, "each_step", 1);
const end_step = get_param!(args_dict, "end_step", 10);

const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const compress = get_param!(args_dict, "compress", false);
const Dmax = get_param!(args_dict, "Dmax", 200);
const DSVD_max = get_param!(args_dict, "DSVD", 500);

log_message("\n ##### Parameters: $(string(args_dict))\n")


#### Methods ####
#################

function load_tensors(step::Int)
    input_file = "U_alpha=$(alpha)_N=$(L)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_$(JOB_ID)_step=$(step)";
#    data = h5open("$(obelix_folder)/D_$(Dmax)/$(input_file).h5", "r");
    data = h5open(datadir("WII/D_$(Dmax)/$(input_file).h5"), "r");
    Wi = Vector{Array{ComplexF64, 4}}();
    for n ∈ 1:L
        push!(Wi, read(data["Tensors/Wi_$(n)"]))
    end
    return MPO(Wi)
end

#### Calculate densities ####
#############################

Y = im*[0 -1; 1 0];

output_file = "rho_alpha=$(alpha)_N=$(L)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_$(JOB_ID)";
bson(datadir("$(output_file).bson"), args_dict);


h5open(datadir("WII/results/$(output_file).h5"), "cw") do f;
    try
        create_group(f, "rho_l");
    catch
    end
    for s ∈ start_step:each_step:end_step
        U_t = load_tensors(s);
        compress == true && (U_t, e_c = mpo_compress(U_t; METHOD = SVD, direction = "left", final_site = 10, Dmax = DSVD_max));
        compress == true && (log_message("\n Compressed W(t) to Dmax = $(DSVD_max) with error $(e_c)"))

        ρl = zeros(ComplexF64, L);
        t = round(dt*s, digits = 3);
        log_message("\n Calculating ρ(t) for t = $(t), Λ  "; color = :blue);
        for Λ ∈ 1:L
            log_message("-> $(Λ)  "; time = false, color = :blue)
            ρl[Λ] = operator_density(U_t, Y, loc_O, Λ, normalized = true)
        end
        ρΛ = prepend!([ρl[n] - ρl[n-1] for n ∈ 2:10], ρl[1]);
        f["rho_l/step_$(s)"] = ρΛ
        log_message("\n ρΛ = $(abs.(ρΛ))")
    end
end



