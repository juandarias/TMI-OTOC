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
using Dates: now, Time, DateTime, Hour, Minute
using IterativeSolvers: cg!, gmres!, bicgstabl!
using HDF5
#using LinearAlgebra
using SparseArrays
using TensorOperations

#using mps_compressor
using dmrg_methods
#using argparser


##### Parameters #####
######################
## System
const alpha = get_param!(args_dict, "alpha", 2.5);
const J = get_param!(args_dict, "J", -1.0);
const Bx = get_param!(args_dict, "Bx", 1.05);
const Bz = get_param!(args_dict, "Bz", -0.5);
const Ni = get_param!(args_dict, "Ni", 5);
const Nj = get_param!(args_dict, "Nj", 5);
const N = get_param!(args_dict, "N", 10);
const tf = get_param!(args_dict, "tf", 1.0);
const ti = get_param!(args_dict, "ti", 0.3);
const dt = get_param!(args_dict, "dt", 0.1);
const Mi = Int(round(ti/dt));
const Mf = Int(round(tf/dt));

## Compressor
const canonize_initial = get_param!(args_dict, "CI", true);
const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const Dmax = get_param!(args_dict, "Dmax", 200);
const DSVD_max = get_param!(args_dict, "DSVD", 500);
const Drate = get_param!(args_dict, "Drate", 1.1);

svd_params = Dict(:METHOD => SVD, :Dmax => DSVD_max);
var_params = Dict(:METHOD => SIMPLE, :tol_compr => tol_compr, :Dmax => Dmax, :rate => Drate);

## Output and input
const output_file = get_param!(args_dict, "output_file", "U_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_$(JOB_ID)");

## Print summary and save parameters
log_message("\n ##### Parameters: $(string(args_dict))\n")
bson(datadir("$(output_file).bson"), args_dict);



##### Use U(m*dt) as seed for var optimizer #####
###################################################

function compress_Ut(t_mpo::MPO{T}, incr_mpo::MPO{T}, dt::Float64, step_start::Int, step_end::Int; svd_params, var_params) where {T}
    for m ∈ step_start:step_end
        tm = round(m*dt, digits=2)
        log_message("\n ##### Calculating U($(tm)) ##### \n"; color = :blue)
        t_mpo = prod(t_mpo, incr_mpo); 
        
        log_message("\n1) Generating seed using SVD with Dmax = $(svd_params[:Dmax])\n"; color = :blue)
        var_params[:seed] = mpo_compress(t_mpo; svd_params...); # generate a (unnormalized) seed by using SVD compression
        
        log_message("\n2) Starting variational compression\n"; color = :blue)
        t_mpo = mpo_compress(t_mpo; var_params...); # compress variationally, mps are normalized inside the method
        
        log_message("Bond dimension of compressed U(t) : $(maximum(t_mpo.D)) \n"; color = :green) 
        save_tensors("$(output_file)_step=$(m)", t_mpo);

        svd_params[:Dmax] = maximum(t_mpo.D); # updates Dmax for SVD seed
    end
    return t_mpo
end


function save_tensors(output_file, mpo)
    h5open(datadir("WII/D_$(Dmax)/$(output_file).h5"), "w") do f;
        create_group(f, "Tensors");
        for i in 1:N
            f["Tensors/Wi_$(i)"] = mpo.Wi[i];
        end
    end
end

function main()
    start_time = now();
    
    # calc U_dt
    W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj);
    U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));

    # calc U_i
    W_initial = calc_Ut(W_II, ti);
    U_initial = MPO(copy([W_initial.Wi[1], fill(W_initial.Wi[2], N-2)..., W_initial.Wi[3]]));


    if canonize_initial == true # Shall I bring it to MPS form and canonize first? This will make some of the tensor dimensions smaller
        mpo_mps = cast_mps(U_initial);
        canonize!(mpo_mps);
        U_initial = cast_mpo(mpo_mps);
    end

    
    ## Create output directory
    try
        Base.Filesystem.mkdir(datadir("WII/D_$(Dmax)"))
    catch
    end

    
    ## Do compression
    var_params[:seed] = U_initial;
    comp_Ut = compress_Ut(U_initial, U_dt, dt, Mi+1, Mf; svd_params = svd_params, var_params = var_params)
    save_tensors(output_file*"_final", comp_Ut);
    
    total_time = string(floor(now() - start_time, Hour))*string(floor(now() - start_time, Minute))
    log_message("\n\n Total compression time : "*total_time)
    return 0
end




main()

