using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using BSON
using Dates: now, Time, DateTime
using IterativeSolvers: cg, cg!, gmres, gmres!
using HDF5
#using LinearAlgebra
using SparseArrays
using TensorOperations

#using mps_compressor
using dmrg_methods
using argparser


##### Parameters #####
######################
args_dict = collect_args(ARGS)

## System
const alpha = get_param!(args_dict, "alpha", 2.5);
const J = get_param!(args_dict, "J", -1.0);
const Bx = get_param!(args_dict, "Bx", 1.05);
const Bz = get_param!(args_dict, "Bz", -0.5);
const Ni = get_param!(args_dict, "Ni", 5);
const Nj = get_param!(args_dict, "Nj", 5);
const N = get_param!(args_dict, "N", 10);
const tf = get_param!(args_dict, "tf", 1.0);
const ti = get_param!(args_dict, "ti", 0.03);
const dt = get_param!(args_dict, "dt", 0.01);
const Mi = Int(round(ti/dt));
const Mf = Int(round(tf/dt));

## Compressor
const canonize_initial = get_param!(args_dict, "CI", false);
const VARMETHOD = COMPRESSOR(get_param!(args_dict, "METHOD", 2));
const fav_solver = ALGORITHM(get_param!(args_dict, "fav_solver", 1));
const alt_solver = ALGORITHM(get_param!(args_dict, "alt_solver", 2));
const abstol = get_param!(args_dict, "abstol", 1e-3);
const reltol = get_param!(args_dict, "reltol", 5e-7);
const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const VERB_COMPR = get_param!(args_dict, "VERB_COMPR", 0);
const maxiter = get_param!(args_dict, "maxiter", 0);
const solver_runs = get_param!(args_dict, "solver_runs", 10);
const Dmax = get_param!(args_dict, "Dmax", 500);

svd_params = Dict(:METHOD => SVD, :Dmax => Dmax)

var_params = Dict(
    :METHOD => VAR_CG,
    :Dmax => Dmax,
    :abstol => abstol,
    :reltol => reltol,
    :tol_compr => tol_compr,
    :verb_compressor => VERB_COMPR,
    :log => true,
    :max_solver_runs => solver_runs #* Increase for higher tolerances
)

maxiter != 0 && (var_params[:maxiter] = maxiter) #* Increase for higher tolerances
if fav_solver == CG
    var_params[:fav_solver] = cg!
    var_params[:alt_solver] = gmres!
else
    var_params[:fav_solver] = gmres!
    var_params[:alt_solver] = cg!
end


## Output and input
global JOB_ID = get_param!(args_dict, "ID", string(DateTime(now()))[1:end-7]);
const output_file = get_param!(args_dict, "output_file", "U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_SVD_"*JOB_ID);

## Print summary
log_message("\n ##### Parameters: $(string(args_dict)) \n")


##### Use SVDed MPO as seed for var optimizer #####
###################################################

function compressUT(t_mpo::MPO{T}, incr_mpo::MPO{T}, dt::Float64, step_start::Int, step_end::Int; svd_params, var_params) where {T}
    for m ∈ step_start:step_end
        log_message("\n ##### Calculating U($(m*dt)) ##### \n"; color = :blue)
        t_mpo = prod(t_mpo, incr_mpo); 
        
        log_message("\n Generating seed using SVD with Dmax = $(svd_params[:Dmax])\n"; color = :blue)
        var_params[:seed] = mpo_compress(t_mpo; svd_params...); # generate a seed by using SVD compression
        # compress variationally, bond dimension of optimal MPO can increase. 
        log_message("\n Starting variational compression\n"; color = :blue)
        t_mpo = mpo_compress(t_mpo; var_params...); 
        
        log_message("\n Bond dimension of compressed U(t) : $(maximum(t_mpo.D)) \n"; color = :blue) 
    end
    return t_mpo
end


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

## Do compression
comp_Ut = compressUT(U_initial, U_dt, dt, Mi+1, Mf; svd_params = svd_params, var_params = var_params)


## Save results
bson(datadir(output_file*".bson"), args_dict);
f = h5open(datadir(output_file*".h5"), "w");
create_group(f, "Tensors");
for i in 1:N
    f["Tensors/Wi_$(i)"] = comp_Ut.Wi[i];
end
close(f)






