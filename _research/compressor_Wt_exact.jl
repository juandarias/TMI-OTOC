#### Initialization ####
########################
using MKL

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using argparser
args_dict = collect_args(ARGS)
global JOB_ID = get_param!(args_dict, "ID", "r_id$(string(rand(1:100)))");

## Modules ##

using BSON
using Dates: now, Time, DateTime, Hour, Minute, round, canonicalize
using HDF5
#using LinearAlgebra
using SparseArrays
using TensorOperations


using dmrg_methods
using operators_basis
include(srcdir("hamiltonians.jl"));

##### Parameters #####
######################
## System


const alpha = get_param!(args_dict, "alpha", 2.5);
const N = get_param!(args_dict, "N", 10);
const kac_normalized = get_param!(args_dict, "kac", true);

if kac_normalized == false
    kac = sum([abs(i-j)^(-alpha) for i ∈ 1:N for j ∈ i+1:N])/(N-1);
else
    kac = 1;
end

const J = get_param!(args_dict, "J", -1.0);
const Bx = get_param!(args_dict, "Bx", 1.05);
const Bz = get_param!(args_dict, "Bz", -0.5);
const Ni = get_param!(args_dict, "Ni", 5);
const Nj = get_param!(args_dict, "Nj", 5);
const tf = (1/kac) * get_param!(args_dict, "tf", 1.0);
const ti = (1/kac) * get_param!(args_dict, "ti", 0.3);
const dt = (1/kac) * get_param!(args_dict, "dt", 0.1);
const Mi = Int(round(ti/dt));
const Mf = Int(round(tf/dt));
const loc_W = get_param!(args_dict, "loc_W", N);



## Compressor
const canonize_initial = get_param!(args_dict, "CI", true);
const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const Dmax = get_param!(args_dict, "Dmax", 200);
const DSVD_max = get_param!(args_dict, "DSVD", 500);
const ϵSVD = get_param!(args_dict, "eps_svd", tol_compr);
const ϵ_U = get_param!(args_dict, "eps_Udt", tol_compr);
const Drate = get_param!(args_dict, "Drate", 1.1);

## Debug parameters
const normWt = get_param!(args_dict, "normWt", false);
const svd_only = get_param!(args_dict, "svd_only", false);
const save_seed = get_param!(args_dict, "save_seed", true);




#svd_params = Dict(:METHOD => SVD, :Dmax => DSVD_max);
svd_params = Dict(:Dmax => Dmax, :ϵmax => ϵSVD);
var_params = Dict(:tol_compr => tol_compr, :Dmax => Dmax, :rate => Drate, :normalize => normWt);

## Output and input
const output_file = get_param!(args_dict, "output_file", "Wt_alpha=$(alpha)_N=$(N)_t=$(kac*tf)_dt=$(kac*dt)_tol=$(tol_compr)_$(JOB_ID)");

## Print summary and save parameters
log_message("\n ##### Parameters: $(string(args_dict))\n")
bson(datadir("$(output_file).bson"), args_dict);



##### Use W(m*dt) as seed for var optimizer #####
###################################################

function compress_Wt(W_t::MPO{T}, U_dt::MPO{T}, dt::Float64, step_start::Int, step_end::Int; svd_params, var_params) where {T}
    local ϵ_c = 0.0;
    for m ∈ step_start:step_end
        tm = round(m*dt, digits=2)

        log_message("\n ##### Loading previous results for W($(tm)) (if available) ##### \n"; color = :blue);
        W_load, loaded = load_tensors("$(output_file)_step=$(m)");

        if loaded == false # Calculates W(t) from scratch
            log_message("\n ##### Starting new calculation for W($(tm)) ##### \n"; color = :blue)
            #t_mpo = prod(t_mpo, incr_mpo);

            #log_message("\n ##### Calculating W($(m*dt)) ##### \n"; color = :blue)
            
            ## First step, calculate W(t_i)*U(dt)
            log_message("\n Preparing W(t_i)*U(dt) \n"; color = :blue)
            WU_dt = prod(W_t, U_dt); # W(t_i)*U(dt)
            #? Perhaps I need an intermediate compression here

            #var_params[:seed] = mpo_compress(WU_dt; svd_params...); # generate a seed by using SVD compression
            #WU_dt = mpo_compress(W_t; var_params...);

            ## Second step, calculate W(t) and compress
            log_message("\n Preparing U(dt)†*W(t_i)*U(dt) \n"; color = :blue)
            W_t_mps = cast_mps(prod(conj(U_dt), WU_dt); normalize = false); # U(dt)†*W(t_i)*U(dt)
            sweep_qr!(W_t_mps);

            ### Generate SVD seed
            tsvd_i = now();
            log_message("\n1) Generating SVD seed with Dmax = $(svd_params[:Dmax]) or ϵmax = $(svd_params[:ϵmax]) -> "; color = :blue)

            seed_mps = deepcopy(W_t_mps);
            ϵ_svd =  mps_compress_svd!(seed_mps; svd_params...); #* Seed is normalized

            if save_seed == true
                save_tensors("$(output_file)_seed_step=$(m)", cast_mpo(seed_mps), ϵ_svd);
            end

            tsvd_t = round(now() - tsvd_i, Minute);
            log_message("Total compression error = $(ϵ_svd), compression time = $(tsvd_t)\n"; color = :blue)
            
            if svd_only == true
                W_t = cast_mpo(seed_mps);
            else
                ### Variational compression
                tvar_i = now();
                log_message("\n\n2) Starting variational compression: "; color = :blue)

                log_message("Normalization -> "; color = :blue)
                normalize!(W_t_mps);
                log_message("Sweeps : \n"; color = :blue)
                W_t_mps, ϵ_c =  mps_compress_var(W_t_mps, seed_mps; var_params...);
                W_t = cast_mpo(W_t_mps);

                tvar_t = round(now() - tvar_i, Minute);
                log_message("\n Final bond dimension = $(maximum(W_t.D)), compression time = $(tvar_t)\n"; color = :green) 
            end

            save_tensors("$(output_file)_step=$(m)", W_t, ϵ_c);

        
        elseif loaded == true # Uses loaded result
            log_message(" ##### Previous calculations found! ##### \n"; color = :blue);
            W_t = W_load;
            svd_params[:Dmax] = maximum(W_t.D); # updates Dmax for SVD seed
        end
    end
    return W_t, ϵ_c
end


function save_tensors(output_file, mpo, ϵ)
    h5open(datadir("exact/D_$(Dmax)/$(output_file).h5"), "w") do f;
        create_group(f, "Tensors");
        create_group(f, "Diagnosis");
        f["Diagnosis/Dmax"] = maximum(mpo.D);
        f["Diagnosis/ϵ_c"] = ϵ;
        for i in 1:N
            f["Tensors/Wi_$(i)"] = mpo.Wi[i];
        end
    end
end

function load_tensors(input_file)
    try
        data = h5open(datadir("exact/D_$(Dmax)/$(input_file).h5"), "r");
        Wi = Vector{Array{ComplexF64, 4}}();
        for n ∈ 1:N
            push!(Wi, read(data["Tensors/Wi_$(n)"]))
        end
        return MPO(Wi), true
    catch e
        return 0, false
    end
end

function JPL(α, L)
    J = zeros(L,L);
    for i ∈ 1:L, j ∈ i+1:L
        J[i,j] = abs(i-j)^(-α)
    end
    kacn = sum(J)/(L-1)
    J += transpose(J)
    return J/kacn
    #return J
end

function main()
    start_time = now();
    
    # calc U(dt)    
    Jij = 4*J*JPL(alpha, N);

    TF = TransverseFieldIsing(Jij,[2 * Bx]);
    TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
    Hlong = 2 * Bz * spdiagm(sum([diag(Sᶻᵢ(i, N)) for i in 1 : N]));
    H_TMI = TF_M + Hlong;
    
    U_exact = exp(-im * dt * Matrix(H_TMI));
    U_dt = operator_to_mpo(U_exact);
    
    #! Compress U_dt
    U_dt_mps = cast_mps(U_dt);
    sweep_qr!(U_dt_mps);
    ϵ_svd =  mps_compress_svd!(U_dt_mps; ϵmax = ϵ_U); #* MPS is normalized
    U_dt_comp = cast_mpo(U_dt_mps);

    # calc W(dt)
    Y = im*[0.0 -1.0; 1.0 0.0];
    W_ti = calc_Wt(U_dt_comp, Y, loc_W);
    
    ## Create output directory
    try
        Base.Filesystem.mkdir(datadir("exact/D_$(Dmax)"))
    catch
    end
    
    ## Do compression
    comp_Wt, ϵ_c = compress_Wt(W_ti, U_dt_comp, dt, Mi+1, Mf; svd_params = svd_params, var_params = var_params)
    save_tensors(output_file*"_final", comp_Wt, ϵ_c);
    
    total_time = canonicalize(round(now() - start_time, Minute));
    log_message("\n\n Total compression time : $(total_time)")
    return 0
end



main()

