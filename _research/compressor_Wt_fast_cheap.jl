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

#using mps_compressor
using dmrg_methods
include(srcdir("observables.jl"))



##### Parameters #####
######################
## System


const alpha = get_param!(args_dict, "alpha", 2.5);
const N = get_param!(args_dict, "N", 10);
const kac_normalized = get_param!(args_dict, "kac", true);

if kac_normalized == false
    #kac = sum([abs(i-j)^(-alpha) for i ∈ 1:N for j ∈ i+1:N])/(N-1);
    kac = 1;
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
const Mi = get_param!(args_dict, "Mi", Int(round(ti/dt)) + 1);
const Mf = get_param!(args_dict, "Mf", Int(round(tf/dt)));
const loc_W = get_param!(args_dict, "loc_W", N);

## Compressor
const canonize_initial = get_param!(args_dict, "CI", true);
const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const Dmax = get_param!(args_dict, "Dmax", 200);
const DSVD_max = get_param!(args_dict, "DSVD", 500);
const ϵSVD = get_param!(args_dict, "eps_svd", tol_compr);
const Drate = get_param!(args_dict, "Drate", 1.1);
const VERB = get_param!(args_dict, "VERB", 1);



## Debug parameters
const normWt = get_param!(args_dict, "normWt", false);
const svd_only = get_param!(args_dict, "svd_only", false);
const save_seed = get_param!(args_dict, "save_seed", true);
const save_each = get_param!(args_dict, "save_each", 5);


#svd_params = Dict(:METHOD => SVD, :Dmax => DSVD_max);
svd_params = Dict(:Dmax => Dmax, :ϵmax => ϵSVD);
var_params = Dict(:tol_compr => tol_compr, :Dmax => Dmax, :rate => Drate, :normalize => normWt, :VERB => VERB);

## Output and input
const output_file = get_param!(args_dict, "output_file", "Wt_alpha=$(alpha)_N=$(N)_t=$(kac*tf)_dt=$(kac*dt)_tol=$(tol_compr)_$(JOB_ID)");

if svd_only == false
    result_file = "rho_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_$(JOB_ID)";
else
    result_file = "rho_svd_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_$(JOB_ID)";
end

## Print summary and save parameters
log_message("\n ##### Parameters: $(string(args_dict))\n")
bson(datadir("$(output_file).bson"), args_dict);


##### Use W(m*dt) as seed for var optimizer #####
###################################################

#! W(t) is manipulated as an MPS during the whole calculation, only when exporting it is casted to MPO form
function compress_Wt(W_t::MPS{T}, U_dt::MPO{T}, dt::Float64, step_start::Int, step_end::Int; svd_params, var_params) where {T}
    local ϵ_c = 0.0;
    for s ∈ step_start:step_end
        tm = round(s*dt, digits=2)

        log_message("\n ##### Loading previous results for W($(tm)), step = $s (if available) ##### \n"; color = :blue);
        W_load, loaded = load_tensors("$(output_file)_step=$(s)");

        if loaded == true # Uses loaded result
            log_message(" ##### Previous calculations found! ##### \n"; color = :blue);
            W_t = W_load;
            svd_params[:Dmax] = maximum(W_t.D); # updates Dmax for SVD seed

        elseif loaded == false # Calculates W(t) from scratch
            log_message("\n ##### Starting new calculation for W($(tm)) ##### \n"; color = :blue)

            ### Calculate W(t + dt) and bring to canonical form
            update_Wt!(W_t, U_dt)

            ### Generate SVD seed
            tsvd_i = now();
            log_message("\n1) Generating SVD seed with Dmax = $(svd_params[:Dmax]) or ϵmax = $(svd_params[:ϵmax]) -> "; color = :blue)

            seed_mps = deepcopy(W_t);
            ϵ_svd =  mps_compress_svd!(seed_mps; svd_params...); #* Seed is normalized

            tsvd_t = round(now() - tsvd_i, Minute);
            log_message("Total compression error = $(ϵ_svd), compression time = $(tsvd_t)\n"; color = :blue)
            ###

            if svd_only == true
                W_t = seed_mps;
            else
                ### Variational compression
                tvar_i = now();
                log_message("\n\n2) Starting variational compression: "; color = :blue)

                log_message("Normalization -> "; color = :blue)
                normalize!(W_t);
                log_message("Sweeps : \n"; color = :blue)
                W_t, ϵ_c =  mps_compress_var(W_t, seed_mps; var_params...);

                tvar_t = round(now() - tvar_i, Minute);
                log_message("Final bond dimension = $(maximum(W_t.D)), compression time = $(tvar_t)#####\n"; color = :green)
                ###
            end

            ### Save state
            if mod(s, save_each) == 0
                save_seed == true && save_tensors("$(output_file)_seed_step=$(s)", cast_mpo(seed_mps), ϵ_svd);
                save_tensors("$(output_file)_step=$(s)", cast_mpo(W_t), ϵ_c);
            end

        end

        ### Calculate densities
        ρΛ = operator_density(cast_mpo(W_t), normalized = true);

        log_message("\nρ($(tm)) = $(abs.(ρΛ))"; color = :blue);

        h5open(datadir("WII/results/$(result_file).h5"), "cw") do f;
            try
                f["rho_l/step_$(s)"] = ρΛ;
            catch e
            end
        end


    end
    return W_t, ϵ_c
end


function save_tensors(output_file, mpo::MPO, ϵ)
    h5open(datadir("WII/D_$(Dmax)/$(output_file).h5"), "w") do f;
        create_group(f, "Tensors");
        create_group(f, "Diagnosis");
        f["Diagnosis/Dmax"] = maximum(mpo.D);
        f["Diagnosis/ϵ_c"] = ϵ;
        for i in 1:N
            f["Tensors/Wi_$(i)"] = mpo.Wi[i];
        end
    end
end

function save_tensors(output_file, mps::MPS, ϵ)
    h5open(datadir("WII/D_$(Dmax)/$(output_file).h5"), "w") do f;
        create_group(f, "Tensors");
        create_group(f, "Diagnosis");
        f["Diagnosis/Dmax"] = maximum(mpo.D);
        f["Diagnosis/ϵ_c"] = ϵ;
        for i in 1:N
            f["Tensors/Wi_$(i)"] = mps.Ai[i];
        end
    end
end

function load_tensors_mpo(input_file)
    try
        data = h5open(datadir("WII/D_$(Dmax)/$(input_file).h5"), "r");
        Wi = Vector{Array{ComplexF64, 4}}();
        for n ∈ 1:N
            push!(Wi, read(data["Tensors/Wi_$(n)"]))
        end
        return MPO(Wi), true
    catch e
        return 0, false
    end
end


function load_tensors(input_file)
    try
        data = h5open(datadir("WII/D_$(Dmax)/$(input_file).h5"), "r");
        Ai = Vector{Array{ComplexF64, 3}}();
        for n ∈ 1:N
            Wi = read(data["Tensors/Wi_$(n)"]);
            Wi = reshape(permutedims(Wi, (2, 1, 3, 4)), (size(Wi, 2), 4, size(Wi, 4)));
            push!(Ai, Wi)
        end

        Wt_mps =  MPS(Ai);
        Wt_mps.d = 4;
        Wt_mps.physical_space = BraKet();
        return Wt_mps, true
    catch e
        #log_message(String(e); color = :yellow);
        return 0, false
    end
end


function main()
    start_time = now();

    # calc U(dt)
    W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj; kac_norm = kac_normalized);
    U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));

    # calc U(ti)
    W_ti = calc_Ut(W_II, dt);
    #U_ti = MPO(copy([W_ti.Wi[1], fill(W_ti.Wi[2], N-2)..., W_ti.Wi[3]]));

    # Calc W(dt)
    𝕐 = im*[0.0 -1.0; 1.0 0.0]
    W_ti = calc_Wt(U_dt, 𝕐, loc_W);

    ## Create output directory
    try
        Base.Filesystem.mkdir(datadir("WII/D_$(Dmax)"))
    catch
    end

    ## Create results file
    h5open(datadir("WII/results/$(result_file).h5"), "cw") do f;
        try
            create_group(f, "rho_l");
        catch
        end
    end

    ## Do compression
    comp_Wt, ϵ_c = compress_Wt(cast_mps(W_ti), U_dt, dt, Mi, Mf; svd_params = svd_params, var_params = var_params)
    #save_tensors(output_file*"_final", comp_Wt, ϵ_c);

    total_time = canonicalize(round(now() - start_time, Minute));
    log_message("\n\n Total compression time : $(total_time)")
    return 0
end



main()
