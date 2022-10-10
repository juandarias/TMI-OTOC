using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

#using IterativeSolvers: cg, cg!, gmres, gmres!
using HDF5
#using LinearAlgebra
using SparseArrays
using TensorOperations

#using mps_compressor
using dmrg_methods


##### Parameters #####
######################

const alpha = 2.5;
const J = -1.0;
const Bx = 1.05;
const Bz = -0.5;
const Ni = 5;
const Nj = 5;
const N = 10;
const t = 1.0;
const dt = 0.01;
const M = Int(t/dt);
const dt_1 = 0.1;
const M_1 = Int(t/dt_1);


##### Compress U(t=1), dt = 0.01 #####
######################################
function compress_Ut(initial_mpo::MPO{T}, incr_mpo::MPO{T}, dt, step_start, step_end; kwargs...) where {T}
    U_mdt = MPO(copy(initial_mpo.Wi));
    for m ∈ step_start:step_end
        printstyled("\n##### Calculating U($(m*dt)) ##### \n")
        U_mdt = prod(U_mdt, incr_mpo; compress = true, LBFGS=true, seed = U_mdt, kwargs...)
        #U_seed = U_mdt
        flush(stdout);
    end
    return U_mdt
end



cg_params = Dict(
    :fav_solver=>gmres!,
    :alt_solver=>cg!,
    :abstol=>5e-7,
    :reltol=>1e-7,
    :tol_compr=>1e-6,
    :verb_compressor=>0,
    #:maxiter=>Int(216*10), #* Increase for higher tolerances
    :log=>true,
    :max_solver_runs=>20 #* Increase for higher tolerances
)


W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj);

W_3dt = calc_Ut(W_II, 3*dt);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));
U_3dt = MPO(copy([W_3dt.Wi[1], fill(W_3dt.Wi[2], N-2)..., W_3dt.Wi[3]]));


lbfgs_params = Dict(
    :verb_compressor => 0,
    :options_solver => Optim.Options(
        store_trace = true, 
        show_trace = false,
        iterations = U_3dt.D[1],
        f_tol = 5e-7
        )

)

# U_comp = compress_Ut(U_3dt, U_dt, dt, 4, M; cg_params...) #* CG version
U_comp = compress_Ut(U_3dt, U_dt, dt, 4, M; lbfgs_params...) #* LBFGS version





using LinearAlgebra
ϵ(Mtilde::Vector) = norm(transpose(Ltilde)*(cr(Mtilde)*Rtilde) - transpose(L)*M_d*R);
g = x -> ForwardDiff.gradient(ϵ, x)
g(seed)


f = h5open(datadir("U_dt=0.01_t=1_compressed.h5"), "w");
create_group(f, "Tensors");
for i in 1:10
    f["Tensors/Wi_$(i)"] = U_comp.Wi[i];
end
close(f)

##### Compress U(t=1), dt = 0.1 #####
######################################
W_II = WII(alpha, N, J, Bx, Bz, dt_1, Ni, Nj);

W_3dt = calc_Ut(W_II, 3*dt_1);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));
U_3dt = MPO(copy([W_3dt.Wi[1], fill(W_3dt.Wi[2], N-2)..., W_3dt.Wi[3]]));

U_comp = compress_Ut(U_3dt, U_dt, dt_1, 4, M; cg_params...)

f = h5open(datadir("U_dt=0.1_t=1_compressed.h5"), "w");
create_group(f, "Tensors");
for i in 1:10
    f["Tensors/Wi_$(i)"] = U_comp.Wi[i];
end
close(f)