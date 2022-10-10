using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using IterativeSolvers: cg, cg!, gmres, gmres!
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
const dt_1 = 0.05;
const M_1 = Int(t/dt_1);


##### Compress U(t=1), dt = 0.01 #####
######################################
function compress_Ut(initial_mpo::MPO{T}, incr_mpo::MPO{T}, seed_mpo::MPO{T}, dt, step_start, step_end; kwargs...) where {T}
    U_mdt = MPO(copy(initial_mpo.Wi));
    for m ∈ step_start:step_end
        printstyled("\n ##### Calculating U($(m*dt)) ##### \n"; color = :green)
        U_mdt = prod(U_mdt, incr_mpo; compress = true, seed = seed_mpo, kwargs...)
        seed_mpo = U_mdt
        flush(stdout);
    end
    return U_mdt
end



cg_params = Dict(
    :fav_solver=>cg!,
    :alt_solver=>gmres!,
    :abstol=>1e-7,
    :reltol=>1e-7,
    :tol_compr=>1e-6,
    :verb_compressor=>0,
    :maxiter=>Int(216*10), #* Increase for higher tolerances
    :log=>true,
    :max_solver_runs=>10 #* Increase for higher tolerances
)


W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj);

W_3dt = calc_Ut(W_II, 3*dt);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));
U_3dt = MPO(copy([W_3dt.Wi[1], fill(W_3dt.Wi[2], N-2)..., W_3dt.Wi[3]]));

U_comp = compress_Ut(U_3dt, U_dt, dt, 4, M; cg_params...)

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

Dmax = 400;
W1_0 = rand(ComplexF64, (2, 1, 2, Dmax));
Wi_0 = rand(ComplexF64, (2, Dmax, 2, Dmax));
WN_0 = rand(ComplexF64, (2, Dmax, 2, 1));
U_seed = MPO([W1_0, fill(Wi_0, N-2)..., WN_0]);

U_comp = compress_Ut(U_3dt, U_dt, U_dt, dt_1, 4, M_1; cg_params...)

f = h5open(datadir("U_dt=0.1_t=1_compressed.h5"), "w");
create_group(f, "Tensors");
for i in 1:10
    f["Tensors/Wi_$(i)"] = U_comp.Wi[i];
end
close(f)

##### Compress using SVD #####
##############################

W_II = WII(alpha, N, J, Bx, Bz, dt_1, Ni, Nj);
W_4dt = calc_Ut(W_II, 4*dt_1);
U_4dt = MPO(copy([W_4dt.Wi[1], fill(W_4dt.Wi[2], N-2)..., W_4dt.Wi[3]]));
U_4dt_mps = cast_mps(U_4dt);

canonize!(U_4dt_mps; Dmax = 1000)


##### Use SVDed MPO as seed for var optimizer #####
###################################################

function compress_Ut(initial_mpo::MPO{T}, incr_mpo::MPO{T}, seed_mpo::MPO{T}, dt, step_start, step_end; kwargs...) where {T}
    U_mdt = MPO(copy(initial_mpo.Wi));
    for m ∈ step_start:step_end
        printstyled("\n ##### Calculating U($(m*dt)) ##### \n"; color = :green)
        U_mdt = prod(U_mdt, incr_mpo; compress = true, seed = seed_mpo, kwargs...)
        seed_mpo = U_mdt
        flush(stdout);
    end
    return U_mdt
end



cg_params = Dict(
    :fav_solver=>cg!,
    :alt_solver=>gmres!,
    :abstol=>1e-7,
    :reltol=>1e-7,
    :tol_compr=>1e-6,
    :verb_compressor=>0,
    :maxiter=>Int(216*10), #* Increase for higher tolerances
    :log=>true,
    :max_solver_runs=>10 #* Increase for higher tolerances
)


