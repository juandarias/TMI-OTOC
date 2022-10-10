using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
#using LinearAlgebra
using SparseArrays
using ProgressBars
using TensorOperations

#using mps_compressor
using dmrg_methods

include(srcdir("observables.jl"));


⊗ = kron


##### Parameters #####
######################

alpha = 2.5;
J = -1.0;
Bx = 1.05;
Bz = -0.5;
Ni = Nj = 5;
N = 10;
dt = 0.05;
t = 1.0;
tf = 5.0;
tol_compr = 5.0e-6;
M = t/dt;

##### Methods #####
###################

function rebuild_Ut(address)
    Ut_h5 = h5open(address);
    tensors = Vector{Array{ComplexF64,4}}();
    for i ∈ 1:10
        push!(tensors, read(Ut_h5["Tensors/Wi_$(i)"]));
    end
    return MPO(tensors)
end

function calc_rho_lambda(Λ, ds)
    steps = Int(floor(96/ds))
    rho_t_l = zeros(ComplexF64, 10, steps)
    i = 0;
    # for s ∈ ProgressBar(5:ds:100)
    @inbounds for s ∈ ProgressBar(5:ds:100)
        i += 1;
        @inbounds for n ∈ 1:10
            rho_t_l[n, i] = operator_density(U_t_mpo[s-3], Y, n, Λ)
        end
    end
    #rho_data["Lambda/$(Λ)"] = rho_t_l    
    return rho_t_l
end


##### Calculations #####
########################

X = [0 1; 1 0]; Z = [1 0; 0 -1]; Y = [0 -im; im 0]
obelix_folder = "/mnt/obelix/TMI-OTOC/";

printstyled("\nReading U(t) \n"; color = :blue)
U_t_mpo = [];
@inbounds for s ∈ 4:100
    address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r1_step=$(s).h5"
    push!(U_t_mpo, rebuild_Ut(address));
end

rho_t_l_Y = [];
for Λ ∈ 1:10
    printstyled("Calculating ρ(t,$(Λ))\n"; color = :blue)
    push!(rho_t_l_Y, calc_rho_lambda(Λ, 2))
end

printstyled("\nSaving data \n"; color = :blue)
rho_data = h5open(datadir("rho_alpha=2.5_dt=0.05_tf=5.h5"), "w");
create_group(rho_data, "Lambda")

for Λ ∈ 1:10
    rho_data["Lambda/$(Λ)"] = rho_t_l_Y[Λ]
end

close(rho_data)