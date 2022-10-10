push!(LOAD_PATH, pwd());
using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5, LinearAlgebra, SparseArrays, Dates, Arpack, TensorOperations
include(srcdir("MPS_conversions.jl"));


data = h5open(datadir("single_site_forward_state_step=1.h5"),"r")
mps = read(data["mps"])
tensors = MPS_tensors(mps);
psi = reconstructcomplexState(mps)


###* Manual state calculation.
#! For L=32, contracting fails due to memory when i > 21
#      ______
#     |     |
# d---|  A  |---D2
#     |_____|
#        |
#       D1

Aᵢ₋₁ = tensors[1]
for i ∈ 2:20
    dims = size(tensors[i]);
    Aᵢ = reshape(permutedims(tensors[i],(2,1,3)),(dims[2],dims[1]*dims[3]))
    Ãᵢ = Aᵢ₋₁*Aᵢ
    Aᵢ₋₁ = reshape(Ãᵢ,(2^i,dims[3]))
end
psi_manual= reshape(Aᵢ₋₁*tensors[32],2^32)



####* Exploiting transfer matrices of canonical form to calculate reduced density matrices

all_tensors = vcat(tensors[1:9], conj.(tensors[1:9]));
dims = size(tensors[9]);
T₁₀ = spdiagm(ones(dims[3]))
push!(all_tensors, T₁₀)

con_indices =  generate_indices(9, edge="left")
push!(con_indices,[79,89])
order_indices= vcat([pS.(["7$n", "8$n"]) for n ∈ 1:9]...)

rho = reshape(TensorOperations.ncon(all_tensors, con_indices, order=order_indices),(2^9, 2^9))

con_indices_psi = con_indices[10:18]
con_indices_psi[end] = [-9,78,-10]
psi = reshape(TensorOperations.ncon(tensors[1:9], con_indices_psi, order=[n for n ∈ 71:78]), (2^9,512));
rho_alt = ncon([psi, adjoint(psi), collect(T₁₀)], [[-1,1],[2,-2],[1,2]], order = [1,2])

norm(rho - transpose(rho_alt)) #! They are equivalent