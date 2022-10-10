using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
using IterativeSolvers: cg, cg!, gmres, gmres!
using LinearAlgebra
#using Optim
using Revise
#using SparseArrays
#using Symbolics
using TensorOperations

using dmrg_types
using dmrg_methods
#using dmrg_timedep
using mps_compressor
includet(srcdir("mpo_timedep.jl"));

### TODO ###
# - replace Taylor expansion of exp Φ by some iterative method
# - compare U(δt) againt ED results for small system size
# - check method behind linsolve of KrylovKit.jl
### TODO ###

#####
##### Build WII representation
#####

alpha = 2.5;
J = -1.0;
Bx = 1.05;
Bz = -0.5;
Ni = Nj = 5;
dt = 0.1;

W_2_5 = WII(alpha, J, Bx, Bz, dt, Ni, Nj);

#####
##### Test Projection
#####

    # e.g. 𝟙^2(X+Y+Z)(X+Y+Z+1)^2

    𝟙 = [1 0; 0 1];     
    ℤ = [1 0; 0 -1];    
    𝕐	= im*[0 -1; 1 0];   
    𝕏 = [0 1; 1 0];
    P = [[𝟙] [𝟙] [(𝕏+𝕐+ℤ)] [(𝕏+𝕐+ℤ+𝟙)] [(𝕏+𝕐+ℤ+𝟙)] [(𝕏+𝕐+ℤ+𝟙)]]


    W1 = collect(reshape(W_2_5.W1, (2,2,Ni+1)))
    Wi = collect(reshape(W_2_5.Wi, (2, Ni+1, 2, Ni+1)))
    WN = collect(reshape(W_2_5.WN, (2, Ni+1, 2)))

    @tensor PW1[a] := W1[x,y,a]*P[1][x,y]

    @tensor PW2[a,b] := Wi[x,a,y,b]*P[2][x,y]
    @tensor PW3[a,b] := Wi[x,a,y,b]*P[3][x,y]
    @tensor PW4[a,b] := Wi[x,a,y,b]*P[4][x,y]
    @tensor PW5[a,b] := Wi[x,a,y,b]*P[5][x,y]

    @tensor PW6[a] := WN[x,a,y]*P[6][x,y]

    @tensor PH = PW1[a]*PW2[a,b]*PW3[b,c]*PW4[c,d]*PW5[d,e]*PW6[e]


#####
##### Variational compression of unitary
#####

### Plan ###
# -reshape MPO to MPS
# -use variational optimization
#! -use AD to calculate derivatives

Udt_seed =  calc_Ut(W_2_5, 0.3);
Udt =  calc_Ut(W_2_5, 0.5); #! too large to calculate the norm.

Udt_mps =  cast_MPS(Udt; normalize = false);
Udt_seed_mps =  cast_MPS(Udt_seed; normalize = false);


##### Using Iterative solver
## InterativeSolvers

solver_parms = Dict(
    :fav_solver=>cg!,
    :alt_solver=>gmres!,
    :abstol=>Real(7.5e-6),
    :reltol=>Real(7.55e-6),
    :verb_compressor=>0,
    :maxiter=>Int(216*5),
    :log=>true,
    :max_solver_runs=>20
)

Udt_comp_dt5 =  do_comprsweep_cg(Udt_mps, Udt_comp, 1e-6; solver_parms...); #! if the MPO is translation invariant, can I just simply optimize the central tensors?







##### Using minimization routine
## Error function:  ̃L_(i-1,j-1)*̃M_(i-1,i)*̃R_(i,j) - (L_(i-1,j-1))ᵀ*M_(i-1,i)*R_(i,j),
# equivalently @tensor ϵ[a,b] := Ltilde[x,a]*Mtilde[x,y]*Rtilde[y,b] -L[x,a]*M[x,y]*R[y,b].
# The cost is O(D_seed^3)
#= 
M = Udt_mps.Ai[3][:,1,:]
ϵ(Mtilde) = norm(transpose(Ltilde)*(Mtilde*Rtilde) - transpose(L)*M*R);



## KrylovKit
using KrylovKit

function mapA(x, Ltilde,Rtilde) # Linear map for matrix-less evaluation of Ax
    dim = Int(sqrt(size(x)[1]));
    x = reshape(x, dim , dim)
    @tensor Ox[a,b] := Ltilde[x,a]*x[x,y]*Rtilde[y,b]
    Ox = reshape(Ox, :, 1)
    return Ox
end

M = Udt_mps.Ai[3][:,1,:];
b = reshape(transpose(L)*M*R, :, 1);
xseed = reshape(Udt_seed_mps.Ai[3][:,1,:], : ,1);

sol, info = linsolve(mapA, b, xseed)
#if info.converged == 1



do_comprsweep(Udt_mps, Udt_seed_mps, 0.01)


 =#


