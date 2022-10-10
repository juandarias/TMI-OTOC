using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

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
using operators_basis

include(srcdir("hamiltonians.jl"));
includet(srcdir("observables.jl"))



## Build Hamiltonian
alpha = 2.5;
J = -1.0;
Bx = 1.05;
Bz = -0.5;
dt = 0.05;
L = 6;

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

Jij = 4*J*JPL(2.5, L);
Bxe = 2*Bx;
Bze = 2*Bz;

TF = TransverseFieldIsing(Jij,[Bxe]);
TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
Hlong = Bze*spdiagm(sum([diag(Sᶻᵢ(i,L)) for i in 1:L]));
H_TMI = TF_M + Hlong;

## Calc W(t)
Y = im*[0 -1; 1 0];
X = [0 1; 1 0];
I = [1 0 ; 0 1];

Dmax = 64;

rho_l = zeros(L, 50);
for s ∈ 1:50
    U_exact = exp(-im * s * dt * collect(H_TMI));
    U_exact_mpo = operator_to_mpo(U_exact);
    #U_t = deepcopy(U_exact_mpo);
    #U_t,  = mpo_compress(U_exact_mpo; METHOD = COMPRESSOR(1), normalize = true, direction = "right", final_site = 1, Dmax = Dmax);
    #println(maximum(U_t.D))
    #W_t = calc_Wt(U_t, Y, L);
    W_t = calc_Wt(U_exact_mpo, Y, L);

    ## Calc rho
    rho_l[:, s] = operator_density(W_t; normalized = false);
    println(s);
end


sum_check =  [sum(rho_l[:,s]) for s ∈ 1:50]
op_size(rho_t) = sum([n * rho_t[n] for n ∈ eachindex(rho_t)])
Y_size_L6 = [op_size(rho_l[:, s]) for s ∈ 1:50]
plot(0.05*collect(1:50), Y_size_L6)


## Compare with ED results

fo = h5open(datadir("op_density/op_dens_mfi_n10_a2.5.hdf5"));
rho_ED = read(fo["op_dens_Y"]);
Y_size_ED = [abs(op_size(rho_ED[s,:])) for s ∈ axes(rho_ED,1)];
t_ed = read(fo["time"]);

plot(t_ed[1:50], Y_size_ED[1:50])
plot!(0.05*collect(1:50), Y_size_L6)

## Effect of normalization

s = 10;
U_exact = exp(-im * s * dt * collect(H_TMI));
U_exact_mpo = operator_to_mpo(U_exact);
#U_t = deepcopy(U_exact_mpo);
#U_t,  = mpo_compress(U_exact_mpo; METHOD = COMPRESSOR(1), normalize = true, direction = "right", final_site = 1, Dmax = Dmax);
#println(maximum(U_t.D))
#W_t = calc_Wt(U_t, Y, L);
W_t = calc_Wt(U_exact_mpo, Y, L);

for n ∈ 1:6
    W_t.Wi[n] = W_t.Wi[n]/(8^(1/L))
end
norm(W_t)
od_t = operator_density(W_t; normalized = true)
sum(od_t)


## Test on compressed W(t)

function load_tensors(input_file)
#    data = h5open("$(obelix_folder)/D_$(Dmax)/$(input_file).h5", "r");
    data = h5open(input_file, "r");
    Wi = Vector{Array{ComplexF64, 4}}();
    for n ∈ 1:10
        push!(Wi, read(data["Tensors/Wi_$(n)"]))
    end
    return MPO(Wi)
end

step = 100;
filename = "Wt_alpha=1.0_N=10_t=5.0_dt=0.05_tol=1.0e-6_r_id30_step=$(step).h5";
datafolder = "/mnt/obelix/TMI/WII/D_500/"

W_t = load_tensors(datafolder*filename);
norm(W_t)

od_t = operator_density(W_t; normalized = true)

