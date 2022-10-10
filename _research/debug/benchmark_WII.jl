using Revise
using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
#using LinearAlgebra
using SparseArrays
using TensorOperations

#using mps_compressor
using dmrg_methods
include(srcdir("observables.jl"))

using operators_basis
include(srcdir("hamiltonians.jl"));

⊗ = kron

##### Parameters #####
######################

alpha = 2.5;
J = -1.0;
Bx = 1.05;
Bz = -0.5;
Ni = Nj = 5;
N = 10;
dt = 0.01;
t = 1.0;
M = t/dt;


##### Magnetization benchmark against exact #####
###############################################

########## Hamiltonian and initial state

function JPL(α, L)
    J = zeros(L,L);
    for i ∈ 1:L, j ∈ i+1:L
        J[i,j] = abs(i-j)^(-α)
    end
    kacn = sum(J)/(L-1)
    J += transpose(J)
    #return J/kacn
    return J
end


Jij = 4*J*JPL(2.5, N);
kac = sum([abs(i-j)^(-alpha) for i ∈ 1:L for j ∈ i+1:N])/(N-1);

Bxe = kac*2*Bx;
Bze = kac*2*Bz;

TF = TransverseFieldIsing(Jij,[Bxe]);
TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
Hlong = Bze*spdiagm(sum([diag(Sᶻᵢ(i,N)) for i in 1:N]));
H_TMI = TF_M + Hlong;


psi_plus = 1/sqrt(2)*[im, 1];
psi_0 = ⊗([psi_plus for i ∈ 1:N]...);
psi_0_mps = MPS([reshape(psi_plus, 1, 2, 1) for _ ∈ 1:N]);

########## Exact results

dt = 0.05;
tf = 5.0;
tol_compr = 5.0e-6;

psi_t_exact = zeros(ComplexF64, 2^10, 100);
@inbounds for s ∈ 1:100
    U_exact = exp(-im*s*dt*collect(H_TMI));
    psi_t_exact[:,s] = U_exact*psi_0;
end


########## Variational compression results

function rebuild_Ut(address)
    Ut_h5 = h5open(address);
    tensors = Vector{Array{ComplexF64,4}}();
    for i ∈ 1:10
        push!(tensors, read(Ut_h5["Tensors/Wi_$(i)"]));
    end
    return MPO(tensors)
end

function calc_psit(psi_initial::MPS, U, steps::Int)
    psi_dt = psi_initial;
    for m ∈ 1:steps
        psi_dt = mpo_mps_product(psi_dt, U);
    end
    normalize!(psi_dt)
    return psi_dt  
end

## Dmax = 500

obelix_folder = "/mnt/obelix/TMI/WII/D_500/";
psi_t_mps_500 = [];
overlap_500 = [];
for s ∈ 4:100
    address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id1_step=$(s).h5"
    U_t_mpo = rebuild_Ut(address);
    psi_t = mpo_mps_product(psi_0_mps, U_t_mpo);
    push!(psi_t_mps_500, psi_t);
    push!(overlap_500, overlap(psi_t, psi_t));
end

## Dmax = 190

obelix_folder = "/mnt/obelix/TMI/WII/D_190/";
psi_t_mps_190 = [];
overlap_190 = [];
for s ∈ 4:100
    address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id70_step=$(s).h5"
    U_t_mpo = rebuild_Ut(address);
    psi_t = mpo_mps_product(psi_0_mps, U_t_mpo);
    push!(psi_t_mps_190, psi_t);
    push!(overlap_190, overlap(psi_t, psi_t));
end


#### Magnetization ####
#######################


function calc_magnetizations(psi_t_mps::Vector{MPS{ComplexF64}})
    steps = length(psi_t_mps);
    X = [0 1; 1 0]; Z = [1 0; 0 -1]; Y = [0 -im; im 0]

    exp_x = zeros(steps, 5);
    exp_y = zeros(steps, 5);
    exp_z = zeros(steps, 5);

    @inbounds for s in 1:steps
        psi_t_compr = deepcopy(psi_t_mps[s]);
        canonize!(psi_t_compr);
        @inbounds for (i, n) in enumerate([9, 7, 5, 3, 1])
            canonize!(psi_t_compr; final_site = n, direction = "right");
            exp_x[s, i] = real(calc_expval(psi_t_compr, X, n; mixed_canonical=true))
            exp_y[s, i] = real(calc_expval(psi_t_compr, Y, n; mixed_canonical=true))
            exp_z[s, i] = real(calc_expval(psi_t_compr, Z, n; mixed_canonical=true))
            println("Calculating for step $(s) and site $(n)")
        end
    end

    return exp_x, exp_y, exp_z
end


function calc_magnetizations(psi_t::Array{ComplexF64,2})
    steps = size(psi_t)[end]
    
    exp_x = zeros(steps, 2, 5);
    exp_y = zeros(steps, 2, 5);
    exp_z = zeros(steps, 2, 5);
    
    @inbounds for s in 1:steps
        @inbounds for (i, n) in enumerate([9, 7, 5, 3, 1])
            exp_x[j, 2, i] = real(psi[:,s]'*σˣᵢ(n,10)*psi[:,s])
            exp_y[j, 2, i] = real(psi[:,s]'*σʸᵢ(n,10)*psi[:,s])
            exp_z[j, 2, i] = real(psi[:,s]'*σᶻᵢ(n,10)*psi[:,s])
            println("Calculating for step $(s) and site $(n)")
        end
    end

    return exp_x, exp_y, exp_z
end


#### Plots ####
###############


figure = plot(0.3:0.05:5.0, exp_val500[1][:, 1], label="W_II D=500");
plot!(0.3:0.05:5.0, exp_val190[1][:, 1], label="W_II D=190")
plot!(ylabel="⟨X₉⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("X9_alpha=2.5_dt=0.05_tf=5_Dmax=500_vs_190.pdf"))


figure = plot(0.3:0.05:5.0, exp_val500[3][:, 5], label="W_II D=500");
plot!(0.3:0.05:5.0, exp_val190[3][:, 5], label="W_II D=190")
plot!(ylabel="⟨Z₁⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("Z1_alpha=2.5_dt=0.05_tf=5_Dmax=500_vs_190.pdf"))


figure = plot(0.3:0.05:5.0, ex[:, 1, 1], label="W_II dt=0.05");
plot!(0.3:0.05:5.0, ex[:, 2, 1], label="exact")
plot!(ylabel="⟨X₉⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("X9_alpha=2.5_dt=0.05_tf=5_Dmax=500.pdf"))


figure = plot(0.3:0.05:5.0, ex[:, 1, 2], label="W_II dt=0.05");
plot!(0.3:0.05:5.0, ex[:, 2, 2], label="exact")
plot!(ylabel="⟨X₇⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("X7_alpha=2.5_dt=0.05_tf=5_Dmax=500.pdf"))


figure = plot(0.3:0.05:5.0, ex[:, 1, 3], label="W_II dt=0.05");
plot!(0.3:0.05:5.0, ex[:, 2, 3], label="exact")
plot!(ylabel="⟨X₅⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("X5_alpha=2.5_dt=0.05_tf=5_Dmax=500.pdf"))

figure = plot(0.3:0.05:5.0, ex[:, 1, 4], label="W_II dt=0.05");
plot!(0.3:0.05:5.0, ex[:, 2, 4], label="exact")
plot!(ylabel="⟨X₃⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("X3_alpha=2.5_dt=0.05_tf=5_Dmax=500.pdf"))


figure = plot(0.3:0.05:5.0, ex[:, 1, 5], label="W_II dt=0.05");
plot!(0.3:0.05:5.0, ex[:, 2, 5], label="exact")
plot!(ylabel="⟨X₁⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("X1_alpha=2.5_dt=0.05_tf=5_Dmax=500.pdf"))


figure = plot(0.3:0.05:5.0, ez[:, 1, 5], label="W_II dt=0.05");
plot!(0.3:0.05:5.0, ez[:, 2, 5], label="exact")
plot!(ylabel="⟨Z₁⟩", xlabel="tJ")
Plots.savefig(figure, plotsdir("Z1_alpha=2.5_dt=0.05_tf=5_Dmax=500.pdf"))


#### Operator density ####
##########################

Y = im*[0 -1; 1 0];

function calc_rhot(H_exact, Op, t_final, dt, L, Dmax)
    t_steps = collect(0:dt:t_final);
    rho_l = zeros(length(t_steps), L);

    for (i, t) ∈ enumerate(t_steps)
        U_exact = exp(-im * t * collect(H_exact));
        U_exact_mpo = operator_to_mpo(U_exact);
        U_t = deepcopy(U_exact_mpo);
        U_t = mpo_compress(U_exact_mpo; METHOD = COMPRESSOR(1), direction = "right", final_site = 1, Dmax = Dmax, normalize = true);

        rho_s = [];
        for Λ ∈ 1:L
            push!(rho_s, abs(operator_density(U_t, Op, L, Λ; normalized = true)))
        end
        rho_l[i, :] = prepend!([rho_s[n] - rho_s[n-1] for n ∈ 2:L], rho_s[1])
    end
    
end

function calc_rhot(root_file, Op, start_step, step_size, end_step, L, loc_O, Dmax)
    t_steps = collect(start_step:step_size:end_step);
    rho_l = zeros(length(t_steps), L);

    for (i, s) ∈ enumerate(t_steps)
        U_t_mpo = rebuild_Ut(root_file*"_step=$(s).h5");
        U_t = mpo_compress(U_t_mpo; METHOD = COMPRESSOR(1), direction = "left", final_site = L, Dmax = Dmax);

        rho_s = [];
        for Λ ∈ 1:L
            push!(rho_s, abs(operator_density(U_t, Op, loc_O, Λ; normalized = true)))
        end
        rho_l[i, :] = prepend!([rho_s[n] - rho_s[n-1] for n ∈ 2:L], rho_s[1])
    end
    return rho_l
end

function read_rho_sol(path, step_start, step_end, step_size)
    file_dat =  h5open(path, "r")
    dat = read(file_dat["rho_l"])
    rho_l = zeros(length(step_start:step_size:step_end), 10)
    for (i, s) in enumerate(step_start:step_size:step_end)
        rho_l[i, :] = abs.(dat["step_$(s)"])
    end
    return rho_l
end

function normalize_rho(rho)
    for s ∈ 1:24
        rho[s, :] = rho[s, :]/sum(rho[s, :])
    end
    return rho
end

op_size(rho_t) = sum([n * rho_t[n] for n ∈ 1:length(rho_t)])


#### Case Dmax = 190 to Dmax = 64 ####

tf = 5.0;
dt = 0.05;
tol_compr = 5e-6;
ID = "new_norm"
Dmax = 190

obelix_folder = "/mnt/obelix/TMI/WII/D_$(Dmax)/";
address = obelix_folder*"U_alpha=$(alpha)_N=10_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_$(ID)"

rho_Y190 = calc_rhot(address, Y, 4, 4, 100, 10, 10, 64)

### Case Dmax = 130

tf = 5.0;
dt = 0.1;
tol_compr = 1.0e-05;
ID = "r_id30"
Dmax = 130

obelix_folder = "/mnt/obelix/TMI/WII/results/";
address = obelix_folder*"rho_alpha=2.5_t=5.0_dt=0.1_tol=1.0e-5_Dmax=130_r_id30.h5"
rho = read_rho_sol(address, 4, 50, 2)
rho_n = normalize_rho(rho)

Y_size_130 = [op_size(rho[s, :]) for s ∈ 1:24];
time_W = collect(4:2:50)*0.1
plot(time_W, Y_size_130, label = L"D_{max} = 130: \quad \hat{Y}(t)")

sum_op = [sum(rho[s, :]) for s in 1:24]
plot!(sum_op)


### Case Dmax = 64

address = obelix_folder*"rho_alpha=2.5_t=5.0_dt=0.1_tol=1.0e-5_r_id30.h5"
rho = read_rho_sol(address, 4, 50, 2)
rho_n = normalize_rho(transpose(rho))

Y_size_64 = [op_size(rho_n[:, s]) for s ∈ 1:24];
plot!(time_W, Y_size_64, label = L"D_{max} = 64: \quad \hat{Y}(t)")
Plots.xlabel!(L"tJ")
Plots.ylabel!(L"L[W(t)]")
plot!(legend = :topleft)

### Case Dmax = 190

rho_190 = [0.01481887291502898, 0.07273606876918837, 0.170055874907529, 0.33268140160977266, 0.21195793895792236, 0.1298838962352793, 0.06364406360965802, 0.024701641440356026, 0.011855936433358805, 0.009202110756298953]
Y_size_190 = op_size(rho_190)

scatter!([2.5], [Y_size_190], label = L"D_{max} = 190: \quad \hat{Y}(t)")


#### Plots

Y_size_exact = [op_size(rho_l10Y_cor[s, :]) for s ∈ 1:25];

obelix_folder = "/mnt/obelix/TMI/WII/results/"
file_name= "$(obelix_folder)rho_alpha=2.5_t=5.0_dt=0.1_tol=1.0e-5_r_id26.h5"
rho_170 =  read_rho_sol(file_name, 4, 50, 2)
rho_170n = normalize_rho(rho_170)
Y_size_170 = [op_size(rho_170n[:, s]) for s ∈ 1:24];


time_W = collect(4:2:50)*0.1


size_op = plot(collect(1:4:100)*0.05, Y_size_exact, label = L"\hat{Y}(t)")
plot!(time_W, Y_size_170, label = L"D_{max} = 64: \quad \hat{Y}(t)")
plot!(legend=:topleft)
Plots.xlabel!(L"tJ")
Plots.ylabel!(L"L[W(t)]")


#TODO
#* 1 Take a single step of Dmax = 500, truncate it to Dmax = 64 and calculate rho
#* 2 Compare also entanglement entropies for different Dmax
tf = 5.0;
dt = 0.01;
tol_compr = 5e-6;
ID = 1
Dmax = 500

obelix_folder = "/mnt/obelix/TMI/WII/D_$(Dmax)/";
s = 50
address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(ID)_step=$(s).h5"
U_t_mpo = rebuild_Ut(address);
U_t = mpo_compress(U_t_mpo; METHOD = COMPRESSOR(1), direction = "left", final_site = 10, Dmax = 64);
operator_density(U_t, X, L, 10; normalized = true)
U_t = mpo_compress(U_t; METHOD = COMPRESSOR(1), direction = "right", final_site = 1, Dmax = 64);
operator_density(U_t_mpo, X, L, 10; normalized = true)


norm(U_t_mpo)

operator_density(U_t, X, L, 10; normalized = true)


Y = im*[0 -1; 1 0];
X = [0 1; 1 0]

rho_s_100_u = [];
for Λ ∈ 1:L
    push!(rho_s_100_u, abs(operator_density(U_t, X, L, Λ; normalized = true)))
end

rho_l_100 = prepend!([rho_s_100[n] - rho_s_100[n-1] for n ∈ 2:L], rho_s_100[1])
rho_l = prepend!([rho_s[n] - rho_s[n-1] for n ∈ 2:L], rho_s[1])

rho_s_100

rho_s_100_u/o^2