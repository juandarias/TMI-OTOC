#### Initialization ####
########################

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using argparser
args_dict = collect_args(ARGS)
global JOB_ID = get_param!(args_dict, "ID", "r_id$(string(rand(1:100)))");


## Modules ##
using HDF5
using LinearAlgebra: kron
using SparseArrays
using TensorOperations
using dmrg_methods
using operators_basis
include(srcdir("hamiltonians.jl"));



##### Parameters #####
######################

## System
const N = get_param!(args_dict, "N", 10);
const alpha = get_param!(args_dict, "alpha", 1.1);
const tf = get_param!(args_dict, "tf", 5.0);
const dt = get_param!(args_dict, "dt", 0.05);
const steps = Int(round(tf/dt));
const J = get_param!(args_dict, "J", -1.0);
const Bx = get_param!(args_dict, "Bx", 1.05);
const Bz = get_param!(args_dict, "Bz", -0.5);
const Ni = get_param!(args_dict, "Ni", 5);
const Nj = get_param!(args_dict, "Nj", 5);



## Compressor
const comp_type = get_param!(args_dict, "comp_type", "svd");
const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const Dmax = get_param!(args_dict, "Dmax", 4096);
const DSVD_max = get_param!(args_dict, "DSVD", 500);
const ϵSVD = get_param!(args_dict, "eps_svd", tol_compr);
const Drate = get_param!(args_dict, "Drate", 1.25);
const normWt = get_param!(args_dict, "normWt", false);

## Others
const calc_exact = get_param!(args_dict, "calc_exact", false);

const analysis = get_param!(args_dict, "analysis", false);

svd_params = Dict(:Dmax => Dmax, :ϵmax => ϵSVD);
var_params = Dict(:tol_compr => tol_compr, :Dmax => Dmax, :rate => Drate, :normalize => normWt);

## Operators
𝟙 = [1 0; 0 1]
X = [0 1; 1 0]
Z = [1 0; 0 -1]
Y = [0 -im; im 0]

## Methods

function save_tensors(output_file, mps::MPS, ϵ)
    h5open(datadir("WII/benchmark/$(output_file).h5"), "w") do f;
        create_group(f, "Tensors");
        create_group(f, "Diagnosis");
        f["Diagnosis/Dmax"] = maximum(mps.D);
        f["Diagnosis/ϵ_c"] = ϵ;
        for i in 1:N
            f["Tensors/Wi_$(i)"] = mps.Ai[i];
        end
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

⊗ = kron


########## Exact results
if calc_exact == true
    psi_plus = 1/sqrt(2)*[im, 1];
    psi_0 = ⊗([psi_plus for i ∈ 1:N]...);

    Jij = 4 * J * JPL(alpha, N);
    TF = TransverseFieldIsing(Jij,[2 * Bx]);
    TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
    Hlong = 2 * Bz * spdiagm(sum([diag(Sᶻᵢ(i, N)) for i in 1 : N]));
    H_TMI = TF_M + Hlong;


    psi_t_exact = [];
    exp_x = zeros(steps, 5);
    exp_y = zeros(steps, 5);
    exp_z = zeros(steps, 5);
    exp_xx = zeros(steps, 5);

    @inbounds for s ∈ 1:steps
        t = round(dt * s, sigdigits = 3)
        println("Calculating exact |Ψ($t)⟩")
        U_exact = exp(-im * s * dt * collect(H_TMI));
        push!(psi_t_exact, U_exact * psi_0);
        @inbounds for n ∈ 1:5
            #exp_x[s, n] = real(psi_t_exact[end]' * σˣᵢ(n,10) * psi_t_exact[end])
            #exp_y[s, n] = real(psi_t_exact[end]' * σʸᵢ(n,10) * psi_t_exact[end])
            #exp_z[s, n] = real(psi_t_exact[end]' * σᶻᵢ(n,10) * psi_t_exact[end])
            exp_xx[s, n] = real(psi_t_exact[end]' * σˣᵢσˣⱼ(n, 6, 10) * psi_t_exact[end])
        end
    end

    h5open(datadir("WII/benchmark/alpha=$(alpha)_N=$(N)_dt=$(dt)_exact_results.h5"), "cw") do f;
        f["Observables/XX"] = exp_xx;
        #= create_group(f, "States");
        create_group(f, "Observables");
        f["Observables/X"] = exp_x;
        f["Observables/Y"] = exp_y;
        f["Observables/Z"] = exp_z;
        for s in 1:steps
            f["States/Psi_step=$(s)"] = psi_t_exact[s];
        end =#
    end
end


######## WII approximation
#= exp_x_mps = zeros(steps, 5);
exp_y_mps = zeros(steps, 5);
exp_z_mps = zeros(steps, 5);
 =#
exp_xx_mps = zeros(steps, 5);

W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj; kac_norm = true);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));

psi_plus = 1/sqrt(2)*[im, 1];
psi_0_mps = MPS([reshape(psi_plus, 1, 2, 1) for _ ∈ 1:N]);


psi_t_mps = [];
push!(psi_t_mps, deepcopy(psi_0_mps));

@inbounds for s ∈ 1:steps
    t = round(dt * s, sigdigits = 3)

    println("Calculating WII |Ψ($t)⟩")
    psi_t = prod(psi_t_mps[end], U_dt);
    sweep_qr!(psi_t);

    comp_type == "var" && (psi_t_var = deepcopy(psi_t);)
    ϵ_svd =  mps_compress_svd!(psi_t; svd_params...);
    println("Total SVD compression error = $(ϵ_svd)")

    if comp_type == "var"
        println("Var compression with Dmax = $(var_params[:Dmax]) or ϵmax = $(var_params[:tol_compr])")
        normalize!(psi_t_var)
        psi_t_var, ϵ_c =  mps_compress_var(psi_t_var, psi_t; var_params...);
        println("Total var compression error = $(ϵ_c)")
        psi_t = psi_t_var
    end

    outfile = "psi_t_alpha=$(alpha)_N=$(N)_dt_$(dt)_$(comp_type)_tol=$(tol_compr)_step=$(s)"
    save_tensors(outfile, psi_t, ϵ_svd);

    for n ∈ 1:5
        #= exp_x_mps[s, n] = real(calc_expval(psi_t, X, n; mixed_canonical=true))
        exp_y_mps[s, n] = real(calc_expval(psi_t, Y, n; mixed_canonical=true))
        exp_z_mps[s ,n] = real(calc_expval(psi_t, Z, n; mixed_canonical=true)) =#
        exp_xx_mps[s ,n] = real(calc_expval(psi_t, [X,X], [n, 6]))
        sweep_qr!(psi_t, direction = "left", final_site = n + 1);
    end
    push!(psi_t_mps, psi_t)
end

h5open(datadir("WII/benchmark/alpha=$(alpha)_N=$(N)_dt=$(dt)_$(comp_type)_tol=$(tol_compr)_WII_results.h5"), "cw") do f;
    #= create_group(f, "Observables");
    f["Observables/X"] = exp_x_mps;
    f["Observables/Y"] = exp_y_mps;
    f["Observables/Z"] = exp_z_mps; =#
    f["Observables/XX"] = exp_xx_mps;
end

println("##### Done #####")


########################
# Analysis observables #
########################

@assert analysis == true

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using Plots
plotlyjs()
using HDF5
using BSON
include(plotsdir("plotting_functions.jl"));


##### Parameters #####

const alpha = 1.1;
const N = 10;
const dt = 0.05;
const tf = 5.0;
const steps = Int(tf/dt)

t_sim = collect(1:steps) * dt;
##### Exact results #####

outfile = datadir("WII/benchmark/alpha=$(alpha)_N=$(N)_dt=$(dt)_exact_results.h5")

res_exact =  h5open(outfile);

X_exact = read(res_exact["Observables/X"]);
Y_exact = read(res_exact["Observables/Y"]);
Z_exact = read(res_exact["Observables/Z"]);
XX_exact = read(res_exact["Observables/XX"]);


plot(t_sim, X_exact, label=["X1" "X2" "X3" "X4" "X5"])
plot(t_sim, Y_exact, label=["Y1" "Y2" "Y3" "Y4" "Y5"])
plot(t_sim, Z_exact, label=["Z1" "Z2" "Z3" "Z4" "Z5"])


##### SVD compression #####
comp_type = "svd";
tol_compr = 1e-9;
dt = 0.05;
#outfile = datadir("WII/benchmark/alpha=$(alpha)_N=$(N)_dt=$(dt)_$(comp_type)_tol=$(tol_compr)_Ni_10_WII_results.h5")
outfile = datadir("WII/benchmark/alpha=$(alpha)_N=$(N)_dt=$(dt)_$(comp_type)_tol=$(tol_compr)_WII_results.h5")
res_svd =  h5open(outfile);

X_svd_dt_0_0025 = read(res_svd["Observables/X"]);
Y_svd_dt_0_0025 = read(res_svd["Observables/Y"]);
Z_svd_dt_0_0025 = read(res_svd["Observables/Z"]);

XX_svd_dt_0_05 = read(res_svd["Observables/XX"]);


dt = 0.01;
tols = [1e-7 1e-9];

X_svd = zeros(2, 500, 5);
Y_svd = zeros(2, 500, 5);
Z_svd = zeros(2, 500, 5);

for (i, tol) ∈ enumerate(tols)
    outfile = datadir("WII/benchmark/alpha=$(alpha)_N=$(N)_dt=$(dt)_$(comp_type)_tol=$(tol)_WII_results.h5")
    res_svd =  h5open(outfile);
    X_svd[i, :, :] = read(res_svd["Observables/X"]);
    Y_svd[i, :, :] = read(res_svd["Observables/Y"]);
    Z_svd[i, :, :] = read(res_svd["Observables/Z"]);
end

lt = [:dash, :dot, :dashdot, :solid, :auto];


plot()
for (i, tol) ∈ enumerate(tols)
    display(plot!(t_sim, X_svd[i, :, 5], label=["X1 svd ϵ=$(tol)"], ls=lt[i]))
end

plot!(t_sim, X_svd_Ni_10[:, 5], label=["X1 svd ϵ=1e-9 Ni=10"])
plot!(collect(1:500) * 0.01, X_svd_dt_0_01[:, 5], label=["X1 svd ϵ=1e-8 dt=0.01"])
plot!(collect(1:1000) * 0.005, X_svd_dt_0_005[:, 5], label=["X1 svd ϵ=1e-9 dt=0.005"])



##### Var compression #####
comp_type = "var"
outfile = datadir("WII/benchmark/alpha=$(alpha)_N=$(N)_dt=$(dt)_$(comp_type)_tol=$(tol_compr)_WII_results.h5")
res_var =  h5open(outfile);

X_var = read(res_var["Observables/X"]);
Y_var = read(res_var["Observables/Y"]);
Z_var = read(res_var["Observables/Z"]);


##### SVD vs exact #####

plot(t_sim, X_exact[:, 5], name = "exact", lw = 2)
plot!(collect(1:500) * 0.01, X_svd[1, :, 5], name = "ϵ=$(tols[1])", lw = 2, ls = lt[1])
plot!(collect(1:500) * 0.01, X_svd[2, :, 5], name = "ϵ=$(tols[2])", lw = 2, ls = lt[2])
plot!(collect(1:1000) * 0.005, X_svd_dt_0_005[:, 5], name = "ϵ=1e-9 dt=0.005", lw = 2, ls = lt[3])
plot!(collect(1:2000) * 0.0025, X_svd_dt_0_0025[:, 5], name = "ϵ=1e-9 dt=0.0025", lw = 2, ls = lt[4])
#plot!(collect(1:500) * 0.01, X_var[:, 5], name = "var ϵ=$(tols[2])", ls = lt[4])
xlabel!("tJ")
ylabel!("⟨X5⟩")


plot(t_sim, XX_exact[:, 5], name = "exact", lw = 2)
plot!(collect(1:1000) * 0.005, XX_svd_dt_0_005[:, 5], name = "ϵ=1e-9 dt=0.005", lw = 2, ls = lt[1])
plot!(collect(1:100) * 0.05, XX_svd_dt_0_05[:, 5], name = "ϵ=1e-9 dt=0.05", lw = 2, ls = lt[2])
xlabel!("tJ")
ylabel!("⟨X5X6⟩")


####################
# Operator density #
####################

lisa_dir = "/mnt/lisa/TMI/WII/results/"
filename = "rho_alpha=0.4_N=16_t=5.0_dt=0.0025_tol=1.0e-7_Dmax=4096_r_id6.h5"

data = h5open(lisa_dir*filename)
rho_t = [];
for s in 2:381
    push!(rho_t, read(data["rho_l/step_$s"]))
end

rho_n = zeros(16, length(rho_t));
for s in axes(rho_t,1), n ∈ 1:16
    rho_n[n, s] = rho_t[s][n]
end

sum_rho = zeros(length(rho_t))
for s in axes(rho_t,1)
    sum_rho[s] = 1 - sum(rho_t[s])
end

scatter(sum_rho)


plot(t_steps, rho_n[1,:], label = L"\ell = 1")

for n ∈ 2:16
    display(plot!(t_steps, rho_n[n,:], label = L"\ell = %$(n)"))
end

op_size_16 = []
for s in 2:381
    rho_s = read(data["rho_l/step_$(s)"]);
    push!(op_size_16, op_size_v(rho_s));
end



ed_out_16 = h5open(datadir("exact/op_density/op_dens_mfi_nk_n16_a0.4.hdf5"))
t_ed_16 = read(ed_out_16["time"])
rho_16_ed = read(ed_out_16["op_dens_Y"]);
op_size_16_ed = op_size(rho_16_ed);



t_steps = collect(2:381)*0.0025
op_size_16_fig = plot(t_steps, op_size_16, label = "dt=0.001, ϵ = 1e-7", ls = lt[1]);
scatter!(t_ed_16[1:20], abs.(op_size_16_ed)[1:20], label = "ED", markershape = :cross)