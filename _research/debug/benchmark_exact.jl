using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());


using HDF5, LinearAlgebra, BSON, SparseArrays
using Plots, LaTeXStrings
pgfplotsx();

sites = 16;

benchmark_results = h5open("data/benchmark_results_rep.h5", "w");
create_group(benchmark_results, "Fidelities")
create_group(benchmark_results, "States")
close(benchmark_results)



#* ------------------------- alpha = 0.5 -----------------------------------
α = 0.5;
eps_svd = 1e-6;
NSV = 256;
data_exact_0_5 = h5open("data/"*"tek_full_mfi_n16_a0.5_hx-1.05_hz0.5.hdf5", "r");
folder_obelix = "/mnt/obelix/TMI/PL_alpha_0.5/benchmark"
file_root = "final"

obs_tdvp = h5open(folder_obelix*"/finalobservables.h5", "r")
read(obs_tdvp["Diagnostics/Bond dimension"])[1:320]


#* 1) Rebuild states up max step
step_end = 320;
save_each = 5;
step_start = 5;
steps = collect(step_start:save_each:step_end);
Ψ_0_5 = zeros(ComplexF64, length(steps), 2^sites);


for n in 1:length(steps)
    step_n= steps[n];
    println("Rebuilding state of step $(step_n)")
    file_name = "$(folder_obelix)/$(file_root)_state_step=$(step_n).h5"
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    mps_in = read(Ψ_i["mps"]);
    Ψ_0_5[n,:] = reconstructcomplexState(mps_in);
    close(Ψ_i)
end

#* Compare to ncon contraction
n=1
file_name = "$(folder_obelix)/$(file_root)_state_step=5.h5"




include(srcdir("ncon.jl"))

for step =1:5
    file_name = datadir("test_state_step=$(step).h5")
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    mps_in = read(Ψ_i["mps"]);
    Ψ = reconstructcomplexState(mps_in);
    close(Ψ_i)
end


#* Check OTOC results

# Exact results
file_name = datadir("tek_mfi_n16_a1.2_hx-1.05_hz0.5.hdf5")
OTOC = h5open(file_name, "r"); # opens state at step i
phi = read(OTOC["phi"]);
phi_prime = read(OTOC["phi_prime"]);
psi = read(OTOC["psi"]);



# TDVP results
folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "L=16_pl_exp_1.2_Bx=_Bz=_eps_svd=1e-6_OTOC_diagnose_reverse_A_state_step="
TDVP_OTOC_A = h5open(folder_obelix*file_root*"5.h5", "r");
mps_in = read(TDVP_OTOC_A["mps"]);
Ψ_A = reconstructcomplexState(mps_in);
close(TDVP_OTOC_A)

folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "L=16_pl_exp_1.2_Bx=_Bz=_eps_svd=1e-6_OTOC_diagnose_reverse_B_state_step="
TDVP_OTOC_B = h5open(folder_obelix*file_root*"5.h5", "r");
mps_in = read(TDVP_OTOC_B["mps"]);
Ψ_B = reconstructcomplexState(mps_in);
close(TDVP_OTOC_B)


folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/"
file_root = "benchmark_pl_exp_1.2_Bx=-2.1_Bz=1.0_eps_svd=1e-6_rep_2_state_step="
TDVP_for = h5open(folder_obelix*file_root*"320.h5", "r");
mps_in = read(TDVP_for["mps"]);
Ψ_for = reconstructcomplexState(mps_in);


folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "L=16_pl_exp_1.2_Bx=_Bz=_eps_svd=1e-6_OTOC_diagnose_forward_state_step="
TDVP_for_v2 = h5open(folder_obelix*file_root*"320.h5", "r");
mps_in = read(TDVP_for_v2["mps"]);
Ψ_for_v2 = reconstructcomplexState(mps_in);

abs(Ψ_for ⋅ Ψ_for_v2)




#* Compare to exact results
ψt_exact = read(data_exact_0_5["psi"]);

fid_0_5 = zeros(length(steps));
for n in 1:length(steps)
    fid_0_5[n] = abs(Ψ_0_5[n, :] ⋅ ψt_exact[:,5*n+1])
end



benchmark_results["Fidelities/fid_pl_exp=$(α)_eps_svd=$(eps_svd)_NSV=$(NSV)"] = fid_0_5
benchmark_results["States/psi_pl_exp=$(α)_eps_svd=$(eps_svd)_NSV=$(NSV)"] = Ψ_0_5



#* ------------------------- alpha = 1.2 -----------------------------------
α = 1.2;
eps_svd = 1e-6;
NSV = 256;
data_exact_1_2 = h5open("data/"*"tek_full_mfi_n16_a1.2_hx-1.05_hz0.5.hdf5", "r");
folder_obelix = "/mnt/obelix/TMI/PL_alpha_$(α)"
file_root = "benchmark_pl_exp_$(α)_Bx=-2.1_Bz=1.0_eps_svd=1e-6_rep_2"
parms_tdvp = load(folder_obelix*"/benchmark_pl_exp_1.2_Bx=-2.1_Bz=1.0_eps_svd=1e-6_rep_2parameters.bson")

#* 1) Rebuild states up max step
step_end = 138;
save_each = 1;
steps = collect(1:save_each:step_end);
Ψ_1_2 = zeros(ComplexF64, length(steps), 2^sites);

for n in 1:length(steps)
    step_n= steps[n];
    println("Rebuilding state of step $(step_n)")
    file_name = "$(folder_obelix)/$(file_root)_state_step=$(step_n).h5"
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    mps_in = read(Ψ_i["mps"]);
    Ψ_1_2[n,:] = reconstructcomplexState(mps_in);
    close(Ψ_i)
end




#* Compare to exact results
ψt_exact = read(data_exact_1_2["psi"]);

fid_1_2 = zeros(length(steps));
for n in 1:length(steps)
    fid_1_2[5*n] = abs(Ψ_1_2[n, :] ⋅ ψt_exact[:,5*n+1])
end

benchmark_results["Fidelities/fid_pl_exp=$(α)_eps_svd=$(eps_svd)_NSV=$(NSV)"] = fid_1_2
benchmark_results["States/psi_pl_exp=$(α)_eps_svd=$(eps_svd)_NSV=$(NSV)"] = Ψ_1_2




#* ------------------------- alpha = 3.0 -----------------------------------
α = 3.0;
eps_svd = 1e-6;
NSV = 256;
data_exact_3_0 = h5open("data/"*"tek_full_mfi_n16_a3.0_hx-1.05_hz0.5.hdf5", "r");
folder_obelix = "/mnt/obelix/TMI/PL_alpha_$(α)"
file_root = "benchmark_pl_exp_$(α)_Bx=-2.1_Bz=1.0_eps_svd=1e-6_rep_2"

#* 1) Rebuild states up max step
step_end = 65;
save_each = 1;
steps = collect(1:save_each:step_end);
Ψ_3_0 = zeros(ComplexF64, length(steps), 2^sites);

for n in 1:length(steps)
    step_n= steps[n];
    println("Rebuilding state of step $(step_n)")
    file_name = "$(folder_obelix)/$(file_root)_state_step=$(step_n).h5"
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    mps_in = read(Ψ_i["mps"]);
    Ψ_3_0[n,:] = reconstructcomplexState(mps_in);
    close(Ψ_i)
end


#* Compare to exact results
ψt_exact = read(data_exact_3_0["psi"]);

fid_3_0 = zeros(length(steps));
for n in 1:length(steps)
    fid_3_0[5*n] = abs(Ψ_3_0[n, :] ⋅ ψt_exact[:,5*n+1])
end

benchmark_results["Fidelities/fid_pl_exp=$(α)_eps_svd=$(eps_svd)_NSV=$(NSV)"] = fid_3_0
benchmark_results["States/psi_pl_exp=$(α)_eps_svd=$(eps_svd)_NSV=$(NSV)"] = Ψ_3_0






#* ------------------------- alpha = 0.5 -----------------------------------

y_pol_initial = h5open("data/"*"Y_pol_initian_state.h5", "r")
data_exact = h5open("data/"*"tek_full_mfi_n16_a0.5_hx-1.05_hz0.5.hdf5", "r")
Ψ_exact = read(data_exact["psi"]);
pl_exp = 1.2;

mps_initial = read(y_pol_initial["mps"])

Y_pol_reverse = reconstructcomplexState(mps_initial);

σʸ = sparse([0 -im; im 0])
function σʸᵢ(i,N)
    σʸ = sparse([0 -im; im 0]);
    II = sparse([1 0; 0 1]);
    return kron([n==i ? σʸ : II for n in N:-1:1]...)
end
Sʸᵢ(i,N) = 0.5*σʸᵢ(i,N);
λ_y1, ϕ_y1 = eigen(Matrix(σʸ))
ϕ_y = ϕ_y1[:,2];

ϕ_y
ψ_y_pol = ϕ_y ⊗ ϕ_y ⊗  ϕ_y ⊗ ϕ_y ⊗ ϕ_y ⊗ ϕ_y ⊗  ϕ_y ⊗ ϕ_y ⊗ ϕ_y ⊗ ϕ_y ⊗  ϕ_y ⊗ ϕ_y ⊗ ϕ_y ⊗ ϕ_y ⊗  ϕ_y ⊗ ϕ_y

abs(ψ_y_pol ⋅ Ψ_exact[:,1])
abs(ψ_y_pol ⋅ Y_pol_reverse)
abs(Ψ_exact[:,1] ⋅ Y_pol_reverse)


for i ∈ 1:16
    println("Exact", ψ_y_pol[:,1]'*(Sʸᵢ(i,16)* ψ_y_pol[:,1]);)
    println("Exact DW", Ψ_exact[:,1]'*(Sʸᵢ(i,16)* Ψ_exact[:,1]);)
    println("MPS", Y_pol_reverse'*(Sʸᵢ(i,16)* Y_pol_reverse);)
end


#* ------------------------- ϵ_svd = 0.0 -----------------------------------

folder = "/mnt/lisa/PL_alpha_0.5/NOTRUNC"
eps_svd = 0.0;
data_tdvp = h5open("$(folder)/PL_alpha_$(pl_exp)_eps_svd=$(eps_svd).h5", "r"); #creates the collection of states
obs_tdvp = h5open(folder*"/L=16_t=16_Bx=0.5_Bz=0.5_NSV=256_eps_svd=0.0.h5", "r")
parms_tdvp = load(folder*"/L=16_t=16_Bx=0.5_Bz=0.5_NSV=256_eps_svd=0.0.bson")


Ψ_tdvp_reverse = read(data_tdvp["states/reverse"]);
Ψ_tdvp_transpose = read(data_tdvp["states/transpose"]);

results_pl_alpha_0_5 = Dict();
results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_reverse"] = zeros(64);
results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"] = zeros(64);

m=1
for n ∈ 5:5:320
    results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_reverse"][m] = abs(Ψ_exact[:,n] ⋅ Ψ_tdvp_reverse[m,:])
    results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"][m] = abs(Ψ_exact[:,n] ⋅Ψ_tdvp_transpose[m,:])
    m+=1
end


results_pl_alpha_0_5["Sz_alpha=$(pl_exp)_eps_svd=$(eps_svd)"] = read(obs_tdvp["Magnetization/Sz"])
results_pl_alpha_0_5["Sy_alpha=$(pl_exp)_eps_svd=$(eps_svd)"] = read(obs_tdvp["Magnetization/Sy"])
results_pl_alpha_0_5["Sx_alpha=$(pl_exp)_eps_svd=$(eps_svd)"] = read(obs_tdvp["Magnetization/Sx"])



#* ---------------- alpha = 0.5, ϵ_svd = 1e-6 -----------------------------------

eps_svd = 1e-6;
folder = "/mnt/obelix/TMI/PL_alpha_0.5"
data_tdvp = h5open("$(folder)/PL_alpha_$(pl_exp)_eps_svd=$(eps_svd).h5", "r"); #creates the collection of states
obs_tdvp = h5open(folder*"/L=16_t=16_pl_exp=1.2_Bx=1.05_Bz=-0.5_NSV=256_eps_svd=1e-6observables.h5", "r")

Ψ_tdvp_reverse = read(data_tdvp["states/reverse"]);
Ψ_tdvp_transpose = read(data_tdvp["states/transpose"]);


results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_reverse"] = zeros(64);
results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"] = zeros(64);

results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"]

m=1
for n ∈ 5:5:320
    results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_reverse"][m] = abs(Ψ_exact[:,n] ⋅ Ψ_tdvp_reverse[m,:])
    results_pl_alpha_0_5["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"][m] = abs(Ψ_exact[:,n] ⋅Ψ_tdvp_transpose[m,:])
    m+=1
end

results_pl_alpha_0_5["Sz_alpha=$(pl_exp)_eps_svd=$(eps_svd)"] = read(obs_tdvp["Magnetization/Sz"])
results_pl_alpha_0_5["Sy_alpha=$(pl_exp)_eps_svd=$(eps_svd)"] = read(obs_tdvp["Magnetization/Sy"])
results_pl_alpha_0_5["Sx_alpha=$(pl_exp)_eps_svd=$(eps_svd)"] = read(obs_tdvp["Magnetization/Sx"])




#* ------------------------------------ α = 1.2 -----------------------------------------------------------------

params_tdvp = load(folder*"/L=16_t=16_pl_exp=1.2_Bx=1.05_Bz=-0.5_NSV=256_eps_svd=1e-6parameters.bson")


eps_svd = 1e-6;
folder = "/mnt/obelix/TMI/PL_alpha_1.2"
data_tdvp = h5open("$(folder)/PL_alpha_$(pl_exp)_eps_svd=$(eps_svd).h5", "r");
data_exact = h5open("data/"*"tek_full_mfi_n16_a1.2_hx-1.05_hz0.5.hdf5", "r")

Ψ_tdvp_reverse = read(data_tdvp["states/reverse"]);
Ψ_tdvp_transpose = read(data_tdvp["states/transpose"]);

results_pl_alpha_1_2 = Dict();
results_pl_alpha_1_2["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_reverse"] = zeros(64);
results_pl_alpha_1_2["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"] = zeros(64);

results_pl_alpha_1_2["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"]

m=1
for n ∈ 5:5:320
    println(m)
    results_pl_alpha_1_2["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_reverse"][m] = abs(Ψ_exact[:,n+1] ⋅ Ψ_tdvp_reverse[m,:])
    results_pl_alpha_1_2["fid_alpha=$(pl_exp)_eps_svd=$(eps_svd)_transpose"][m] = abs(Ψ_exact[:,n+1] ⋅Ψ_tdvp_transpose[m,:])
    m+=1
end





#* ------------------------------------- Observables ------------------------------------------------------------

time_steps = 0.05*collect(5:5:320)
markers = [:cross, :xcross, :diamond, :utriangle, :circle];

eps_svd = 0.0;
Sy_notrunc = scatter(time_steps, results_pl_alpha_0_5["Sy_alpha=$(pl_exp)_eps_svd=$(eps_svd)"][1,1:64], label="site 1", markershape = markers[1], msc=1)
scatter!(time_steps, results_pl_alpha_0_5["Sy_alpha=$(pl_exp)_eps_svd=$(eps_svd)"][4,1:64], label="site 4", markershape = markers[2], msc=2)
scatter!(time_steps, results_pl_alpha_0_5["Sy_alpha=$(pl_exp)_eps_svd=$(eps_svd)"][8,1:64], label="site 8", markershape = markers[3], msc=3)
scatter!(xlabel=L"$t/J$", ylabel=L"$\langle \sigma_y^{(i)} \rangle$",thickness_scaling=1.5)
savelatexfig(Sy_notrunc, plotsdir("benchmark","PL_alpha_0.5/"), "Y_eps_svd_0.0")


eps_svd = 1e-6;
Sz_trunc = scatter(time_steps, results_pl_alpha_0_5["Sz_alpha=$(pl_exp)_eps_svd=$(eps_svd)"][1,1:64], label="site 1", markershape = markers[1])
scatter!(time_steps, results_pl_alpha_0_5["Sz_alpha=$(pl_exp)_eps_svd=$(eps_svd)"][16,1:64], label="site 4", markershape = markers[2])
scatter!(time_steps, results_pl_alpha_0_5["Sz_alpha=$(pl_exp)_eps_svd=$(eps_svd)"][8,1:64], label="site 8", markershape = markers[3])


parms_tdvp[:Args]
hcat(parms_tdvp[:coupling_matrix]...)-read(obs_tdvp["Parameters/Jij"])


obs_tdvp

ent_spec = read(obs_tdvp["Entanglement_Spectrum"])

ent_spec320 = ent_spec["step = 1595"]

entr_tdvp = read(obs_tdvp["Entropy/Bond entropy"])

function entropy(entang_spectrum)
    ent_entropy = Float64[];
    for n ∈ 1:15
        es = filter(!iszero, entang_spectrum[:,n])
        push!(ent_entropy, -1*sum([wᵢ*log2(wᵢ) for wᵢ ∈ es]))
    end
    return ent_entropy
end

entr_tdvp_rec_320 = vNeumannnentropy(ent_spec320)
entr_tdvp_rec_1 = vNeumannnentropy(ent_spec1)

function vNeumannnentropy(entang_spectrum)
    ent_entropy = Float64[];
    for n ∈ 1:15
        es = filter(!iszero, entang_spectrum[:,n])
        push!(ent_entropy, -1*sum([wᵢ^2*log(wᵢ^2) for wᵢ ∈ es]))
    end
    return ent_entropy
end


benchmark_results = h5open("data/benchmark_results_rep.h5", "r");
fid_0_5 = read(benchmark_results["Fidelities/fid_pl_exp=0.5_eps_svd=1.0e-6_NSV=256"])
fid_1_2 = read(benchmark_results["Fidelities/fid_pl_exp=1.2_eps_svd=1.0e-6_NSV=256"])
fid_3_0 = read(benchmark_results["Fidelities/fid_pl_exp=3.0_eps_svd=1.0e-6_NSV=256"])
x_data = collect(0.25:0.25:16)


scatter(x_data, log10.(-fid_0_5[5:5:320].+1), label=L"$\xi = 0.5$")
xlabel!("t/J")
ylabel!(L"$\log (1-\langle \Psi | \Psi^\prime \rangle)$")

scatter(x_data, log10.(-fid_1_2[5:5:320].+1), label=L"$\xi = 1.2$")
xlabel!("t/J")
ylabel!(L"$\log (1-\langle \Psi | \Psi^\prime \rangle)$")

scatter(x_data, log10.(-fid_3_0[5:5:320].+1), label=L"$\xi = 3.0$")
xlabel!("t/J")
ylabel!(L"$\log (1-\langle \Psi | \Psi^\prime \rangle)$")