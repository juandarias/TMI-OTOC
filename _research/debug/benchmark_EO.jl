using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());


using HDF5, LinearAlgebra, SparseArrays
include(srcdir("MPS_conversions.jl"));
include(srcdir("ncon.jl"));
include(plotsdir("plotting_functions.jl"));




#* Methods

function loadstate(folder, file_name, step)
    data = h5open(folder*file_name*"_state_step=$(step).h5", "r");
    state = reconstructcomplexState(read(data["mps"]));
    return state
end

function readBonddimension(path)
    data = h5open(path*"_observables.h5","r")
    return read(data["Diagnostics/Bond dimension"])    
end

function readEntaglement(path)
    data = h5open(path*"_observables.h5","r")
    return read(data["Entropy/Bond entropy"])    
end

fid(phi,psi) = abs(phi ⋅ psi)

#= 
function MPS_tensors(mps)
    
    #* Sort A matrices by lattice site. Sites are labelled from right to left in exported states
    sites = length(mps) ÷ 4;
    keys_mps = collect(keys(mps));
    index_key_A(A_key) = findfirst(x-> x==A_key, keys_mps)

    Atensors = Array{ComplexF64}[];
    for site ∈ sites-1:-1:0
        key_1 = string(site)*"_1_((),())"
        key_0 = string(site)*"_0_((),())"
        AiRe_0_A = index_key_A(key_0*"Re");
        AiIm_0_A = index_key_A(key_0*"Im");
        AiRe_1_A = index_key_A(key_1*"Re");
        AiIm_1_A = index_key_A(key_1*"Im");
        Acomplex_0 = im*mps[keys_mps[AiIm_0_A]] + mps[keys_mps[AiRe_0_A]]; #Re[A] + i Im[A]
        Acomplex_1 = im*mps[keys_mps[AiIm_1_A]] + mps[keys_mps[AiRe_1_A]]; #Re[A] + i Im[A]
        
        
        if site == sites-1
            push!(Atensors, vcat(Acomplex_0,Acomplex_1)); #first site
        elseif site == 0
            push!(Atensors, hcat(Acomplex_0,Acomplex_1)); #last site
        else
            Ai = zeros(ComplexF64, 2,size(Acomplex_0)...)
            Ai[1,:,:] = Acomplex_0;
            Ai[2,:,:] = Acomplex_1;
            push!(Atensors, Ai)
        end
    end
    
    return Atensors
end


function MPS_fidelity(tensors_A, tensors_B) 
    
    #* Contract tensors
    sites = length(tensors_A);
    tensors = [tensors_A..., conj.(tensors_B)...]
    pInt(AS) = parse.(Int, AS)
    IndexArray_Ket = [pInt(["9"*"$n", "7"*string(n-1), "7"*"$n"]) for n ∈ 2:sites-1]
    prepend!(IndexArray_Ket, [[91,71]]); #first site 
    append!(IndexArray_Ket, [pInt(["7"*string(sites-1), "9"*"$sites"])]); #first site 

    IndexArray_Bra = [pInt(["9"*"$n", "8"*string(n-1), "8"*"$n"]) for n ∈ 2:sites-1]
    prepend!(IndexArray_Bra, [[91,81]]); #first site 
    append!(IndexArray_Bra, [pInt(["8"*string(sites-1), "9"*"$sites"])]); #first site 
    
    IndexArray = [IndexArray_Ket..., IndexArray_Bra...]

    fid = ncon(tensors, IndexArray);
    return abs(fid)
end;

 =#
#******************************** N=16 benchmark **********************************#

ΔS = 0.05;
folder_adaptive = "/mnt/obelix/TMI/diagnosis/ALGS/ADAPTIVE/N16_Benchmark/"
root_adaptive = "DeltaS=$ΔS"

folder_single = "/mnt/obelix/TMI/diagnosis/ALGS/SINGLE/N16_Benchmark/"
root_single = "Nsv_256"
root_single_new = "Nsv_new"
root_single_AFM = "Nsv256_new_AFM"

folder_double = "/mnt/obelix/TMI/diagnosis/ALGS/ADAPTIVE/N16_Benchmark/"
root_double = "two_site_ALG_reference"

data_exact = h5open("data/"*"tek_full_mfi_n16_a0.5_hx-1.05_hz0.5.hdf5", "r");
ψt_exact = read(data_exact["psi"]);

#***** Forward Evolution *****#

#** Single-site TDVP
step_final = 160
fid_single = zeros(32);
fid_single_new = zeros(32);
i=1
for step_n ∈ 5:5:step_final
    file_opt = "$(folder_single)/$(root_single)_forward_state_step=$(step_n).h5"
    psi_opt = h5open(file_opt, "r"); # opens state at step i
    mps_opt = read(psi_opt["mps"]);
    psi_t_mps = reconstructcomplexState(mps_opt)
    fid_single[i] = abs(psi_t_mps ⋅ ψt_exact[:,step_n+1])    
    close(psi_opt);
    file_new = "$(folder_single)/$(root_single_new)_forward_state_step=$(step_n).h5"
    psi_new = h5open(file_new, "r"); # opens state at step i
    mps_new = read(psi_new["mps"]);
    psi_t_mps = reconstructcomplexState(mps_new)
    fid_single_new[i] = abs(psi_t_mps ⋅ ψt_exact[:,step_n+1])
    i+=1;
    close(psi_new);
end

step_final = 160
fid_single_MPS = zeros(32);
i=1
for step_n ∈ 5:5:step_final
    file_ref = "$(folder_single)/$(root_single)_forward_state_step=$(step_n).h5"
    file_new = "$(folder_single)/$(root_single_new)_forward_state_step=$(step_n).h5"
    data_ref = h5open(file_ref, "r"); # opens state at step i
    data_new = h5open(file_new, "r"); # opens state at step i
    tensors_ref = MPS_tensors(read(data_ref["mps"]));
    tensors_new = MPS_tensors(read(data_new["mps"]));
    fid_single_MPS[i] = MPS_fidelity(tensors_new, tensors_ref);
    i+=1;
end




#** Two-site TDVP
step_final = 160
fid_double = zeros(32);
i=1
for step_n ∈ 5:5:step_final
    file_opt = "$(folder_double)/$(root_double)_forward_state_step=$(step_n).h5"
    psi_opt = h5open(file_opt, "r"); # opens state at step i
    mps_opt = read(psi_opt["mps"]);
    psi_t_mps = reconstructcomplexState(mps_opt)
    fid_double[i] = abs(psi_t_mps ⋅ ψt_exact[:,step_n+1])
    close(psi_opt);
    i+=1;
end

#** Adaptive TDVP
ΔS = [0.05, 0.1, 0.3];
root_adaptive = "DeltaS=$ΔS"

step_final = 160
fid_adaptive = zeros(3,32);

for j ∈ 1:3
    i=1;
    root_adaptive = "DeltaS=$(ΔS[j])"
    for step_n ∈ 5:5:step_final
        file_opt = "$(folder_adaptive)/$(root_adaptive)_forward_state_step=$(step_n).h5"
        psi_opt = h5open(file_opt, "r"); # opens state at step i
        mps_opt = read(psi_opt["mps"]);
        psi_t_mps = reconstructcomplexState(mps_opt)
        fid_adaptive[j,i] = abs(psi_t_mps ⋅ ψt_exact[:,step_n+1])
        close(psi_opt);
        i+=1;
    end
end


#***** Backward Evolution: using single-site as reference *****#
#*** OTOC A,B: two-site 
step_final = 160
fid_double_OTOC_A = zeros(32);
i=1
for step_n ∈ 5:5:step_final
    file_opt = "$(folder_double)$(root_double)_reverse_A_state_step=$(step_n).h5"
    file_ref = "$(folder_single)$(root_single)_reverse_A_state_step=$(step_n).h5"
    data_opt = h5open(file_opt, "r"); # opens state at step i
    data_ref = h5open(file_ref, "r"); # opens state at step i
    tensors_opt = MPS_tensors(read(data_opt["mps"]));
    tensors_ref = MPS_tensors(read(data_ref["mps"]));
    fid_double_OTOC_A[i] = MPS_fidelity(tensors_opt, tensors_ref);
    i+=1;
end


#*** OTOC A,B: Adaptive 
ΔS = [0.05, 0.1, 0.3];
step_final = 160
fid_adaptive_OTOC_B = zeros(3,32);

for j ∈ 1:3
    i=1
    root_adaptive = "DeltaS=$(ΔS[j])"
    for step_n ∈ 5:5:step_final
        file_opt = "$(folder_adaptive)/$(root_adaptive)_reverse_B_state_step=$(step_n).h5"
        file_ref = "$(folder_single)/$(root_single)_reverse_B_state_step=$(step_n).h5"
        data_opt = h5open(file_opt, "r"); # opens state at step i
        data_ref = h5open(file_ref, "r"); # opens state at step i
        tensors_opt = MPS_tensors(read(data_opt["mps"]));
        tensors_ref = MPS_tensors(read(data_ref["mps"]));
        fid_adaptive_OTOC_B[j,i] = MPS_fidelity(tensors_opt, tensors_ref);
        i+=1;
    end
end

#*** Fidelities AFM vs FM
step_final = 160
fid_AFM = zeros(32);
i=1
for step_n ∈ 5:5:step_final
    file_AFM = "$(folder_single)$(root_single_new)_forward_state_step=$(step_n).h5"
    file_ref = "$(folder_single)$(root_single_AFM)_forward_state_step=$(step_n).h5"
    data_AFM = h5open(file_AFM, "r"); # opens state at step i
    data_ref = h5open(file_ref, "r"); # opens state at step i
    tensors_AFM = MPS_tensors(read(data_AFM["mps"]));
    tensors_ref = MPS_tensors(read(data_ref["mps"]));
    fid_AFM[i] = MPS_fidelity(tensors_AFM, tensors_ref);
    i+=1;
end

scatter(collect(0.05:0.25:8),fid_AFM)
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"$\langle \Psi_{old}|\Psi_{new}\rangle$")

#***************** Half-chain entropies *****************


S_single_forward = readEntaglement(folder_single*root_single*"_forward")
S_single_forward_AFM = readEntaglement(folder_single*root_single_AFM*"_forward")

S_single_reverse = readEntaglement(folder_single*root_single*"_reverse_A")

S_double_forward = readEntaglement(folder_double*root_double*"_forward")
S_double_reverse = readEntaglement(folder_double*root_double*"_reverse_A")

S_adaptive_forward = Matrix{Float64}[]
S_adaptive_reverse = Matrix{Float64}[]
for j ∈ 1:3
    root_adaptive = "DeltaS=$(ΔS[j])"
    push!(S_adaptive_forward, readEntaglement(folder_adaptive*root_adaptive*"_forward"))
    push!(S_adaptive_reverse, readEntaglement(folder_adaptive*root_adaptive*"_reverse_A"))
end


#***************** Bond dimension *****************
D_double_forward = readBonddimension(folder_double*root_double*"_forward")
D_adaptive_forward = Vector{Int32}[]
D_adaptive_reverse = Vector{Int32}[]
for j ∈ 1:3
    root_adaptive = "DeltaS=$(ΔS[j])"
    push!(D_adaptive_forward, readBonddimension(folder_adaptive*root_adaptive*"_forward"))
    push!(D_adaptive_reverse, readBonddimension(folder_adaptive*root_adaptive*"_reverse_A"))
end




#******************************** N=16, new framework **********************************#

folder_single = "/mnt/obelix/TMI/diagnosis/ALGS/SINGLE/N16_Benchmark/"
root_single = "Nsv_256"
root_single_new = "Nsv_new"

step_final = 160
fid_single_n16 = zeros(32);
i=1
for step_n ∈ 5:5:step_final
    file_new = "$(folder_single)/$(root_single_new)_forward_state_step=$(step_n).h5"
    file_ref = "$(folder_single)/$(root_single)_forward_state_step=$(step_n).h5"
    data_new = h5open(file_new, "r"); # opens state at step i
    data_ref = h5open(file_ref, "r"); # opens state at step i
    tensors_new = MPS_tensors(read(data_new["mps"]));
    tensors_ref = MPS_tensors(read(data_ref["mps"]));
    fid_single_n16[i] = MPS_fidelity(tensors_new, tensors_ref);
    i+=1;
end



#******************************** N=32, new framework **********************************#

#************** Single site results ********************#
##region
folder_single = "/mnt/obelix/TMI/L32/alpha_3.0/NSV_256/"
root_single = "single_site"

folder_single_new = "/mnt/obelix/TMI/L32/alpha_3.0/NSV_256/"
root_single_new = "single_site_new"


###* Fidelities
step_final = 100
fid_forward_alpha3_0 = zeros(50);
i=1
for step_n ∈ 1:2:step_final
    file_new = "$(folder_single_new)/$(root_single_new)_forward_state_step=$(step_n).h5"
    file_ref = "$(folder_single)/$(root_single)_forward_state_step=$(step_n).h5"
    data_new = h5open(file_new, "r"); # opens state at step i
    data_ref = h5open(file_ref, "r"); # opens state at step i
    #psi_t_ref = reconstructcomplexState(read(data_ref["mps"]))
    #psi_t_new = reconstructcomplexState(read(data_new["mps"]))
    tensors_new = MPS_tensors(read(data_new["mps"]));
    tensors_ref = MPS_tensors(read(data_ref["mps"]));
    fid_forward_alpha3_0[i] = MPS_fidelity(tensors_new, tensors_ref);
    #fid_forward_rep[i] = fid(psi_t_ref, psi_t_new);
    i+=1;
    #close(data_new);close(data_ref);
end


fid_L32_alpha3_0_NSV256 = scatter(collect(0.1:0.2:10), -fid_forward_alpha3_0.+1, yscale=:log10)
scatter!(thickness_scaling=1.5);
xlabel!("t/J"); ylabel!(L"$1-\langle \Psi_{old}|\Psi_{new}\rangle$");
title!(L"$N=32, \; \alpha=3.0, \; D_{max}=256 $. 1-site algorithm")
savelatexfig(fid_L32_alpha3_0_NSV256, plotsdir("benchmark/UPDATE/N32_alpha3_0_NSV256_SS_fid"), tex=true)


file_new = "$(folder_single_new)/$(root_single_new)_forward_observables.h5"
file_ref = "$(folder_single)/$(root_single)_forward_observables.h5"


function readMagnetization(path)
    data = h5open(path*"_observables.h5", "r"); # opens state at step i
    Sx = read(data["Magnetization/Sx"])
    Sy = read(data["Magnetization/Sy"])
    Sz = read(data["Magnetization/Sz"])
    return Sx, Sy, Sz
end

S_new = readMagnetization(file_new)
S_ref = readMagnetization(file_ref)


Sx_L32_alpha3_0_NSV256 = scatter(collect(0.1:0.1:10), S_new[1][1,1:100], label=L"$S^{1}_x$ new", shape=:cross, msc=1);
scatter!(collect(0.1:0.1:10), S_ref[1][1,1:100], label=L"$S^{1}_x$", shape=:xcross, msc=2);
scatter!(collect(0.1:0.1:10), S_new[1][16,1:100], label=L"$S^{16}_x$ new", shape=:cross, msc=3);
scatter!(collect(0.1:0.1:10), S_ref[1][16,1:100], label=L"$S^{16}_x$", shape=:xcross, msc=4);
scatter!(thickness_scaling=1.5);
xlabel!("t/J"); ylabel!(L"\langle S^{(i)}_x \rangle");
title!(L"$N=32, \; \alpha=3.0, \; D_{max}=256$ 1-site algorithm")
savelatexfig(Sx_L32_alpha3_0_NSV256, plotsdir("benchmark/UPDATE/N32_alpha3_0_NSV256_SS_Sx"), tex=true)



###* Entanglement
S_single_new = readEntaglement(folder_single_new*root_single_new*"_forward")
S_single_ref = readEntaglement(folder_single*root_single*"_forward")

S_L32_alpha3_0_NSV256 = scatter(collect(0.1:0.1:10), S_single_new[16,1:100], label="new", shape=:cross, msc=1)
scatter!(collect(0.1:0.1:10), S_single_ref[16,1:100], label="old", shape=:xcross, msc=2)
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"S_{16}");
title!(L"$N=32, \; \alpha=3.0, \; D_{max}=256 $. 1-site algorithm")
savelatexfig(S_L32_alpha3_0_NSV256, plotsdir("benchmark/UPDATE/N32_alpha3_0_NSV256_SS_S"), tex=true)


###* Bond dimension
D_single_ref = readBonddimension(folder_single*root_single*"_forward")
D_single_new = readBonddimension(folder_single_new*root_single_new*"_forward")

##endregion

#************** Convergence results ********************#
###* α = 3.5 *###
##region
#! Old framework: 520864
#! New framework: 524078 (ΔS=0.05; 1e-12), 524077 (1e-12), 524034 (1e-10), 523908(1e-6), 523909(1e-7), 523909(1e-8)

#! α=2.5= 523909(1e-8)

folder_adaptive = "/mnt/obelix/TMI/L32/alpha_3.5/NSV_256/"
folder_adaptive_new = "/mnt/obelix/TMI/L32/alpha_3.5/NSV_256/"

folder_single_lisa = "/mnt/lisa/L32/alpha_3.5/NSV_400/"
folder_single = "/mnt/obelix/TMI/L32/alpha_3.5/NSV_256/"
root_single = "single_site"

#* Fidelities
step_final = 100
fid_forward_alpha3_5_single = zeros(20);
i=1
for step_n ∈ 1:5:step_final
    file_NSV400 = "$(folder_single_lisa)/$(root_single)_forward_state_step=$(step_n).h5"
    file_NSV256 = "$(folder_single)/$(root_single)_forward_state_step=$(step_n).h5"
    data_NSV400 = h5open(file_NSV400, "r"); # opens state at step i
    data_NSV256 = h5open(file_NSV256, "r"); # opens state at step i
    tensors_NSV400 = MPS_tensors(read(data_NSV400["mps"]));
    tensors_NSV256 = MPS_tensors(read(data_NSV256["mps"]));
    fid_forward_alpha3_5_single[i] = MPS_fidelity(tensors_NSV400, tensors_NSV256);
    i+=1;
    println(step_n)
end


folder_single_lisa = "/mnt/lisa/L32/alpha_3.0/NSV_400/"
folder_single = "/mnt/obelix/TMI/L32/alpha_3.0/NSV_256/"
root_single = "single_site"


fid_forward_alpha3_0_single = zeros(20);
i=1
for step_n ∈ 1:5:step_final
    file_NSV400 = "$(folder_single_lisa)/$(root_single)_forward_state_step=$(step_n).h5"
    file_NSV256 = "$(folder_single)/$(root_single)_forward_state_step=$(step_n).h5"
    data_NSV400 = h5open(file_NSV400, "r"); # opens state at step i
    data_NSV256 = h5open(file_NSV256, "r"); # opens state at step i
    tensors_NSV400 = MPS_tensors(read(data_NSV400["mps"]));
    tensors_NSV256 = MPS_tensors(read(data_NSV256["mps"]));
    fid_forward_alpha3_0_single[i] = MPS_fidelity(tensors_NSV400, tensors_NSV256);
    i+=1;
    println(step_n)
end

fid_SS_alpha_3_3_5 = scatter(collect(0.5:0.5:10), -fid_forward_alpha3_0_single.+1, label=L"$\alpha=3.0$", marker=:cross, yscale=:log10, msc=:auto);
scatter!(collect(0.5:0.5:10), -fid_forward_alpha3_5_single.+1, label=L"$\alpha=3.5$", marker=:xcross, yscale=:log10, msc=:auto)
scatter!(thickness_scaling=1.5);
xlabel!("t/J"); ylabel!(L"$1-\langle \tilde{\Psi}|\Psi\rangle$");
title!(L"$N=32, \; \tilde{D}_{max}=400, \; D_{max}=256 $. 1-site algorithm")
savelatexfig(fid_SS_alpha_3_3_5, plotsdir("benchmark/ALGS/N32_alpha3_5_3_0_SS_conv_fid"), tex=true)

#* Entropies
S_single_new = readEntaglement(folder_adaptive*root_single*"_forward")
S_single_NSV400 = readEntaglement(folder_single_lisa*root_single*"_forward")

ΔS = [0.1, 0.05];
S_adaptive_ref = [];
for n = 1:2
    root_adaptive = "adaptive_DeltaS=$(ΔS[n])"
    push!(S_adaptive_ref, readEntaglement(folder_adaptive*root_adaptive*"_forward"))
end

eps_tw = ["1e-6", "1e-7", "1e-8", "1e-10", "1e-12"]
S_adaptive_new = [];
root_adaptive_new = "adaptive_DeltaS=0.1_rep"
push!(S_adaptive_new,readEntaglement(folder_adaptive_new*root_adaptive_new*"_forward"))

for n = 2:5
    root_adaptive_new = "adaptive_DeltaS=0.1_eps_tw_$(eps_tw[n])"
    push!(S_adaptive_new, readEntaglement(folder_adaptive_new*root_adaptive_new*"_forward"))
end

root_adaptive_new = "adaptive_DeltaS=0.05_eps_tw_1e-12"
push!(S_adaptive_new, readEntaglement(folder_adaptive_new*root_adaptive_new*"_forward"))


S_N32_alpha3_5_NSV_256_adap = scatter(collect(0.1:0.1:10), S_adaptive_ref[1][16,1:100], label=L"ref: $\Delta S/S =0.1,\; \epsilon_{svd}=1^{-6}$",shape=:cross);
scatter!(collect(0.1:0.1:10), S_adaptive_ref[2][16,1:100], label=L"ref: $\Delta S/S =0.05,\; \epsilon_{svd}=1^{-6}$",shape=:cross, msc=6);
scatter!(collect(0.1:0.1:10), S_adaptive_new[1][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-6}$",shape=:xcross, msc=1);
scatter!(collect(0.1:0.1:10), S_adaptive_new[2][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-7}$",shape=:xcross, msc=2);
scatter!(collect(0.1:0.1:10), S_adaptive_new[3][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-8}$",shape=:xcross, msc=3);
scatter!(collect(0.1:0.1:10), S_adaptive_new[4][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-10}$",shape=:xcross, msc=4);
scatter!(collect(0.1:0.1:10), S_adaptive_new[5][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-12}$",shape=:xcross, msc=5);
scatter!(collect(0.1:0.1:10), S_adaptive_new[6][16,1:100], label=L"$\Delta S/S =0.05,\; \epsilon_{tW}=1^{-12}$",shape=:diamond, msc=6);
scatter!(collect(0.1:0.1:10), S_single_new[16,1:100], label="1-site",shape=:utriangle, msc=7)
scatter!(collect(0.1:0.1:10), S_single_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=8)
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"S_{16}")
title!(L"$N=32, \; \alpha=3.5, \; D_{max}=256$. Adaptive")
savelatexfig(S_N32_alpha3_5_NSV_256_adap, plotsdir("benchmark/ALGS/N32_alpha3_5_conv_S"), tex=true)


#* Magnetizations
M_single_new = readMagnetization(folder_adaptive*root_single*"_forward")
M_single_NSV400 = readMagnetization(folder_single_lisa*root_single*"_forward")

ΔS = [0.1, 0.05];
M_adaptive_ref = [];
for n = 1:2
    root_adaptive = "adaptive_DeltaS=$(ΔS[n])"
    push!(M_adaptive_ref, readMagnetization(folder_adaptive*root_adaptive*"_forward"))
end

eps_tw = ["1e-6", "1e-7", "1e-8", "1e-10", "1e-12"]
M_adaptive_new = [];
root_adaptive_new = "adaptive_DeltaS=0.1_rep"
push!(M_adaptive_new,readMagnetization(folder_adaptive_new*root_adaptive_new*"_forward"))

for n = 2:5
    root_adaptive_new = "adaptive_DeltaS=0.1_eps_tw_$(eps_tw[n])"
    push!(M_adaptive_new, readMagnetization(folder_adaptive_new*root_adaptive_new*"_forward"))
end

root_adaptive_new = "adaptive_DeltaS=0.05_eps_tw_1e-12"
push!(M_adaptive_new, readMagnetization(folder_adaptive_new*root_adaptive_new*"_forward"))



Sx_N32_alpha3_5_NSV_256_adap = scatter(collect(0.1:0.1:10), M_adaptive_ref[1][1][1,1:100], label=L"ref: $\Delta S/S =0.1,\; \epsilon_{svd}=1^{-6}$",shape=:cross);
scatter!(collect(0.1:0.1:10), M_adaptive_ref[2][1][1,1:100], label=L"ref: $\Delta S/S =0.05,\; \epsilon_{svd}=1^{-6}$",shape=:cross, msc=6);
scatter!(collect(0.1:0.1:10), M_adaptive_new[1][1][1,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-6}$",shape=:xcross, msc=1);
#scatter!(collect(0.1:0.1:10), M_adaptive_new[2][1][1,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-7}$",shape=:xcross, msc=2);
#scatter!(collect(0.1:0.1:10), M_adaptive_new[3][1][1,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-8}$",shape=:xcross, msc=3);
scatter!(collect(0.1:0.1:10), M_adaptive_new[4][1][1,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-10}$",shape=:xcross, msc=4);
scatter!(collect(0.1:0.1:10), M_adaptive_new[5][1][1,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{tW}=1^{-12}$",shape=:xcross, msc=5);
scatter!(collect(0.1:0.1:10), M_adaptive_new[6][1][1,1:100], label=L"$\Delta S/S =0.05,\; \epsilon_{tW}=1^{-12}$",shape=:diamond, msc=6);
scatter!(collect(0.1:0.1:10), M_single_new[1][1,1:100], label="1-site",shape=:utriangle, msc=7, mc=:transparent);
scatter!(collect(0.1:0.1:10), M_single_NSV400[1][1,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=8, mc=:transparent);
scatter!(thickness_scaling=1.1)
ylims!(-0.02, 0.04)
xlims!(7,10)
xlabel!("t/J"); ylabel!(L"$\langle S_x^{(1)}\rangle$");
title!(L"$N=32, \; \alpha=3.5, \; D_{max}=256$")
savelatexfig(Sx_N32_alpha3_5_NSV_256_adap, plotsdir("benchmark/ALGS/N32_alpha3_5_conv_Sx1"), tex=true)


Sx_L32_alpha3_5_NSV256 = scatter(collect(0.1:0.1:10), S_new[1][1,1:100], label=L"$S^{1}_x$ new", shape=:cross, msc=1);
scatter!(collect(0.1:0.1:10), S_ref[1][1,1:100], label=L"$S^{1}_x$", shape=:xcross, msc=2);
scatter!(collect(0.1:0.1:10), S_new[1][16,1:100], label=L"$S^{16}_x$ new", shape=:cross, msc=3);
scatter!(collect(0.1:0.1:10), S_ref[1][16,1:100], label=L"$S^{16}_x$", shape=:xcross, msc=4);
scatter!(thickness_scaling=1.5);
xlabel!("t/J"); ylabel!(L"\langle S^{(i)}_x \rangle");
title!(L"$N=32, \; \alpha=3.0, \; D_{max}=256$ 1-site algorithm")
savelatexfig(Sx_L32_alpha3_0_NSV256, plotsdir("benchmark/UPDATE/N32_alpha3_0_NSV256_SS_Sx"), tex=true)

##endregion

###* α = 3.0 *###
##region


folder_obelix = "/mnt/obelix/TMI/L32/alpha_3.0/"
folder_lisa = "/mnt/lisa/L32/alpha_3.0/NSV_400/"
root_single = "single_site"
root_adaptive = "single_site"

#* Fidelities


#* Entropies
##region
S_single_NSV128 = readEntaglement("$(folder_obelix)NSV_128/$(root_single)_forward")
S_single_NSV256 = readEntaglement("$(folder_obelix)NSV_256/$(root_single)_forward")
S_single_NSV400 = readEntaglement("$(folder_lisa)$(root_single)_forward")

ΔS = [0.1, 0.05];
S_adaptive_alpha3_0 = [];
for n = 1:2
    root_adaptive = "adaptive_DeltaS=$(ΔS[n])"
    push!(S_adaptive_alpha3_0, readEntaglement(folder_obelix*root_adaptive*"_forward"))
end


S_N32_alpha3_0_conv = scatter(collect(0.1:0.1:10), S_adaptive_alpha3_0[1][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter!(collect(0.1:0.1:10), S_adaptive_alpha3_0[2][16,1:100], label=L"$\Delta S/S =0.05,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter!(collect(0.1:0.1:10), S_single_NSV128[16,1:100], label=L"1-site $D_{max}=128$",shape=:utriangle, msc=:4,mc=:transparent);
scatter!(collect(0.1:0.1:10), S_single_NSV256[16,1:100], label=L"1-site $D_{max}=256$",shape=:ltriangle, msc=:5,mc=:transparent);
scatter!(collect(0.1:0.1:10), S_single_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.2);
xlabel!("t/J"); ylabel!(L"S_{16}")
title!(L"$N=32, \; \alpha=3.0$")
savelatexfig(S_N32_alpha3_0_conv, plotsdir("benchmark/ALGS/N32_alpha3_0_conv_S"), tex=true)
##endregion

#* Magnetizations
##region
M_single_NSV128 = readMagnetization("$(folder_obelix)NSV_128/$(root_single)_forward")
M_single_NSV256 = readMagnetization("$(folder_obelix)NSV_256/$(root_single)_forward")
M_single_NSV400 = readMagnetization("$(folder_lisa)$(root_single)_forward")

ΔS = [0.1, 0.05];
M_adaptive_alpha3_0 = [];
for n = 1:2
    root_adaptive = "adaptive_DeltaS=$(ΔS[n])"
    push!(M_adaptive_alpha3_0, readMagnetization("$(folder_obelix)NSV_256/$(root_adaptive)_forward"))
end

Sx_N32_alpha3_0_conv = scatter(collect(0.1:0.1:10), M_adaptive_alpha3_0[1][1][1,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter!(collect(0.1:0.1:10), M_adaptive_alpha3_0[2][1][1,1:100], label=L"$\Delta S/S =0.05,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter!(collect(0.1:0.1:10), M_single_NSV128[1][1,1:100], label=L"1-site $D_{max}=128$",shape=:utriangle, msc=:4,mc=:transparent);
scatter!(collect(0.1:0.1:10), M_single_NSV256[1][1,1:100], label=L"1-site $D_{max}=256$",shape=:ltriangle, msc=:5,mc=:transparent);
scatter!(collect(0.1:0.1:10), M_single_NSV400[1][1,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.1)
ylims!(0.0, 0.05)
xlims!(7,10)
xlabel!("t/J"); ylabel!(L"$\langle S_x^{(11)}\rangle$");
title!(L"$N=32, \; \alpha=3.0$")
savelatexfig(Sx_N32_alpha3_0_conv, plotsdir("benchmark/ALGS/N32_alpha3_0_conv_Sx1_full"), tex=true)

##endregion

##endregion

###* α = 2.5 *###
##region

folder_obelix = "/mnt/obelix/TMI/L32/alpha_2.5/"
folder_lisa = "/mnt/lisa/L32/alpha_2.5/NSV_400/"
folder_iop = "/mnt/iop/Simulations/TMI/L32/alpha_2.5/"
root_single = "single_site"




#* Fidelities


#* Entropies
##region
S_single_alpha2_5_NSV128 = readEntaglement("$(folder_obelix)NSV_128/$(root_single)_forward")
S_single_alpha2_5_NSV256 = readEntaglement("$(folder_obelix)NSV_256/$(root_single)_forward")
S_single_alpha2_5_NSV400 = readEntaglement("$(folder_iop)NSV_400/$(root_single)_forward")
S_single_alpha2_5_NSV512 = readEntaglement("$(folder_iop)NSV_512/$(root_single)_forward")

ΔS = [0.1, 0.05];
S_adaptive_alpha2_5 = [];
for n = 1:2
    root_adaptive = "adaptive_DeltaS=$(ΔS[n])"
    push!(S_adaptive_alpha2_5, readEntaglement("$(folder_obelix)NSV_256/$(root_adaptive)_forward"))
end


S_N32_alpha2_5_conv = scatter(collect(0.1:0.1:10), S_adaptive_alpha2_5[1][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter!(collect(0.1:0.1:10), S_adaptive_alpha2_5[2][16,1:100], label=L"$\Delta S/S =0.05,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter(collect(0.1:0.1:10), S_single_alpha2_5_NSV128[16,1:100], label=L"1-site $D_{max}=128$",shape=:utriangle, msc=:4,mc=:transparent);
#scatter!(collect(0.1:0.1:10), S_single_alpha2_5_NSV256[16,1:100], label=L"1-site $D_{max}=256$",shape=:ltriangle, msc=:5,mc=:transparent);
scatter!(collect(0.1:0.1:10), S_single_alpha2_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(collect(0.1:0.1:10), S_single_alpha2_5_NSV512[16,1:100], label=L"1-site $D_{max}=512$",shape=:ltriangle, msc=:8,mc=:transparent);
scatter!(thickness_scaling=1.2);
xlabel!("tJ"); ylabel!(L"S_{16}")
title!(L"$N=32, \; \alpha=2.5$")
savelatexfig(S_N32_alpha2_5_conv, plotsdir("benchmark/ALGS/N32_alpha2_5_conv_S"), tex=true)
##endregion

#* Magnetizations
##region
M_single_alpha2_5_NSV128 = readMagnetization("$(folder_obelix)NSV_128/$(root_single)_forward")
#M_single_NSV256 = readMagnetization("$(folder_obelix)NSV_256/$(root_single)_forward")
M_single_alpha2_5_NSV400 = readMagnetization("$(folder_iop)NSV_400/$(root_single)_forward")
M_single_alpha2_5_NSV512 = readMagnetization("$(folder_iop)NSV_512/$(root_single)_forward")

ΔS = [0.1, 0.05];
M_adaptive_alpha2_5 = [];
for n = 1:2
    root_adaptive = "adaptive_DeltaS=$(ΔS[n])"
    push!(M_adaptive_alpha2_5, readMagnetization("$(folder_obelix)NSV_256/$(root_adaptive)_forward"))
end

Sx_N32_alpha2_5_conv = scatter(collect(0.1:0.1:10), M_adaptive_alpha2_5[1][1][16,1:100], label=L"$\Delta S/S =0.1,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter!(collect(0.1:0.1:10), M_adaptive_alpha2_5[2][1][16,1:100], label=L"$\Delta S/S =0.05,\; \epsilon_{svd}=10^{-6}$",shape=:cross, msc=:auto);
scatter!(collect(0.1:0.1:10), M_single_alpha2_5_NSV128[1][16,1:100], label=L"1-site $D_{max}=128$",shape=:utriangle, msc=:4,mc=:transparent);
#scatter!(collect(0.1:0.1:10), M_single_NSV256[1][16,1:100], label=L"1-site $D_{max}=256$",shape=:ltriangle, msc=:5,mc=:transparent);
scatter!(collect(0.1:0.1:10), M_single_alpha2_5_NSV400[1][16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.1);
ylims!(-0.01, 0.01)
xlims!(7,10)
xlabel!("t/J"); ylabel!(L"$\langle S_x^{(16)}\rangle$");
title!(L"$N=32, \; \alpha=2.5$")
savelatexfig(Sx_N32_alpha2_5_conv, plotsdir("benchmark/ALGS/N32_alpha2_5_conv_Sx16"), tex=true)

##endregion

##endregion

###* α=1.5 *###
root_folder = "/mnt/obelix/TMI/L32/alpha_1.5/"
root_file_single = "single_site_forward"
root_file_two = "single_site_forward"

#* Entropies
S_single_alpha1_5_NSV512 = readEntaglement("$(root_folder)NSV_512/$(root_file_single)")
S_single_alpha1_5_NSV600 = readEntaglement("$(root_folder)NSV_600/$(root_file_single)")
S_single_alpha1_5_NSV700 = readEntaglement("$(root_folder)NSV_700/$(root_file_single)")


#* Magnetizations
M_single_alpha1_5_NSV512 = readMagnetization("$(root_folder)NSV_512/$(root_file_single)")
M_single_alpha1_5_NSV600 = readMagnetization("$(root_folder)NSV_600/$(root_file_single)")
M_single_alpha1_5_NSV700 = readMagnetization("$(root_folder)NSV_700/$(root_file_single)")


#* Wall time
file_data =  readlines("$(root_folder)NSV_512/two_site_eps_tw_5e-13_log.txt")
length(file_data)

time_TDVP_low = file_data[vcat(collect(230:-6:170), collect(165:-5:65))]
function time_step(line)
    si =  findfirst("\t: ",line)[end]
    return parse(Float64,line[si+1:si+4])
end

function bondsize_step(line)
    si =  findfirst("Ψt:",line)[end]
    return parse(Float64,line[si+1:si+4])
end

cum_time(ts)=[sum(reverse(ts)[1:n]) for n ∈ 1:length(ts)]

bs_two_site_eps_tw_low_eps_NSV_512 = bondsize_step.(time_TDVP[1:end-1])
ts_two_site_eps_tw_low_eps_NSV_512 = time_step.(time_TDVP)

bs_two_site_eps_tw_high_eps_NSV_512 = bondsize_step.(time_TDVP[1:end-1])
ts_two_site_eps_tw_high_eps_NSV_512 = time_step.(time_TDVP)

append!(bs_two_site_eps_tw_high_eps_NSV_512,4)



#* Plotting

Sy_data = [S_single_alpha1_5_NSV512[15,1:100],S_single_alpha1_5_NSV600[15,1:100],S_single_alpha1_5_NSV700[15,1:100]]
y_color = [4 6 8]
y_fill = fill(:transparent,(1,3))
y_shape = [:dtriangle :utriangle :ltriangle]
y_label = [L"1-site $D_{max}=512$" L"1-site $D_{max}=600$" L"1-site $D_{max}=700$"]

plot_S_alpha1_5 = scatter(collect(0.1:0.1:10), Sy_data, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape)
#scatter!(collect(0.1:0.1:10), S_single_alpha1_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.5, legend=:topleft)
xlabel!("tJ"); ylabel!(L"S_{16}")
#title!(L"$N=32, \; \alpha=1.5$")
xlims!((8,10));ylims!(3.5,4.5)
savelatexfig(plot_S_alpha1_5, plotsdir("LMU/N32_alpha1_5_conv_S_inset"), tex=true)


My_data = [M_single_alpha1_5_NSV512[3][1,1:20],M_single_alpha1_5_NSV600[3][1,1:20],M_single_alpha1_5_NSV700[3][1,1:20]]

plot_Mx_alpha1_5 = scatter(collect(0.5:0.5:10), My_data, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape)
#scatter!(collect(0.1:0.1:10), S_single_alpha2_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.5, legend=:topright);
xlabel!("tJ"); ylabel!(L"\langle S^{(1)}_{x}\rangle")
#title!(L"$N=32, \; \alpha=1.5$")
savelatexfig(plot_Mx_alpha1_5, plotsdir("LMU/N32_alpha1_5_Sx1"), tex=true)

y_data = [cum_time(ts_two_site_eps_tw_high_eps_NSV_512) prepend!(zeros(20),cum_time(ts_two_site_eps_tw_low_eps_NSV_512)) cum_time(ts_single_NSV_512[end-31:end]) cum_time(ts_single_NSV_700[end-31:end])];

cum_time(ts_single_NSV_700)/60

ctg = repeat([L"2-site, low $\epsilon$",L"2-site, high $\epsilon$", L"1-site $D_{max}=512$",L"1-site $D_{max}=700$"], inner = 32)
t_acc_alpha1_5 = groupedbar(y_data, bar_position = :dodge, bar_width=1.0, lc=:transparent, legend=:topleft, group=ctg,thickness_scaling=1.2)
xlabel!("tJ"); ylabel!(L"$t_{\textrm{acc}}$ (min)")
xticks!([0,10,20,30],["0", "1", "2", "3"])
savelatexfig(t_acc_alpha1_5, plotsdir("LMU/t_acc_alpha1_5"), tex=true)

ctg = repeat([L"2-site, low $\epsilon$",L"2-site, high $\epsilon$"], inner = 32)
y_data_D = [prepend!(zeros(20), reverse(bs_two_site_eps_tw_low_eps_NSV_512)) reverse(bs_two_site_eps_tw_high_eps_NSV_512)]
D_alpha_1_5 = groupedbar(y_data_D, bar_position = :dodge, bar_width=1.0, lc=:transparent, legend=:topleft, group=ctg, thickness_scaling=1.2)
xlabel!(L"$tJ$"); ylabel!(L"$D$")
xticks!([0,10,20,30],["0", "1", "2", "3"])
savelatexfig(D_alpha_1_5, plotsdir("LMU/D_alpha1_5"), tex=true)


###* α=1.1
root_folder = "/mnt/obelix/TMI/L32/alpha_1.1/"
root_file_single = "single_site_forward"

#* Entropies
S_single_alpha1_5_NSV512 = readEntaglement("$(root_folder)NSV_512/$(root_file_single)")
S_single_alpha1_5_NSV600 = readEntaglement("$(root_folder)NSV_600/$(root_file_single)")
S_single_alpha1_5_NSV700 = readEntaglement("$(root_folder)NSV_700/$(root_file_single)")


#* Magnetizations
M_single_alpha1_5_NSV512 = readMagnetization("$(root_folder)NSV_512/$(root_file_single)")
M_single_alpha1_5_NSV600 = readMagnetization("$(root_folder)NSV_600/$(root_file_single)")
M_single_alpha1_5_NSV700 = readMagnetization("$(root_folder)NSV_700/$(root_file_single)")



#* Plotting

Sy_data = [S_single_alpha1_5_NSV512[16,1:100],S_single_alpha1_5_NSV600[16,1:100],S_single_alpha1_5_NSV700[16,1:100]]
y_color = [4 6 8]
y_fill = fill(:transparent,(1,3))
y_shape = [:dtriangle :utriangle :ltriangle]
y_label = [L"1-site $D_{max}=512$" L"1-site $D_{max}=600$" L"1-site $D_{max}=700$"]

plot_S_alpha1_5 = scatter(collect(0.1:0.1:10), Sy_data, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape)
#scatter!(collect(0.1:0.1:10), S_single_alpha1_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=2.5, legend=:topleft)
xlabel!("tJ"); ylabel!(L"S_{16}");
#title!(L"$N=32, \; \alpha=1.5$")
xlims!((8,10));ylims!(3.5,4.5)
savelatexfig(plot_S_alpha1_5, plotsdir("LMU/N32_alpha1_5_conv_S_inset"), tex=true)


My_data = [M_single_alpha1_5_NSV512[1][1,1:20],M_single_alpha1_5_NSV600[1][1,1:20],M_single_alpha1_5_NSV700[1][1,1:20]]

plot_Mx_alpha1_5 = scatter(collect(0.5:0.5:10), My_data, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape)
#scatter!(collect(0.1:0.1:10), S_single_alpha2_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.5, legend=:topright);
xlabel!("tJ"); ylabel!(L"\langle S^{(1)}_{x}\rangle")
#title!(L"$N=32, \; \alpha=1.5$")
savelatexfig(plot_Mx_alpha1_5, plotsdir("LMU/N32_alpha1_5_Sx1"), tex=true)

y_data = [cum_time(ts_two_site_eps_tw_high_eps_NSV_512) prepend!(zeros(20),cum_time(ts_two_site_eps_tw_low_eps_NSV_512)) cum_time(ts_single_NSV_512[end-31:end]) cum_time(ts_single_NSV_700[end-31:end])];

cum_time(ts_single_NSV_700)/60

ctg = repeat([L"2-site, low $\epsilon$",L"2-site, high $\epsilon$", L"1-site $D_{max}=512$",L"1-site $D_{max}=700$"], inner = 32)
t_acc_alpha1_5 = groupedbar(y_data, bar_position = :dodge, bar_width=1.0, lc=:transparent, legend=:topleft, group=ctg,thickness_scaling=1.2)
xlabel!("tJ"); ylabel!(L"$t_{\textrm{acc}}$ (min)")
xticks!([0,10,20,30],["0", "1", "2", "3"])
savelatexfig(t_acc_alpha1_5, plotsdir("LMU/t_acc_alpha1_5"), tex=true)

ctg = repeat([L"2-site, low $\epsilon$",L"2-site, high $\epsilon$"], inner = 32)
y_data_D = [prepend!(zeros(20), reverse(bs_two_site_eps_tw_low_eps_NSV_512)) reverse(bs_two_site_eps_tw_high_eps_NSV_512)]
D_alpha_1_5 = groupedbar(y_data_D, bar_position = :dodge, bar_width=1.0, lc=:transparent, legend=:topleft, group=ctg, thickness_scaling=1.2)
xlabel!(L"$tJ$"); ylabel!(L"$D$")
xticks!([0,10,20,30],["0", "1", "2", "3"])
savelatexfig(D_alpha_1_5, plotsdir("LMU/D_alpha1_5"), tex=true)


#********* Plotting

using Plots
pgfplotsx();
using LaTeXStrings

#******* Forward evolution
#*** Fidelities
single_site = scatter(collect(0.05:0.25:8), -fid_single_MPS.+1, label=L"1-site exact", yscale=:log10)


adaptive_foward_plot = scatter(collect(0.05:0.25:8), -fid_double.+1, label=L"2-site, $D_{\textrm{max}}=256$", yscale=:log10)
scatter!(collect(0.05:0.25:8), -fid_adaptive[1,:].+1, label=L"2-site adaptive, $\Delta S/S=0.05$", yscale=:log10)
scatter!(collect(0.05:0.25:8), -fid_adaptive[2,:].+1, label=L"2-site adaptive, $\Delta S/S=0.1$", yscale=:log10)
scatter!(collect(0.05:0.25:8), -fid_adaptive[3,:].+1, label=L"2-site adaptive, $\Delta S/S=0.3$", yscale=:log10)


#*** Entropy
S_ref_forward = S_single_forward[7,2:160];
ΔSrel(S) = (S-S_ref_forward)./S_ref_forward

#S_forward = scatter(collect(0.10:0.05:8), S_single_forward[7,2:160], label=L"1-site, $D_{\textrm{max}}=256$")
S_forward = scatter(collect(0.10:0.05:8), abs.(ΔSrel(S_double_forward[7,2:160])), label=L"2-site, $D_{\textrm{max}}=256$", yscale=:log10, shape=:cross, msc=1)
scatter!(collect(0.10:0.05:8), abs.(ΔSrel(S_adaptive_forward[1][7,2:160])), label=L"2-site adaptive, $\Delta S/S=0.05$", yscale=:log10, shape=:xcross, msc=2)
scatter!(collect(0.10:0.05:8), abs.(ΔSrel(S_adaptive_forward[2][7,2:160])), label=L"2-site adaptive, $\Delta S/S=0.1$", yscale=:log10, shape=:diamond, msc=3)
#scatter!(collect(0.10:0.05:8), ΔSrel(S_adaptive_forward[3][7,2:160]), label=L"2-site adaptive, $\Delta S/S=0.3$")
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"$\Delta S/ S_{\textrm{ss}}$")
title!(L"$N=16, \; \alpha=0.5$ Ref: 1-site")
savelatexfig(S_forward, plotsdir("benchmark/ALGS/N16_Benchmark_adaptive_forward_DeltaS_vN_log"), tex=true)

S_forward_single = scatter(collect(0.10:0.05:8), S_single_forward[7,2:160], label=L"$S^7$", shape=:diamond, msc=1)
scatter!(collect(0.10:0.05:8), S_single_forward[1,2:160], label=L"$S^1$", shape=:cross,msc=3)
scatter!(collect(0.10:0.05:8), S_single_forward[15,2:160], label=L"$S^{15}$", shape=:xcross, msc=2)
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"$S_{\textrm{vN}}$")
title!(L"$N=16, \; \alpha=0.5$")
savelatexfig(S_forward_single, plotsdir("benchmark/ALGS/N16_Benchmark_single_forward_S_vN"), tex=true)


#*** AFM
S_AFM = scatter(collect(0.05:0.05:8), S_single_forward_AFM[7,1:160], label=L"1-site, $D_{\textrm{max}}=256$, AFM", yscale=:log10, shape=:cross, msc=1)
scatter!(collect(0.05:0.05:8), S_single_forward[7,1:160], label=L"1-site, $D_{\textrm{max}}=256$", yscale=:log10, shape=:xcross, msc=2)


S_AFM = scatter(collect(0.05:0.05:8), S_single_new[7,1:160], label=L"1-site, $D_{\textrm{max}}=256$, AFM", shape=:cross, msc=1)
scatter!(collect(0.05:0.05:8), S_single_ref[7,1:160], label=L"1-site, $D_{\textrm{max}}=256$", shape=:xcross, msc=2)


#****** Backward
#*** Fidelities
backward_OTOC_A =  scatter(collect(8.05:0.25:16), -fid_double_OTOC_A.+1, label=L"2-site, $D_{\textrm{max}}=256$", yscale=:log10)
scatter!(collect(8.05:0.25:16), -fid_adaptive_OTOC_A[1,:].+1, label=L"2-site adaptive, $\Delta S/S=0.05$", yscale=:log10)
scatter!(collect(8.05:0.25:16), -fid_adaptive_OTOC_A[2,:].+1, label=L"2-site adaptive, $\Delta S/S=0.1$", yscale=:log10)
scatter!(collect(8.05:0.25:16), -fid_adaptive_OTOC_A[3,:].+1, label=L"2-site adaptive, $\Delta S/S=0.3$", yscale=:log10)

scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"$1-\langle \Psi_{old}|\Psi_{new}\rangle$")
title!(L"$L=32, \; NSV=128$")

savelatexfig(backward_OTOC_A, plotsdir("benchmark/ALGS/N16_Benchmark_adaptive_OTOC_A"), tex=true)


#*** Entropy
S_ref_reverse = S_single_reverse[7,1:160];
ΔSrel_rev(S) = (S-S_ref_reverse)./S_ref_reverse

S_reverse = scatter(collect(0.10:0.05:8), S_single_forward[7,2:160], label=L"1-site, $D_{\textrm{max}}=256$")
#S_reverse = scatter(collect(0.05:0.05:8), ΔSrel_rev(S_double_forward[7,2:160]), label=L"2-site, $D_{\textrm{max}}=256$")
S_reverse = scatter(collect(0.05:0.05:8), ΔSrel_rev(S_adaptive_reverse[1][7,1:160]), label=L"2-site adaptive, $\Delta S/S=0.05$")
scatter!(collect(0.05:0.05:8), ΔSrel_rev(S_adaptive_reverse[2][7,1:160]), label=L"2-site adaptive, $\Delta S/S=0.1$")
#scatter!(collect(0.05:0.05:8), ΔSrel_rev(S_adaptive_reverse[3][7,1:160]), label=L"2-site adaptive, $\Delta S/S=0.3$")
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"$\Delta S/ S_{\textrm{ss}}$")
title!(L"$N=16, \; \alpha=0.5$")
savelatexfig(S_forward, plotsdir("benchmark/ALGS/N16_Benchmark_adaptive_reverse_DeltaS_vN"), tex=true)

S_reverse_single = scatter(collect(0.05:0.05:8), S_single_reverse[7,1:160], label=L"$S^7$", shape=:diamond, msc=1)
scatter!(collect(0.05:0.05:8), S_single_reverse[1,1:160], label=L"$S^1$", shape=:cross,msc=3)
scatter!(collect(0.05:0.05:8), S_single_reverse[15,1:160], label=L"$S^{15}$", shape=:xcross, msc=2)
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"$S_{\textrm{vN}}$")
title!(L"$N=16, \; \alpha=0.5$")
savelatexfig(S_forward, plotsdir("benchmark/ALGS/N16_Benchmark_single_reverse_S_vN"), tex=true)


S_reverse_adaptive = scatter(collect(0.05:0.05:8), S_adaptive_reverse[1][7,1:160], label=L"$S^7$", shape=:diamond, msc=1)
scatter!(collect(0.05:0.05:8), S_adaptive_reverse[1][1,1:160], label=L"$S^1$", shape=:cross,msc=3)
scatter!(collect(0.05:0.05:8), S_adaptive_reverse[1][15,1:160], label=L"$S^{15}$", shape=:xcross, msc=2)
scatter!(thickness_scaling=1.5)
xlabel!("t/J"); ylabel!(L"$S_{\textrm{vN}}$")
title!(L"$N=16, \; \alpha=0.5$")
savelatexfig(S_forward, plotsdir("benchmark/ALGS/N16_Benchmark_adaptive_0.1_reverse_S_vN"), tex=true)

#*** Bond dimension
D_forward = scatter(collect(0.05:0.10:8), D_double_forward[1:2:160],label=L"2-site, $D_{\textrm{max}}=256$")
scatter!(collect(0.05:0.10:8), D_adaptive_forward[1][1:2:160],label=L"2-site adaptive, $\Delta S/S=0.05$")
scatter!(collect(0.05:0.10:8), D_adaptive_forward[2][1:2:160],label=L"2-site adaptive, $\Delta S/S=0.1$")
scatter!(collect(0.05:0.10:8), D_adaptive_forward[3][1:2:160],label=L"2-site adaptive, $\Delta S/S=0.3$")
scatter!(thickness_scaling=1.2)
xlabel!("t/J"); ylabel!(L"$D_{\textrm{max}}$")
title!(L"$N=16, \; \alpha=0.5$")
savelatexfig(D_forward, plotsdir("benchmark/ALGS/N16_Benchmark_adaptive_forward_D_max"), tex=true)


D_reverse =  scatter(collect(8.10:0.10:16), D_adaptive_reverse[1][1:2:160],label=L"2-site adaptive, $\Delta S/S=0.05$")
scatter!(collect(8.10:0.10:16), D_adaptive_reverse[2][1:2:160],label=L"2-site adaptive, $\Delta S/S=0.1$")
scatter!(collect(8.10:0.10:16), D_adaptive_reverse[3][1:2:160],label=L"2-site adaptive, $\Delta S/S=0.3$")
scatter!(thickness_scaling=1.2)
xlabel!("t/J"); ylabel!(L"$D_{\textrm{max}}$")
title!(L"$N=16, \; \alpha=0.5$")
savelatexfig(D_reverse, plotsdir("benchmark/ALGS/N16_Benchmark_adaptive_reverse_D_max"), tex=true)