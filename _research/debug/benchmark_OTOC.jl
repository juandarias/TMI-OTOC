using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5, LinearAlgebra, BSON, SparseArrays
include(srcdir("/MPS_conversions.jl"));
include(srcdir("/ncon.jl"));
include(plotsdir("plotting_functions.jl"));



function loadstate(folder, file_name, step)
    data = h5open(folder*file_name*"_state_step=$(step).h5", "r");
    state = reconstructcomplexState(read(data["mps"]));
    return state
end


fid(phi,psi) = abs(phi ⋅ psi)

function Yᵢ(i,N)
    Y = sparse([0 -im; im 0]);
    𝟙 = sparse([1 0; 0 1]);
    return kron([n==i ? Y : 𝟙 for n in 1:N]...)
end

#* Check OTOC results

# Exact results
file_name = datadir("tek_mfi_n16_a1.2_hx-1.05_hz0.5.hdf5")
OTOC = h5open(file_name, "r"); # opens state at step i
phi = read(OTOC["phi"]);
phi_prime = read(OTOC["phi_prime"]);
psi = read(OTOC["psi"]);

# Forward quench
folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/"
file_root = "benchmark_pl_exp_1.2_Bx=-2.1_Bz=1.0_eps_svd=1e-6_rep_2"


Ψ_ref_s320 = loadstate(folder_obelix, file_root, 320);
Ψ_ref_s160 = loadstate(folder_obelix, file_root, 160);


##### * Run 518039

#* Checking forward quench

folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "loop_test_t=16_rep2_forward"

Ψ_OTOC_t8_s160 = loadstate(folder_obelix, file_root, 160);

fid(Ψ_ref_s160, Ψ_OTOC_t8_s160) #! Forward quench is correct


#* Check OxV action for W_A
folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "loop_test_t=16_rep2_reverse_A_step_160"

Ψ_OTOC_t8_OTOC_A_s0 = loadstate(folder_obelix, file_root, 0);
OTOC_A_s0 = Yᵢ(16,16)*Ψ_OTOC_t8_s160

fid(OTOC_A_s0, Ψ_OTOC_t8_OTOC_A_s0) #! Application of W_A is correct

#* Check final state of reverse evolution for W_A
folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "loop_test_t=16_rep2_reverse_A_step_160"

Ψ_OTOC_t8_OTOC_A_s160 = loadstate(folder_obelix, file_root, 160);

read(OTOC["time"])[161]
fid(phi_prime[:,161], Ψ_OTOC_t8_OTOC_A_s160) #! States after reverse evolution match!!


#* Check OxV action for W_B
folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "loop_test_t=16_rep2_reverse_B_step_160"

Ψ_OTOC_t8_OTOC_B_s0 = loadstate(folder_obelix, file_root, 0);
OTOC_B_s0 = Yᵢ(9,16)*Ψ_OTOC_t8_s160

fid(OTOC_B_s0, Ψ_OTOC_t8_OTOC_B_s0) #! Application of W_B is correct

#* Check final state of reverse evolution for W_B
folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "loop_test_t=16_rep2_reverse_B_step_160"

Ψ_OTOC_t8_OTOC_B_s160 = loadstate(folder_obelix, file_root, 160);

read(OTOC["time"])[161]
fid(phi[:,161], Ψ_OTOC_t8_OTOC_B_s160) #! States after reverse evolution match!!



#* Check OxV action

function Zᵢ(i,N)
    Z = sparse([1 0; 0 -1]);
    𝟙 = sparse([1 0; 0 1]);
    return kron([n==i ? Z : 𝟙 for n in N:-1:1]...)
end

function Xᵢ(i,N)
    X = sparse([0 1; 1 0]);
    𝟙 = sparse([1 0; 0 1]);
    return kron([n==i ? X : 𝟙 for n in N:-1:1]...)
end


function Mi(state)
    L = log2(length(state))
    magZ = [real(state'*(Zᵢ(i,L)*state)) for i ∈ 1:L]
    magY = [real(state'*(Yᵢ(i,L)*state)) for i ∈ 1:L]
    magX = [real(state'*(Xᵢ(i,L)*state)) for i ∈ 1:L]
    return magZ, magY, magX
end



folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"

file_root = "pl_exp_1.2_OTOC_diagnose_W=16_t=1.0_reverse_A"
Ψ_OTOC_t1_s1_Y0 = loadstate(folder_obelix, file_root, 1);
Ψ_OTOC_t1_s20_Y0 = loadstate(folder_obelix, file_root, 20);

file_root = "pl_exp_1.2_OTOC_diagnose_W=16_t=1.0_reverse_B"
Ψ_OTOC_t1_s1_Y15 = loadstate(folder_obelix, file_root, 1);


phi_0_Y0 =  Yᵢ(1,16)*Ψ_OTOC_t1_s20
phi_0_Y15 = Yᵢ(16,16)*Ψ_OTOC_t1_s20 #! this is correct, the action of the operator is the expected one

phi_0_Z0 =  Zᵢ(1,16)*Ψ_OTOC_t1_s20
phi_0_Z15 = Zᵢ(16,16)*Ψ_OTOC_t1_s20 #! this is correct, the action of the operator is the expected one

#? Fidelities
fid(Ψ_OTOC_t1_s1_Y0 , Ψ_OTOC_t1_s20) #!but the states are different
fid(Ψ_OTOC_t1_s1_Y0 , phi_0_Y15) #* same states
fid(Ψ_OTOC_t1_s1_Y15 , phi_0_Y0) #* same states
fid(Ψ_OTOC_t1_s1_Y15 , Ψ_OTOC_t1_s20) #!but the states are different


#? Magnetizations
mZ_t1_s20, mY_t1_s20 , mX_t1_s20 =Mi(Ψ_OTOC_t1_s20);
mZ_t1_s0_Y15, mY_t1_s0_Y15 , mX_t1_s0_Y15 =Mi(Ψ_OTOC_t1_s0_Y15);
mZ_t1_s0_Y0, mY_t1_s0_Y0 , mX_t1_s0_Y0 =Mi(Ψ_OTOC_t1_s0_Y0);

mZ_phi_0_Z15, mY_phi_0_Z15 , mX_phi_0_Z15 =Mi(phi_0_Z15);
mZ_phi_0_Z0, mY_phi_0_Z0 , mX_phi_0_Z0 =Mi(phi_0_Z0);

mZ_t1_s0_Y15 - mZ_t1_s20 #! Mz is the same
mY_t1_s0_Y15 - mY_t1_s20 #! 
mX_t1_s0_Y15 - mX_t1_s20 #! 


mZ_t1_s0_Y0 - mZ_t1_s20 #! Mz is the same

mY_t1_s0_Y0 - mY_t1_s20 #! <Y16> is different
mY_t1_s0_Y0[16] , mY_t1_s20[16] #! <Y16> is different

mX_t1_s0_Y0 - mX_t1_s20 #! <X16> is different
mX_t1_s0_Y0[16],  mX_t1_s20[16] #! <X16> is different



#* Check final state OTOC against exact results

folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"

file_root = "pl_exp_1.2_OTOC_diagnose_W=16_t=1.0_reverse_A"
Ψ_OTOC_t1_s20_Y0 = loadstate(folder_obelix, file_root, 20);

file_root = "pl_exp_1.2_OTOC_diagnose_W=16_t=1.0_reverse_B"
Ψ_OTOC_t1_s20_Y15 = loadstate(folder_obelix, file_root, 20);

#? Calculating exact values
ψi = Yᵢ(16,16)*Ψ_ref_s20
phi_exact_Y0 = shortReverseQuench(ψi, Jij, Bx, Bz,20);
phi_exact_Y15 = shortReverseQuench(ψi, Jij, Bx, Bz,20);

fid(phi_exact_Y15[:,20], Ψ_OTOC_t1_s20_Y0)
fid(phi_exact_Y0[:,20], Ψ_OTOC_t1_s20_Y15)

fid(phi_prime[:,21],Ψ_OTOC_t1_s20_Y15) #! same states, thus labelling of sites is not inverted





# OTOC t=1
folder_obelix = "/mnt/obelix/TMI/PL_alpha_1.2/OTOC/"
file_root = "L=16_pl_exp_1.2_Bx=_Bz=_eps_svd=1e-6_OTOC_diagnose_W=16_t=1.0_"
TDVP_for_v2 = h5open(folder_obelix*file_root*"320.h5", "r");
mps_in = read(TDVP_for_v2["mps"]);
Ψ_for_v2 = reconstructcomplexState(mps_in);

abs(Ψ_for ⋅ Ψ_for_v2)

