using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());

using Expokit, BSON, HDF5, SparseArrays, Arpack
using KrylovKit
include(srcdir()*"/hamiltonians.jl");
using operators_basis


params_tdvp = h5open(datadir("L=16_t=16_Bx=-1.05_Bz=0.5_NSV=256_eps_svd=1e-6_observables.h5"))
Jij = read(params_tdvp["Parameters/Jij"])


Nsites = 16;
Bx = [2.1];
Bz = -1;
Δt = 0.05;

TF = TransverseFieldIsing(Jij,Bx);
TF_M = convert(SparseMatrixCSC, TF);
Hlong = Bz*spdiagm(sum([diag(Sᶻᵢ(i,Nsites)) for i in 1:Nsites]));
H_TMI = TF_M + Hlong;

H_dense = zeros(2^16,2^16);

for j ∈ 0:2^8-1
    H_dense[2^8*j+1:2^8*(j+1),:] = collect(TF_M[2^8*j+1:2^8*(j+1),:])    
end

collect(TF_M[2^10+1:2^10*2,:])


b, λ = eigs(H_TMI, nev=2^16)
b, λ = eigen()




ψts = zeros(ComplexF64, 2^16,10,16);
for i ∈ 1:16
    ψi = σʸᵢ(i,16)*Ψ_for_v2
    ψts[:,:,i] = shortReverseQuench(ψi, Jij, Bx, Bz);
end


for i ∈ 1:16
    println(abs(ψts[:,5,i] ⋅ Ψ_B))
end





function shortQuench(Jij, Bx, Bz)
    
    TF = TransverseFieldIsing(Jij,[Bx]);
    TF_M = convert(SparseMatrixCSC, TF);
    Hlong = Bz*spdiagm(sum([diag(Sᶻᵢ(i,Nsites)) for i in 1:Nsites]));
    H_TMI = TF_M + Hlong;

    ψts = spzeros(ComplexF64, 2^16,10);
    ψt = ψ_y_pol;
    for s=1:10
        ψt = expmv(-im*Δt, H_TMI, Vector(ψt); tol=1e-10);
        ψts[:,s] = ψt
    end
    return ψts
end

function shortReverseQuench(Ψin, Jij, Bx, Bz, steps)
    function Zᵢ(i,N)
        Z = sparse([1 0; 0 -1]);
        𝟙 = sparse([1 0; 0 1]);
        return kron([n==i ? Z : 𝟙 for n in N:-1:1]...)
    end
    
    TF = TransverseFieldIsing(Jij,[Bx]);
    TF_M = convert(SparseMatrixCSC, TF);
    Hlong = Bz*spdiagm(sum([diag(0.5*Zᵢ(i,Nsites)) for i in 1:Nsites]));
    H_TMI = TF_M + Hlong;

    ψts = spzeros(ComplexF64, 2^16,steps);
    ψt = Ψin;
    for s=1:steps
        ψt = expmv(im*Δt, H_TMI, Vector(ψt); tol=1e-10);
        ψts[:,s] = ψt
    end
    return ψts
end


#* Compariing with exact results

data_exact = h5open("data/"*"tek_full_mfi_n16_a1.2_hx-1.05_hz0.5.hdf5", "r")
mps_data_step5 = h5open(folder*"/L=16_t=16_pl_exp=1.2_Bx=1.05_Bz=-0.5_NSV=256_eps_svd=0.0_tol_Lanczos=1e-10_state_step=5.h5", "r");
ψ_mps_step5 =  reconstructcomplexState(read(mps_data_step5["mps"]));

ψt_exact = read(data_exact["psi"]);

abs(ψts[:,1] ⋅ ψt_exact[:,2])
abs(Ψ_tdvp_reverse[1,:] ⋅ ψt_exact[:,2])

#* ----------------------------------- Debugging the initial steps of the time-evolution


folder_debug = "/mnt/obelix/TMI/debug"
folder_debug = "/mnt/c/Users/Juan/surfdrive/QuantumSimulationPhD/Code/TDVP/_research/data/debug"

pl_exp = 0.5;
Bx = 2.1;
Bz = -1.0;

close(mps_data_step1)
mps_data_step1 = h5open(folder_debug*"/debug_pl_exp_$(pl_exp)_Bx=$(Bx)_Bz=$(Bz)_state_step=1.h5", "r");
ψ_mps_step1 =  reconstructcomplexState(read(mps_data_step1["mps"]));

params_tdvp = load(folder_debug*"/debug_pl_exp_$(pl_exp)_Bx=$(Bx)_Bz=$(Bz)parameters.bson");
Jij = hcat(params_tdvp[:coupling_matrix]...);

ψts = shortQuench(Jij, Bx, Bz);

abs(ψts[:,5] ⋅ Ψ_for)



debug_tdvp["Fidelities/Bz=$(Bz)_Bx=$(Bx)"] = abs(ψts[:,1] ⋅ ψ_mps_step1)


close(debug_tdvp)
debug_tdvp = h5open("$(folder_debug)/PL_alpha_$(pl_exp)_debug.h5", "w");
create_group(debug_tdvp, "Fidelities")