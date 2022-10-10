using DrWatson
@quickactivate
using HDF5, LinearAlgebra, OrderedCollections, SparseArrays


psi1 = h5open("data/"*"tek_full_mfi_n16_a0.5_hx-1.05_hz0.5.hdf5", "r")
psi_xpol = h5open("data/"*"x_pol_state_OTOC.h5", "r")
psi_rand = h5open("data/"*"MpsBackup_fullMmax=20.h5", "r")
mps_rand = read(psi_rand, "mps")


function reconstructState(mps)

    #* basis of Hilbert space
    basis(n)= Vector{Bool}(digits(n, base=2, pad=N))
    basis(n,D)= Vector{Bool}(digits(n, base=2, pad=D))

    #* Sort A matrices by lattice site. Apparently the sites are labelled from right to left
    sites = Int(length(mps)/2);
    mps = sort(mps_in);
    keys_mps = keys(mps);
    Azero = []; Aone = [];;
    keys_zero = [];

    i=1
    for key in keys_mps
        mod(i,2) == 0 && push!(Aone, mps[key])
        mod(i,2) == 1 && push!(Azero, mps[key])
        i+=1
    end
    Aone = reverse(Aone)
    Azero = reverse(Azero)
    
    #* Rebuild state
    state = zeros(2^sites);
    for n ∈ 0:2^sites-1    
        tensors = [];
        basis_n=basis(n, sites);
        for σ ∈ 1:sites
            basis_n[σ] == 0 ? push!(tensors, Azero[σ]) : push!(tensors, Aone[σ]);
        end
        state[n+1] = prod(tensors)[1] #Calculates the coefficient of the basis state n
    end
    return state
end


function reconstructcomplexState(mps; transpose=false) #* if the MPS have been transposed during saving to HDF5 correct by setting tranpose = true

    #* basis of Hilbert space
    basis(n)= Vector{Bool}(digits(n, base=2, pad=N))
    basis(n,D)= Vector{Bool}(digits(n, base=2, pad=D))

    #* Sort A matrices by lattice site. Apparently the sites are labelled from right to left
    sites = length(mps) ÷ 4;
    keys_mps = collect(keys(mps));
    index_key(A_key) = findfirst(x-> x==A_key, keys_mps)

    Azero = [];     
    for site ∈ 0:sites-1
        Akey = string(site)*"_0_((),())"
        AiRe = index_key(Akey*"Re");
        AiIm = index_key(Akey*"Im");
        Acomplex = im*mps[keys_mps[AiIm]] + mps[keys_mps[AiRe]]; #Re[A] + i Im[A]
        transpose == false ? push!(Azero, Acomplex) : push!(Azero, Acomplex');
    end

    Aone = [];     
    for site ∈ 0:sites-1
        Akey = string(site)*"_1_((),())"
        AiRe = index_key(Akey*"Re");
        AiIm = index_key(Akey*"Im");
        Acomplex = im*mps[keys_mps[AiIm]] + mps[keys_mps[AiRe]]; #Re[A] + i Im[A]
        transpose == false ? push!(Aone, Acomplex) : push!(Aone, Acomplex');
    end

    if transpose == false
        Aone = reverse(Aone)
        Azero = reverse(Azero)
    end
    
    #* Rebuild state
    state = zeros(ComplexF64, 2^sites);
    for n ∈ 0:2^sites-1    
        tensors = [];
        basis_n=basis(n, sites);
        for σ ∈ 1:sites
            basis_n[σ] == 0 ? push!(tensors, Azero[σ]) : push!(tensors, Aone[σ]);
        end
        state[n+1] = prod(tensors)[1] #Calculates the coefficient of the basis state n
    end
    return state
end;




Ψ_reb_notrunc = spzeros(ComplexF64, 64, 2^16);
Ψ_reb_notrunc_transp = spzeros(ComplexF64, 64, 2^16);

Threads.nthreads()

m=1;
for n in 5:5:320
    mps_step_n = read(Ψ_all_notrunc["mps/step_$n"]);
    ψ_step = reconstructcomplexState(mps_step_n);
    ψ_step_transpose = reconstructcomplexState(mps_step_n, transpose=true);
    Ψ_reb_notrunc[m,:] = ψ_step;
    Ψ_reb_notrunc_transp[m,:] = ψ_step_transpose;
    m +=1;
end



#Ψ_reb_all_trunc = h5open(lisa_loc_trunc*"PL_alpha_0.5_eps_svd=1e-6_states.h5", "w"); #creates the collection of states

Ψ_reb_all_notrunc = h5open(lisa_loc_notrunc*"PL_alpha_0.5_eps_svd=0.0_states.h5", "w"); #creates the collection of states

create_group(Ψ_reb_all_notrunc, "states")

Ψ_reb_all_notrunc["states/reverse"] = Matrix(Ψ_reb_notrunc);
Ψ_reb_all_notrunc["states/transpose"] = Matrix(Ψ_reb_notrunc_transp);

close(Ψ_reb_all_notrunc)

#* -----------------------------------------


lisa_loc_notrunc = "/mnt/lisa/PL_alpha_0.5/NOTRUNC/"
Ψ_all_notrunc = h5open(lisa_loc_notrunc*"PL_alpha_0.5_eps_svd=0.0.h5", "r"); #creates the collection of states



steps = collect(5:5:320);
Ψ_reb_notrunc = zeros(ComplexF64, 64, 2^16);
Ψ_reb_notrunc_transp = zeros(ComplexF64, 64, 2^16);

for n in 1:64
    step_n = steps[n]
    mps_step_n = read(Ψ_all_notrunc["mps/step_$step_n"]);
    ψ_step = reconstructcomplexState(mps_step_n);
    ψ_step_transpose = reconstructcomplexState(mps_step_n, transpose=true);
    Ψ_reb_notrunc[n,:] = ψ_step;
    Ψ_reb_notrunc_transp[n,:] = ψ_step_transpose;
end


Ψ_reb_all_notrunc = h5open(lisa_loc_notrunc*"PL_alpha_0.5_eps_svd=0.0_states.h5", "w"); #creates the collection of states
create_group(Ψ_reb_all_notrunc, "states")

Ψ_reb_all_notrunc["states/reverse"] = Matrix(Ψ_reb_notrunc);
Ψ_reb_all_notrunc["states/transpose"] = Matrix(Ψ_reb_notrunc_transp);
close(Ψ_reb_all_notrunc);



