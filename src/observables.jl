############################## Methods ##############################

function operator_density(U_t::MPO, O::Matrix{T}, loc_O::Int64, Λ::Int64; normalized = true) where {T}
    
    ##* Idea
    # -move ONC to site 1. In some cases, this can lead to a smaller dim of the tensors D and D†
    # -apply operator ̂O to U(t)
    # -trace W(t) from left to right to produce tensor D
    # -contract D and D† with new colum of tensors of Wₖ(t) and Wₖ(t)†


    # move ONC inside L\Λ. I don't like to have to do this "casting" twice, should fix it at some point
    #U_mps = cast_mps(U_t);
    #canonize!(U_mps; direction = "right", final_site = 1);
    #U_t = cast_mpo(U_mps);

    ## Calculates Ô ⋅ U(t)
    @tensor OxW[u, l, d, r] := O[u, α] * U_t.Wi[loc_O][α, l, d, r]; 
    OxUt(i) = ((i == loc_O) ? (return OxW) : (return U_t.Wi[i]))

    ## Calculate Wₖ(t) = tr_D W(t)
    L = U_t.L;
    D = L - Λ;
    Li = ones(1,1); # dummy identity matrix

    for i ∈ 1:D
        @tensor Li_U[u, r1, r2, d] := Li[α, r2] * conj(U_t.Wi[i])[d, α, u, r1] # Contract results with next tensor of U†(t)
        @tensor Li[r1, r2] := Li_U[α, r1, β, γ] * OxUt(i)[γ, β, α, r2] # Contract Ô⋅U(t) with U(t)†
    end

    ### Calculate tr Wₖ(t)Wₖ(t)
    ## Contract with tensors at i = D+1
    @tensor LiUD[u, r1, r2, d] := Li[α, β] * conj(U_t.Wi[D + 1])[γ, α, u, r1] * OxUt(D + 1)[γ, β, d, r2];    

    ## First contraction to create large rank-4 tensor
    @tensor Li[r1, r2, r3, r4] := LiUD[α, r1, r2, β] * LiUD[β, r3, r4, α]
    
    ## Contract till end of chain
    for i ∈ D+2:L
        @tensor Li[r1, r2, r3, r4] := Li[α, β, γ, δ] * conj(U_t.Wi[i])[p2, α, p1, r1] * OxUt(i)[p2, β, p3, r2] * conj(U_t.Wi[i])[p4, γ, p3, r3] * OxUt(i)[p4, δ, p1, r4]
    end

    normalized == false && return Li[1, 1, 1, 1]/(2^(L + D))
    normalized == true && return Li[1, 1, 1, 1]*2^Λ
end


"""
    function operator_density(W_t::MPO; normalized = true)

## Description
Calculates the operator density ``\\rho_\\ell`` of an operator ``\\mathcal{W}(t)`` represented as a MPO, where the size of the support is counted from the last site `L` of the system. Therefore the initial support of ``\\mathcal{W}(0)`` should be also `L` for correct results.

## Notes
This version considers a non-hermitian ``\\mathcal{W}(t)``. The exact representation of ``\\mathcal{W}(t)`` is however hermitian but compression of this operator leads to losing this property.
"""
function operator_density(W_t::MPO; normalized = true)
    
	L = W_t.L;
    ρ_s = Float64[];

    ## Calculate ρₛ = ∑ρₗ
    for Λ ∈ 1:L
        ## Calculate Wₖ(t) = tr_D W(t)
        D = L - Λ;

        VWi = W_t.Wi[1];
        for i ∈ 1:D
            @tensor VWi[u, l, d, r] := VWi[α, l, α, β] * W_t.Wi[i+1][u, β, d, r]
        end
        VWi = VWi[:, 1, :, :];

        ## Calculate tr W†ₖWₖ. 
        
        ##* After compression W ≠ W†, which has to be taken into account when calculating ⟨W,W⟩. 
        ## TODO: Find where the hermitian character is lost!

        # @tensor Li[r1, r2] := VWi[α, β, r1] * VMi[β, α, r2] # Site D+1. For W hermitian
        @tensor Li[r1, r2] := VWi[α, β, r1] * conj(VWi)[α, β, r2] # Site D+1
        for i ∈ D+2:L
            @tensor LiU[u, r1, r2, d] := Li[α, r2] * W_t.Wi[i][u, α, d, r1];
            #@tensor Li[r1, r2] := LiU[α, r1, γ, β] * W_t.Wi[i][β, γ, α, r2]; # For W hermitian
            @tensor Li[r1, r2] := LiU[α, r1, γ, β] * conj(W_t.Wi[i])[α, γ, β, r2];
        end

        normalized == false && push!(ρ_s, abs(Li[1, 1])/(2^(L + D)))
        normalized == true && push!(ρ_s, abs(Li[1, 1])/2^D)
    end
    
    ## Calculate ρₗ
    ρ_l = prepend!([ρ_s[n] - ρ_s[n-1] for n ∈ 2:L], ρ_s[1]);

    return ρ_l
end


entanglement_entropy(es::Vector{Float64}) = -1 * sum([wᵢ^2 * log(wᵢ^2) for wᵢ ∈ es]);


"""
    operator_size(rho_t)

The dimensions of the input is `number of steps` x `number of sites`
"""
operator_size(rho_t) = [sum([l * rho_t[s, l] for l ∈ axes(rho_t, 2)]) for s ∈ axes(rho_t, 1)];


#= 
function operator_density_old(U_t::MPO, O, loc_O::Int64, Λ::Int64)
    L = U_t.L;
    𝟙 = [1 0; 0 1];    ℤ = [1 0; 0 -1];        𝕐	= im*[0 -1; 1 0];       𝕏 = [0 1; 1 0];
    𝕀 = (𝕏+𝕐+ℤ+𝟙); # operator space identity
    𝕏𝕐ℤ = 𝕏+𝕐+ℤ; 
    
    # Applies single-site operator ̂O to U(t)
    OxU = deepcopy(U_t);
    @tensor OxW[u, l, d, r] := U_t.Wi[loc_O][u, l, x, r]*O[x, d]; 
    OxU.Wi[loc_O] = OxW; 

    # Contracts projector with U(t)†. #? not sure
    PxU = deepcopy(U_t);
    for i ∈ 1:L #! note the inverted order of the physical legs of the tensor Wᵢ
        i > Λ && (@tensor PxW[u, l, d, r] := conj(U_t.Wi[i])[x, l, u, r]*𝟙[x, d];) # for sites beyond the support Λ of the operator P
        i < Λ && (@tensor PxW[u, l, d, r] := conj(U_t.Wi[i])[x, l, u, r]*𝕀[x, d];) # for sites inside the support Λ of the operator P
        i == Λ && (@tensor PxW[u, l, d, r] := conj(U_t.Wi[i])[x, l, u, r]*𝕏𝕐ℤ[x, d];) # for the edge of the support Λ of the operator P
        PxU.Wi[i] = PxW;
    end
    

    # Calculate tr(PU†OU) using rank-4 tensors
    # Li×Wi tensor contraction:
    #  _            ⟂
    # | |-- r1   --|_|--
    # | |           |
    # |_|-- r2
       
    Li_U = OxU.Wi[1];
    for i ∈ 1:L-1
        @tensor Li[r1, r2] := Li_U[x, y, z, r1]*PxU.Wi[i][z, y, x, r2] # Contractions along PxU
        @tensor Li_U[u, r2, d, r1] := Li[x, r2]*OxU.Wi[i+1][u, x, d, r1] # Contractions along OxU
    end
    
    overlap = 0.0;
    @tensor overlap = Li_U[x, y, z, e]*PxU.Wi[L][z, y, x, e] # Contraction with last tensor
    return overlap
end



function operator_density_wrong(U_t::MPO, O, loc_O::Int64, Λ::Int64)
    
    ##* Idea
    # -move ONC to site 1. In some cases, this can lead to a smaller dim of the tensors D and D†
    # -apply operator ̂O to U(t)
    # -trace W(t) from left to right to produce tensor D
    # -contract D and D† with new colum of tensors of Wₖ(t) and Wₖ(t)†


    # move ONC inside L\Λ. I don't like to have to do this "casting" twice, should fix it at some point
    #U_mps = cast_mps(U_t);
    #canonize!(U_mps; direction = "right", final_site = 1);
    #U_t = cast_mpo(U_mps);

    ## Calculates Ô ⋅ U(t) and U(t) ⋅ Ô
    @tensor OxW[u, l, d, r] := O[u, α] * U_t.Wi[loc_O][α, l, d, r]; 
    OxUt(i) = ((i == loc_O) ? (return OxW) : (return U_t.Wi[i]))

    @tensor WxO[u, l, d, r] := O[α, d] * U_t.Wi[loc_O][u, l, α, r]; 
    UtxO(i) = ((i == loc_O) ? (return WxO) : (return U_t.Wi[i]));


    ## Calculate Wₖ(t) = tr_D W(t)
    D = L - Λ;
    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_U[u, r1, r2, d] := Li[α, r2] * conj(U_t.Wi[i])[d, α, u, r1] # Contract results with next tensor of U†(t)
        @tensor Li[r1, r2] := Li_U[α, r1, β, γ] * OxUt(i)[γ, β, α, r2] # Contract Ô⋅U(t) with U(t)†
    end

    ## Calculate Wₖ† = tr_D W†(t)    
    Li_dag = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_dag_U[u, r1, r2, d] := Li_dag[α, r2] * UtxO(i)[u, α, d, r1] # Contract results with U(t).W_i
        @tensor Li_dag[r1, r2] := Li_dag_U[α, r1, β, γ] * conj(U_t.Wi[i])[α, β, γ, r2] # Contract U(t)U(t)†
    end


    ### Calculate tr Wₖ(t)Wₖ(t)†
    ## Contract with tensors at i = D+1
    @tensor LiUD[u, r1, r2, d] := Li[α, β] * conj(U_t.Wi[D + 1])[γ, α, u, r1] * OxUt(D + 1)[γ, β, d, r2];    
    @tensor LiUD_dag[u, r3, r4, d] := Li_dag[α, β] * UtxO(D + 1)[u, α, γ, r3] * conj(U_t.Wi[D + 1])[d, β, γ, r4]; #! r3 and r4 orders?

    ## First contraction to create large rank-4 tensor
    @tensor Li[r1, r2, r3, r4] := LiUD[α, r1, r2, β] * LiUD_dag[β, r3, r4, α]
    
    ## Contract till end of chain
    for i ∈ D+2:L
        @tensor Li[a, b, c, d] := Li[α, β, γ, δ] * conj(U_t.Wi[i])[p2, α, p1, a] * OxUt(i)[p2, β, p3, b] * UtxO(i)[p3, γ, p4, c] * conj(U_t.Wi[i])[p1, δ, p4, d]; # U(t)*U(t)†*U(t)†*U(t)
    end

    return Li[1, 1, 1, 1]*(2^Λ)
end

 =#