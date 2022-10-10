abstract type Hamiltonian{N} end

using LinearAlgebra
LinearAlgebra.ishermitian(H::Hamiltonian) = true
Base.ndims(H::Hamiltonian{N}) where {N} = N


struct TransverseFieldIsing{N} <: Hamiltonian{N}
    dims::NTuple{N, Int}
    h::Array{Float64, 1}
    J::Array{Float64, N}
    row_ixs::Vector{Int}
    col_ixs::Vector{Int}
    nz_vals::Vector{ComplexF64}
    # inner c-tor
    function TransverseFieldIsing{N}(dims, h, J, row_ixs, col_ixs, nz_vals) where {N}
        new(dims, h, J, row_ixs, col_ixs, nz_vals)
    end
end

#Base.convert(SparseMatrixCSC, tf::TransverseFieldIsing) = sparse(tf.row_ixs, tf.col_ixs, tf.nz_vals)

function TransverseFieldIsing(J::Array{Float64, 2}, h::Vector{Float64}; term::String="Both")
    N_sites = size(J)[2]
    length(h) == 1 || length(h) == N_sites || throw(ArgumentError("h must be either length 1 or length $N_sites"))
    length(h) == 1 && (hfull = fill(h[1], N_sites))
    
    row_ixs = Int[]
    col_ixs = Int[]
    nz_vals = ComplexF64[]
    for input_ind in 0:2^N_sites-1 #input-ind -> input state index
        input_bits = Vector{Bool}(digits(input_ind, base=2, pad=N_sites)) #state representation
        
        
        #return only TF term
        if term != "I"
            # TF-term
            @inbounds for ii in 1:N_sites
                output_bits = copy(input_bits)
                output_bits[ii] ⊻= true #flip state of spin with a xor
                output_ind = sum([output_bits[jj]<<(jj-1) for jj in 1:N_sites]) #output state index, << is the logical left shift operator
                push!(row_ixs, input_ind+1)
                push!(col_ixs, output_ind+1)
                push!(nz_vals, (1/2)*hfull[ii])
            end
        end
        
        #return only I term
        if term != "TF"
            # I-term
            push!(row_ixs, input_ind+1)
            push!(col_ixs, input_ind+1)
            nz_val = 0.0
            @inbounds for site in 1:N_sites
                @inbounds for next_site in site+1:N_sites
                    bond_J = (input_bits[site] == input_bits[next_site]) ? (1/4)*J[site,next_site] : -(1/4)*J[site,next_site]
                    nz_val += bond_J
                end
            end
            push!(nz_vals, nz_val)
        end
    end
    return TransverseFieldIsing{2}((N_sites,N_sites), hfull, J, row_ixs, col_ixs, nz_vals)
end

##########################################
# Old
##########################################

struct TransverseFieldIsingOld{N} <: Hamiltonian{N}
    dims::NTuple{1, Int}
    h::Array{Float64, 1}
    J::Array{Float64, 2}
    Ham::Array{Float64, 2}
    # inner c-tor
    function TransverseFieldIsingOld{N}(dims, h, J, Ham) where {N}
        new(dims, h, J, Ham)
    end
end






#https://juliaphysics.github.io/PhysicsTutorials.jl/tutorials/general/quantum_ising/quantum_ising.html
function TransverseFieldIsing1DNN(;N,h)
    ⊗(x,y) = kron(x,y)
    id = [1 0; 0 1] |> sparse
    σˣ = [0 1; 1 0] |> sparse
    σᶻ = [1 0; 0 -1] |> sparse
    
    first_term_ops = fill(id, N)
    first_term_ops[1] = σᶻ
    first_term_ops[2] = σᶻ
    
    second_term_ops = fill(id, N)
    second_term_ops[1] = σˣ
    
    H = spzeros(Int, 2^N, 2^N) # note the spzeros instead of zeros here
    for i in 1:N-1
        H -= foldl(⊗, first_term_ops)
        first_term_ops = circshift(first_term_ops,1)
    end
    
    for i in 1:N
        H -= h*foldl(⊗, second_term_ops)
        second_term_ops = circshift(second_term_ops,1)
    end
    H
end

function TransversalIsingSimple(coupling_matrix, field)
    N = size(coupling_matrix)[2]
    return sum([coupling_matrix[i,j]*σᶻᵢσᶻⱼ(i,j,N) for i=1:N for j=i+1:N]) + sum([field[i]*σˣᵢ(i,N) for i=1:N])
end

function TransverseFieldIsingOld(coupling_matrix, field)
    N = size(coupling_matrix)[2]
    length(field) == 1 || length(field) == N || throw(ArgumentError("h must be either length 1 or length $N"))
    length(field) == 1 && (field = fill(field[1], N))

    ⊗(x,y) = kron(x,y)    
    id = [1 0; 0 1] |> sparse
    σˣ = [0 1; 1 0] |> sparse
    second_term_ops = fill(id, N)
    second_term_ops[1] = σˣ
    
    H = spzeros(Float64, 2^N, 2^N)
    for i=1:N, j=i+1:N
        H += coupling_matrix[i,j]*σᶻᵢσᶻⱼ(i,j,N)
    end

    
    for i in 1:N
        H += field[i]*foldl(⊗, second_term_ops)
        second_term_ops = circshift(second_term_ops,1)
    end
    

    return TransverseFieldIsingOld{2}((N,), field, coupling_matrix, H)
end
