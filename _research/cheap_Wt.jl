WU_dt = prod(W_t, U_dt); # W(t_i)*U(dt)
#? Perhaps I need an intermediate compression here

#var_params[:seed] = mpo_compress(WU_dt; svd_params...); # generate a seed by using SVD compression
#WU_dt = mpo_compress(W_t; var_params...);

## Second step, calculate W(t) and compress
log_message("\n Preparing U(dt)†*W(t_i)*U(dt) \n"; color = :blue)
W_t_mps = cast_mps(prod(conj(U_dt), WU_dt); normalize = false); # U(dt)†*W(t_i)*U(dt)
sweep_qr!(W_t_mps);


function calcWt!(Udt, Wt)
    local Ri
    for i ∈ 1:N
        @tensor Wi[u, r1, r2, r3, d, l1, l2, l3] := conj(Udt.Wi[i])[u, r1, β, l1] * Wt.Wi[i][α, r2, β, l2] * Udt.Wi[i][β, r3, d, l3];
        Wi = permutedims(Wi, (2,3,4,1,5,6,7,8));
        i == 1 && (Atilde = reshape(Wi, (prod(size(Wi)[1:5]), :)));
        i != 1 && (Atilde = Ri * reshape(Wi, (size(Ri, 2), :)));
        if i != N
            Atilde = reshape(Atilde, (d * size(Ri, 1), :));
            Qi, Ri = qr(Atilde);
            Qi = Matrix(Qi);
            update_tensor!(Wt, reshape(Qi, Int(size(Qi,1)/d), d, :), i);
        else
            update_tensor!(Wt, reshape(Atilde, (:, d, size(Atilde, 2))), N);
        end
    end

end

function cast_mps(mpo::MPO{T}; L::Int = mpo.L, normalize = false, Dmax::Int = 100) where {T}

    mps = MPS([zeros(T, 0,0,0) for i ∈ 1:L]); # initializes MPS of appropiate type

    update_tensor!(mps, reshape(permutedims(mpo.Wi[1], (2,1,3,4)), 1, 4, mpo.D[1]), 1);
    for i ∈ 2:L-1
        update_tensor!(mps, reshape(permutedims(mpo.Wi[i], (2,1,3,4)), mpo.D[i-1], 4, mpo.D[i]), i);
    end
    update_tensor!(mps, reshape(permutedims(mpo.Wi[end], (2,1,3,4)), mpo.D[end], 4, 1), L);

    mps.physical_space = BraKet();
    mps.d = mps.d^2;

    if normalize == true
        if maximum(mps.D) > Dmax # For large tensors, reduces the memory cost of calculating the norm at the price of doing a QR sweep of the MPS
            sweep_qr!(mps);
            println("Doing a sweep")
        end
        n = norm(mps);
        nxs = n^(-1/L);
        prod!(nxs, mps);
        #for n ∈ 1:L
        #    mps.Ai[n] = mps.Ai[n]/nxs;
        #end
    end

    return mps
end

function prod(
    mpo_top::MPO{T},
    mpo_bottom::MPO{T};
    compress::Bool=false,
    kwargs...
    ) where {T}

    Wi_prod = Vector{Array{T,4}}();
    L = mpo_top.L;

    for i ∈ 1:L
        @tensor Wi[u, l1, l2, d, r1, r2] := mpo_top.Wi[i][u, l1, x, r1]*mpo_bottom.Wi[i][x, l2, d, r2];
        i == 1  && push!(Wi_prod, reshape(Wi, (2, 1, 2, mpo_top.D[i]*mpo_bottom.D[i])));
        i != 1 && i != L && push!(Wi_prod, reshape(Wi, (2, mpo_top.D[i-1]*mpo_bottom.D[i-1], 2, mpo_top.D[i]*mpo_bottom.D[i])));
        i == L && push!(Wi_prod, reshape(Wi, (2, mpo_top.D[i-1]*mpo_bottom.D[i-1], 2, 1)));
    end

    if compress == false
        return MPO(Wi_prod);
    elseif compress == true
        println("returning compressed MPO")
        mpo, ϵ_c = mpo_compress(MPO(Wi_prod); kwargs...);
        return mpo, ϵ_c
    end
end




function sweep_qr!(mps::MPS; final_site::Int = mps.L, direction::String = "left")
    #Ai_new = Vector{Array{ComplexF64,3}}();
    L = mps.L;
    d = mps.d;

    if direction == "left" && mps.oc != L
        mps.canonical == None() && (mps.oc = 1;)
        @assert mps.oc < final_site "New orthogonality center must be right from current one"

        Atilde = reshape(mps.Ai[mps.oc], (:, mps.D[mps.oc]));

        for i ∈ mps.oc:final_site-1
            Qi, Ri = qr(Atilde);
            Qi = Matrix(Qi);
            update_tensor!(mps, reshape(Qi, Int(size(Qi,1)/d), d, :), i);

            Atilde = Ri * reshape(mps.Ai[i+1], (size(Ri,2), :));
            Atilde = reshape(Atilde, (d * mps.D[i], :));
        end
        update_tensor!(mps, reshape(Atilde, (:, d, size(Atilde, 2))), final_site);

        mps.canonical == None() && direction == "left" && (mps.canonical = Left();)
        mps.canonical == Right() && direction == "left" && (mps.canonical = Mixed();)
        mps.oc = final_site;

    elseif direction == "right" && mps.oc != 1
        mps.canonical == None() && (mps.oc = L;)
        @assert mps.oc > final_site "New orthogonality center must be left from current one"

        Atilde = reshape(mps.Ai[mps.oc], (mps.D[mps.oc-1], :));

        for i ∈ mps.oc:-1:final_site+1
            Qi, Ri = qr(adjoint(Atilde)); # Ai = R†Q†
            Qi = Matrix(Qi);
            update_tensor!(mps, reshape(collect(Qi'), (:, d, Int(size(Qi, 1)/d))), i);
            Atilde = reshape(mps.Ai[i-1], :, size(Ri, 2)) * adjoint(Ri); #Ai-1*Ri†
            Atilde = reshape(Atilde, (Int(size(Atilde, 1)/d), :));
        end
        update_tensor!(mps, reshape(Atilde, (size(Atilde, 1), d, :)), final_site);

        mps.canonical == None() && direction == "right" && (mps.canonical = Right();)
        mps.canonical == Left() && direction == "right" && (mps.canonical = Mixed();)
        mps.oc = final_site;
    end

    return nothing
end
