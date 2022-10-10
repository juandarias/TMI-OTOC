#### Norm compressed W(t) ####
##############################

function load_tensors(input_file)
#    data = h5open("$(obelix_folder)/D_$(Dmax)/$(input_file).h5", "r");
    data = h5open(input_file, "r");
    Wi = Vector{Array{ComplexF64, 4}}();
    for n ∈ 1:10
        push!(Wi, read(data["Tensors/Wi_$(n)"]))
    end
    return MPO(Wi)
end


Wt_norm = zeros(96);

for (s, step) ∈ enumerate(5:100)
    filename = "Wt_alpha=2.5_N=10_t=5.0_dt=0.05_tol=1.0e-6_kac_corr_step=$(step).h5";
    datafolder = "/mnt/obelix/TMI/WII/D_190/"

    Wt = load_tensors(datafolder*filename);

    @tensor tr_norm[l1, l2, r1, r2] := Wt.Wi[1][α, l1, β, r1] * Wt.Wi[1][β, l2, α, r2];
    tr_norm = tr_norm[1, 1, :, :];

    for i ∈ 2:10
        @tensor tr_norm[r1, r2] := tr_norm[γ, δ] * Wt.Wi[i][α, γ, β, r1] * Wt.Wi[i][β, δ, α, r2];
    end

    Wt_norm[s] = abs(tr_norm[1,1]);
end

norm_Wt = plot(collect(5:100)*0.05, log10.(abs.(-Wt_norm .+ 1)), xlabel = L"tJ", ylabel=L"\log_{10} [1-\langle \mathcal{W}, \mathcal{W} \rangle]", label = L"\alpha = 2.5, \quad dt=0.05");
plot!(thickness_scaling=1.5, legend = :topleft)
savelatexfig(norm_Wt, plotsdir("WII/Wt_norm_alpha=2.5_dt=0.5_Dmax=190_kac_corr"))

step = 50;
filename = "Wt_alpha=2.5_N=10_t=5.0_dt=0.05_tol=1.0e-6_kac_corr_step=$(step).h5";
datafolder = "/mnt/obelix/TMI/WII/D_190/"

Wt = load_tensors(datafolder*filename);
Wt_mps = cast_mps(Wt);
overlap(Wt_mps, Wt_mps)


Wt_norm_alt = zeros(96);

for (s, step) ∈ enumerate(5:100)
    filename = "Wt_alpha=2.5_N=10_t=5.0_dt=0.05_tol=1.0e-6_kac_corr_step=$(step).h5";
    datafolder = "/mnt/obelix/TMI/WII/D_190/"

    Wt = load_tensors(datafolder*filename);
    Wt_mps = cast_mps(Wt);
    println(maximum(Wt_mps.D))
    
    Wt_norm_alt[s] = abs(overlap(Wt_mps, Wt_mps));
end

plot(collect(5:100)*0.05, log10.(abs.(-Wt_norm .+ 1)));
plot!(collect(5:100)*0.05, log10.(abs.(-Wt_norm_alt .+ 1)))


## Checking inner product using matrices
Wtm = Wt.Wi[1][:,1,:,:];
for i ∈ 2:9
    @tensor Wtm[ui, un, di, dn, ai] := Wtm[ui, di, x]*Wt.Wi[i][un, x, dn, ai]
    Wtm = reshape(Wtm, 2^(i), 2^(i), Wt.D[i])
end 
@tensor Wtm[ui, un, di, dn] := Wtm[ui, di, x]*(Wt.Wi[10][:, :, :, 1])[un, x, dn];

Wtm = reshape(Wtm, 2^N, 2^N);

tr(adjoint(Wtm)*Wtm)


@tensor tr_norm[l1, l2, r1, r2] := Wt.Wi[1][α, l1, β, r1] * (Wt.Wi[1])[β, l2, α, r2];
tr_norm = tr_norm[1, 1, :, :];

for i ∈ 2:10
    @tensor tr_norm[r1, r2] := tr_norm[γ, δ] * Wt.Wi[i][α, γ, β, r1] * (Wt.Wi[i])[β, δ, α, r2];
end

tr_norm[1,1]




𝕐 = im*[0.0 -1.0; 1.0 0.0]
Wt = calc_Wt(U_dt, 𝕐, 2);



## Checking inner product calculation for unitary

W_II = WII(2.5, 10, -1.0, 1.0, 2.0, 0.1, 5, 5);
W_II_dag = WII(2.5, 10, -1.0, 1.0, 2.0, -0.1, 5, 5);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, 10-2)..., W_II.WN]));
U_dt_dag = MPO(copy([W_II_dag.W1, fill(W_II_dag.Wi, 10-2)..., W_II_dag.WN]));

@tensor tr_norm[l1, l2, r1, r2] := U_dt.Wi[1][α, l1, β, r1] * conj(U_dt.Wi[1])[β, l2, α, r2];
tr_norm = tr_norm[1, 1, :, :];

for i ∈ 2:10
    @tensor tr_norm[r1, r2] := tr_norm[γ, δ] * U_dt.Wi[i][α, γ, β, r1] * conj(U_dt.Wi[i])[β, δ, α, r2];
end

tr_norm

Umps = cast_mps(U_dt);

overlap(Umps, Umps)


#### Uncompressed W(t)

W_II = WII(2.5, 10, -1.0, 1.0, 2.0, 0.005, 5, 5);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, 10-2)..., W_II.WN]));
𝕐 = im*[0.0 -1.0; 1.0 0.0]
Wt = calc_Wt(U_dt, 𝕐, 2);

## Rebuild operator
function rebuild_operator(MPO)
    N = MPO.L;
    Wtm = MPO.Wi[1][:,1,:,:];
    for i ∈ 2:N-1
        @tensor Wtm[ui, un, di, dn, ai] := Wtm[ui, di, x]*MPO.Wi[i][un, x, dn, ai]
        Wtm = reshape(Wtm, 2^(i), 2^(i), MPO.D[i])
    end 
    @tensor Wtm[ui, un, di, dn] := Wtm[ui, di, x]*(MPO.Wi[N][:, :, :, 1])[un, x, dn];
    return reshape(Wtm, 2^N, 2^N);
end

Wtm = rebuild_operator(U_dt);

tr(Wtm*Wtm)
tr(adjoint(Wtm)*Wtm)





## MPS overlap
Wt_mps = cast_mps(Wt);
overlap(Wt_mps, Wt_mps)

## Contraction
function inner_product(MPO)

    L = MPO.L;
    @tensor tr_norm[l1, l2, r1, r2] := MPO.Wi[1][α, l1, β, r1] * conj(MPO.Wi[1])[α, l2, β, r2];
    tr_norm = tr_norm[1, 1, :, :];

    for i ∈ 2:L
        @tensor tr_norm[r1, r2] := tr_norm[γ, δ] * MPO.Wi[i][α, γ, β, r1] * conj(MPO.Wi[i])[α, δ, β, r2];
    end
    return tr_norm[1,1]
end

inner_product(Wt)

function inner_productB(MPO)
    L = MPO.L;

    @tensor tr_norm[l1, l2, r1, r2] := MPO.Wi[1][α, l1, β, r1] * MPO.Wi[1][β, l2, α, r2];
    tr_norm = tr_norm[1, 1, :, :];

    for i ∈ 2:L
        @tensor tr_norm[r1, r2] := tr_norm[γ, δ] * MPO.Wi[i][α, γ, β, r1] * MPO.Wi[i][β, δ, α, r2];
    end

    return tr_norm[1,1]
end

inner_productB(Wt)

#### Compressed W(t)
#! W(t) ≠ W(t)†
#? When is the hermitianity broken? In the case of an Hermitian operator, one expects that the tensors of the MPO obey A^{ket,bra}_{left,right} = (A^{bra,ket}_{left,right})*
#* - SVD truncation preserves symmetry
step = 50;
filename = "Wt_alpha=2.5_N=10_t=5.0_dt=0.05_tol=1.0e-6_kac_corr_step=$(step).h5";
datafolder = "/mnt/obelix/TMI/WII/D_190/"

Wt = load_tensors(datafolder*filename);

Wtm = rebuild_operator(Wt);
tr(Wtm*Wtm)
tr(adjoint(Wtm)*Wtm) #! 

inner_product(Wt)
inner_productB(Wt)


