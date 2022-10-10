using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
using TensorOperations
using UnicodePlots

using dmrg_methods
using argparser

include(srcdir("observables.jl"))


##### Parameters #####
######################
args_dict = collect_args(ARGS)

## System
const L = get_param!(args_dict, "L", 10);
const alpha = get_param!(args_dict, "alpha", 2.5);
const tf = get_param!(args_dict, "tf", 5.0);
const dt = get_param!(args_dict, "dt", 0.05);

const op = get_param!(args_dict, "op", "X");
const loc_O = get_param!(args_dict, "loc_O", 1);


const start_step = get_param!(args_dict, "start_step", 1);
const end_step = get_param!(args_dict, "end_step", 10);
const t_step = get_param!(args_dict, "step", 10);


const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const Dmax = get_param!(args_dict, "Dmax", 190);
const DSVD_max = get_param!(args_dict, "DSVD", 500);

const JOB_ID = get_param!(args_dict, "ID", "X");


#### Load Ut ####
#################

function rebuild_Ut(address)
    Ut_h5 = h5open(address);
    tensors = Vector{Array{ComplexF64,4}}();
    for i ∈ 1:10
        push!(tensors, read(Ut_h5["Tensors/Wi_$(i)"]));
    end
    return MPO(tensors)
end


obelix_folder = "/mnt/obelix/TMI/WII/D_$(Dmax)/"
address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID)_step=$(t_step).h5"
U_t_mpo = rebuild_Ut(address);
U_t = mpo_compress(U_t_mpo; METHOD = COMPRESSOR(1), direction = "left", final_site = 10, Dmax = DSVD_max);


## Calculates Ô ⋅ U(t) and U(t) ⋅ Ô
X = [0 1; 1 0];
O = X;
O = [1 0; 0 1]

@tensor OxW[u, l, d, r] := O[u, α] * U_t.Wi[loc_O][α, l, d, r]; 
OxUt(i) = ((i == loc_O) ? (return OxW) : (return U_t.Wi[i]))

@tensor WxO[u, l, d, r] := O[α, d] * U_t.Wi[loc_O][u, l, α, r]; 
UtxO(i) = ((i == loc_O) ? (return WxO) : (return U_t.Wi[i]));

begin
    rho_l = [];

    for Λ ∈ 1:L
        
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
        #@tensor Li[r1, r2, r3, r4] := LiUD[α, r1, r2, β] * LiUD_dag[β, r3, r4, α]
        @tensor Li[r1, r2, r3, r4] := LiUD_dag[α, r1, r2, β] * LiUD[β, r3, r4, α]
        
        ## Contract till end of chain
        for i ∈ D+2:L
            #@tensor Li[a, b, c, d] := Li[α, β, γ, δ] * conj(U_t.Wi[i])[p2, α, p1, a] * OxUt(i)[p2, β, p3, b] * UtxO(i)[p3, γ, p4, c] * conj(U_t.Wi[i])[p1, δ, p4, d]; # U(t)*U(t)†*U(t)†*U(t)
            @tensor Li[a, b, c, d] := Li[α, β, γ, δ] * UtxO(i)[p1, α, p2, a] * conj(U_t.Wi[i])[p3, β, p2, b] * conj(U_t.Wi[i])[p4, γ, p3, c] * OxUt(i)[p4, δ, p1, d]; # U(t)*U(t)†*U(t)†*U(t)
        end

        push!(rho_l, Li[1, 1, 1, 1] / 2^(L + D))
    end
    println("Sum is: ", sum(abs.(rho_l)))
    display(scatterplot(abs.(rho_l)))#, yscale = :log10))

end

1/sum(abs.(rho_l))


#### Calculate reduced density matrices ####
############################################
begin
    rho_l = [];

    for Λ = 1:L
        D = L - Λ;
        Li = ones(1,1); # dummy identity matrix

        ## Calculate Wₖ(t) = tr_D W(t)
        for i ∈ 1:D
            @tensor Li_U[u, r1, r2, d] := Li[α, r2] * conj(U_t.Wi[i])[d, α, u, r1] # Contract results with next tensor of U†(t)
            @tensor Li[r1, r2] := Li_U[α, r1, β, γ] * OxUt(i)[γ, β, α, r2] # Contract Ô⋅U(t) with U(t)†
        end

        @tensor rho[u, r1, r2, d] := Li[α, β] * conj(U_t.Wi[D + 1])[γ, α, u, r1] * OxUt(D + 1)[γ, β, d, r2];    

        for i ∈ D+2:L
            rho = reshape(rho, 2^(i - D - 1), U_t.D[i - 1], U_t.D[i - 1], 2^(i - D - 1));
            @tensor rho[u, un, r1, r2, d, dn] := rho[u, α, β, d] * conj(U_t.Wi[i])[γ, α, un, r1] * OxUt(i)[γ, β, dn, r2]
        end
        rho = reshape(rho, 2^Λ, 1, 1, 2^Λ);
        rho = rho[:,  1, 1,  :];

        size(rho)

        ## Calculate Wₖ† = tr_D W†(t)    
        Li_dag = ones(1,1); # dummy identity matrix
        for i ∈ 1:D
            @tensor Li_dag_U[u, r1, r2, d] := Li_dag[α, r2] * UtxO(i)[u, α, d, r1] # Contract results with U(t).W_i
            @tensor Li_dag[r1, r2] := Li_dag_U[α, r1, β, γ] * conj(U_t.Wi[i])[α, β, γ, r2] # Contract U(t)U(t)†
        end

        @tensor rho_dag[u, r3, r4, d] := Li_dag[α, β] * UtxO(D + 1)[u, α, γ, r3] * conj(U_t.Wi[D + 1])[d, β, γ, r4]; #! r3 and r4 orders?

        for i ∈ D+2:L
            rho_dag = reshape(rho_dag, 2^(i - D - 1), U_t.D[i - 1], U_t.D[i - 1], 2^(i - D - 1));
            @tensor rho_dag[u, un, r3, r4, d, dn] := rho_dag[u, α, β, d] * UtxO(i)[un, α, γ, r3] * conj(U_t.Wi[i])[dn, β, γ, r4]
        end
        rho_dag = reshape(rho_dag, 2^Λ, 1, 1, 2^Λ);
        rho_dag = rho_dag[:,  1, 1,  :];
        
        
        ## Calculate tr Wₖ†Wₖ
        #@tensor trWWdag = rho[α, β] * rho_dag[α, β]
        @tensor trWWdag = rho[α, β] * rho[β, α] #!!!!!!!!!! Why this? Why rho_dag != rho†
        #trWWdag = tr(rho * rho) #!!!!!
        push!(rho_l, trWWdag/ 2^(L + D))
    end
    println("Sum is: ", sum(abs.(rho_l)))
    display(scatterplot(real.(rho_l)))#, yscale = :log10))
    display(abs.(rho_l))


end
#@tensor U_exact_r[l1, u1, u2, u3, u4, u5, u6, d1, d2, d3, d4, d5, d6, r5] := U_exact_mpo.Wi[1][u1, l1, d1, α] * U_exact_mpo.Wi[2][u2, α, d2, β] * U_exact_mpo.Wi[3][u3, β, d3, γ] * U_exact_mpo.Wi[4][u4, γ, d4, ϵ] * U_exact_mpo.Wi[5][u5, ϵ, d5, ν] * U_exact_mpo.Wi[6][u6, ν, d6, r5];



display(scatterplot(real.(rho_l), yscale = :log10))
println()
println(abs.(rho_l))
println("Sum is: ", sum(real.(rho_l)))

rho_m = []

for Λ ∈ 1:L
    push!(rho_m, operator_density(U_t, X, 1, Λ))
end

display(scatterplot(real.(rho_m), yscale = :log10))
println()
println(real.(rho_m))
println("Sum is: ", sum(real.(rho_m)))


#### Calculating tr(W⋅W) ####
#############################

begin
    rho_l = [];

    for Λ ∈ 1:L
        
        ## Calculate Wₖ(t) = tr_D W(t)
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

        push!(rho_l, Li[1, 1, 1, 1] / 2^(L + D))
    end
    println("Sum is: ", sum(abs.(rho_l)))
    display(scatterplot(abs.(rho_l)))#, yscale = :log10))
    display(abs.(rho_l))

end

includet(srcdir("observables.jl"))


t_step = 5

obelix_folder = "/mnt/obelix/TMI/WII/D_$(Dmax)/"
address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID)_step=$(t_step).h5"
U_t_mpo = rebuild_Ut(address);
U_t = mpo_compress(U_t_mpo; METHOD = COMPRESSOR(1), direction = "left", final_site = 10, Dmax = 30);
L = U_t_mpo.L


Y = im*[0 -1; 1 0];

U_t = mpo_compress(U_exact_mpo; METHOD = COMPRESSOR(1), direction = "left", final_site = 10, Dmax = 20, normalize = true);
U_t.D

W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj);
W_II_dag = WII(alpha, N, J, Bx, Bz, -dt, Ni, Nj);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));
U_dt_dag = MPO(copy([W_II_dag.W1, fill(W_II_dag.Wi, N-2)..., W_II_dag.WN]));

U_mps = cast_mps(U_dt);
U_mps_dag = cast_mps(U_dt_dag);

overlap(U_mps, U_mps)/2^N

W_initial = calc_Ut(W_II, 0.1);
U_t = MPO(copy([W_initial.Wi[1], fill(W_initial.Wi[2], N-2)..., W_initial.Wi[3]]));


U_t = mpo_compress(U_exact_mpo; METHOD = COMPRESSOR(1), direction = "right", final_site = 1, Dmax = 64, normalize = true);


begin
    rho_s = []
    for Λ ∈ 1:L
        push!(rho_s, operator_density(U_t, X, L, Λ; normalized = true))
    end
    rho_l = [rho_s[n] - rho_s[n-1] for n ∈ 2:L]
    prepend!(rho_l, rho_s[1])
    println("Sum is: ", sum(abs.(rho_l)))
    display(scatterplot(abs.(rho_l), yscale = :log10))
    display(abs.(rho_l))
    display(abs.(rho_s))
end

