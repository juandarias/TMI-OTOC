using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
using TensorOperations

using dmrg_methods
using argparser

include(srcdir("observables.jl"))

##### Parameters #####
######################
args_dict = collect_args(ARGS)

## System
const L = get_param!(args_dict, "L", 10);
const alpha = get_param!(args_dict, "alpha", 2.5);
const tf = get_param!(args_dict, "tf", 1.0);
const dt = get_param!(args_dict, "dt", 0.1);

const op = get_param!(args_dict, "op", "X");
const loc_O = get_param!(args_dict, "loc_O", 1);


const start_step = get_param!(args_dict, "start_step", 1);
const end_step = get_param!(args_dict, "end_step", 10);

const tol_compr = get_param!(args_dict, "tol_compr", 5e-6);
const Dmax = get_param!(args_dict, "Dmax", 200);

const JOB_ID = get_param!(args_dict, "ID", "X");

## Open results
function load_tensors(step::Int)
    input_file = "U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID)_step=$(step)";
#    data = h5open("$(obelix_folder)/D_$(Dmax)/$(input_file).h5", "r");
    data = h5open(datadir("WII/D_$(Dmax)/$(input_file).h5"), "r");
    Wi = Vector{Array{ComplexF64, 4}}();
    for n ∈ 1:L
        push!(Wi, read(data["Tensors/Wi_$(n)"]))
    end
    return MPO(Wi)
end


## Calculate densities
Y = im*[0 -1; 1 0];

output_file = "rho_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID)";
h5open(datadir("WII/results/$(output_file).h5"), "cw") do f;
    try
        create_group(f, "rho_l");
    catch
    end
    for s ∈ start_step:end_step
        U_t = load_tensors(s)
        ρl = zeros(ComplexF64, L);
        t = round(dt*s, digits = 3)
        log_message("\n Calculating ρ(t) for t = $(t), Λ  "; color = :blue);
        for Λ ∈ 1:L
            log_message("-> $(Λ)  "; time = false, color = :blue)
            ρl[Λ] = operator_density(U_t, Y, loc_O, Λ)
        end
        f["rho_l/step_$(s)"] = ρl
    end
end


alpha=2.5;
T=5.0;
dt=0.1;
tol_compr=1.0e-5;
JOB_ID=30;

output_file = "rho_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID).h5";
obelix_folder = "/mnt/obelix/TMI/WII/results/"

rho_data = h5open(obelix_folder * output_file, "r")

i = 20
rho_l = read(rho_data["rho_l/step_$(i)"])

rho_Λ(rho_l) = [rho_l[n] - rho_l[n-1] for n ∈ 10:-1:2]
    
rho_Λ(rho_l)


#### Test operator density calculation ####
###########################################

function rebuild_Ut(address)
    Ut_h5 = h5open(address);
    tensors = Vector{Array{ComplexF64,4}}();
    for i ∈ 1:10
        push!(tensors, read(Ut_h5["Tensors/Wi_$(i)"]));
    end
    return MPO(tensors)
end

alpha = 2.5;
T = 5.0;
dt = 0.05;
tol_compr = 5.0e-6;
JOB_ID = 70;
s = 10;
Dmax = 190;

obelix_folder = "/mnt/obelix/TMI/WII/D_$(Dmax)/"
address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID)_step=$(s).h5"
U_t_mpo = rebuild_Ut(address);

U_t = mpo_compress(U_t_mpo; METHOD = SVD, direction = "right", final_site = 1, Dmax = 20);

Ucm = cast_mps(U_t;  normalize = true);
Ulm = cast_mps(U_t_mpo;  normalize = true);
overlap(Ucm, Ulm)

## Checking norm = tr(U ⋅ U†)
Li = ones(1,1); # dummy identity matrix
for i ∈ 1:10
    @tensor Li_U[u, r2, d, r1] := Li[x, r2]*U_t.Wi[i][u, x, d, r1] # Contract results with next tensor of Ô⋅U(t)
    @tensor Li[r1, r2] := Li_U[x, y, z, r1]*conj(U_t.Wi[i])[x, y, z, r2] # Contract Ô⋅U(t) with U(t)† #! order of legs? Tensors are symmetric!
end
#* Li = 1 !!! Works


## Apply operator to Ut at l=1
Y = im*[0 -1; 1 0];

@tensor OxW[u, l, d, r] := Y[x, d]*U_t.Wi[1][u, l, x, r]; #! which is the right contraction order for Y?
#@tensor tOxW[u, l, d, r] := Y[d, x]*U_t.Wi[1][u, l, x, r];
OxUt(i) = ((i == 1) ? (return OxW) : (return U_t.Wi[i]))


rho_l = [];
for Λ ∈ 1:L

## Calculate partial trace Wₖ(t) = tr_L\Λ (W(t)), where W(t) = U(t)†⋅Ô⋅U(t)
    D = L - Λ

    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_U[u, r2, d, r1] := Li[x, r2]*OxUt(i)[u, x, d, r1] # Contract results with next tensor of Ô⋅U(t)
        @tensor Li[r1, r2] := Li_U[x, y, z, r1]*conj(U_t.Wi[i])[x, y, z, r2] # Contract Ô⋅U(t) with U(t)† #! order of legs?
    end


    ## Calculate partial trace W†ₖ(t) = tr_L\Λ (W†(t)), where W†(t) = U(t)⋅Ô⋅U†(t)

    Li_dag = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_dag_U[u, r2, d, r1] := Li_dag[x, r2] * conj(U_t.Wi[i])[d, x, u, r1] # Contract results with next tensor of U(t)†   #! order of legs?
        @tensor Li_dag[r1, r2] := Li_dag_U[x, y, z, r1] * OxUt(i)[z, y, x, r2] # Contract U(t)† with Ô⋅U(t)
    end


    ## Calculate tr Wₖ(t)Wₖ(t)†
    # Contract with tensors at i = D+1
    @tensor LiUD[u, r1, r2, d] := Li[x, y]*OxUt(D+1)[u, x, z, r1]*conj(U_t.Wi[D+1])[d, y, z, r2];
    @tensor LiUD_dag[u, r1, r2, d] := Li_dag[x, y] * conj(U_t.Wi[D+1])[z, x, u, r1] * OxUt(D+1)[z, y, d, r2];

    @tensor Di[r1, r2, r3, r4] := LiUD[x, r1, r2, y] * LiUD_dag[y, r3, r4, x];

    # Contract rest of chain
    for i ∈ D+2:L
        @tensor Di[r1, r2, r3, r4] := Di[r, s, t, u]*OxUt(i)[d1, r, d2, r1]*conj(U_t.Wi[i])[d3, s, d2, r2]*conj(U_t.Wi[i])[d4, t, d3, r3]*OxUt(i)[d4, u, d1, r4]; # U(t)*U(t)†*U(t)†*U(t)
    end

    push!(rho_l, Di[1,1,1,1])
end

#### Contraction exact unitary ####
###################################

using operators_basis
include(srcdir("hamiltonians.jl"));


alpha = 2.5;
J = -1.0;
Bx = 1.05;
Bz = -0.5;
L = 6;
s = 100;



function JPL(α, L)
    J = zeros(L,L);
    for i ∈ 1:L, j ∈ i+1:L
        J[i,j] = abs(i-j)^(-α)
    end
    kacn = sum(J)/(L-1)
    J += transpose(J)
    return J/kacn
    #return J
end

Jij = 4*J*JPL(2.5, L);
Bxe = 2*Bx;
Bze = 2*Bz;

TF = TransverseFieldIsing(Jij,[Bxe]);
TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
Hlong = Bze*spdiagm(sum([diag(Sᶻᵢ(i,L)) for i in 1:L]));
H_TMI = TF_M + Hlong;

U_exact = exp(-im * s * dt * collect(H_TMI));
U_exact_mpo = operator_to_mpo(U_exact);
U_t = deepcopy(U_exact_mpo);



#@tensor U_exact_r[l1, u1, u2, u3, u4, u5, u6, d1, d2, d3, d4, d5, d6, r5] := U_exact_mpo.Wi[1][u1, l1, d1, α] * U_exact_mpo.Wi[2][u2, α, d2, β] * U_exact_mpo.Wi[3][u3, β, d3, γ] * U_exact_mpo.Wi[4][u4, γ, d4, ϵ] * U_exact_mpo.Wi[5][u5, ϵ, d5, ν] * U_exact_mpo.Wi[6][u6, ν, d6, r5];
#U_exact_r =  U_exact_r[1, : ,: ,:, :, :, :, :, :, :, :, :, :, 1];
#size(U_exact_r)

# norm(reshape(U_exact_r, (2^6,2^6)) - U_exact)


## Checking norm = tr(U ⋅ U†)
Li = ones(1,1); # dummy identity matrix
for i ∈ 1:L
    @tensor Li_U[u, r2, d, r1] := Li[x, r2]*U_t.Wi[i][u, x, d, r1] # Contract results with next tensor of Ô⋅U(t)
    @tensor Li[r1, r2] := Li_U[x, y, z, r1]*conj(U_t.Wi[i])[x, y, z, r2] # Contract Ô⋅U(t) with U(t)† #! order of legs? Tensors are symmetric!
end
#* Li = 1 !!! Works


## Apply operator to Ut at l=1
Y = im*[0 -1; 1 0];
X = [0 1; 1 0]
I = [1 0 ; 0 1]
Y1 = σʸᵢ(1, L); # kron(I, I, I, I, I, Y)
X1 = σˣᵢ(1, L); # kron(I, I, I, I, I, Y)

# Exact
W_t = U_exact' * X1 * U_exact;

## Calculate partial trace Wₖ(t) = tr_L\Λ (W(t)), where W(t) = U(t)†⋅Ô⋅U(t)
# Exact
b2 = generate_basis(2)
I4 = kron(I,I,I,I);
I2 = kron(I,I);
W_t_2 = sum([(b2[i]' ⊗ I2) * W_t * (b2[i] ⊗ I2) for i ∈ 1:2^2]);
W_t_2 = sum([(I2 ⊗ b2[i]') * W_t * (I2 ⊗ b2[i] ) for i ∈ 1:2^2]);


## MPO
Λ = 4;
D = L - Λ

@tensor OxW[u, l, d, r] := X[d, x]*U_t.Wi[1][u, l, x, r]; #! which is the right contraction order for Y?
@tensor OxW[u, l, d, r] := X[u, x]*U_t.Wi[1][x, l, d, r]; #! which is the right contraction order for Y?
#@tensor tOxW[u, l, d, r] := Y[d, x]*U_t.Wi[1][u, l, x, r];
OxUt(i) = ((i == 1) ? (return OxW) : (return U_t.Wi[i]))

begin #* W†(t) = U(t)WU†(t)
    @tensor OxW[u, l, d, r] := X[d, x]*U_t.Wi[1][u, l, x, r]; #! which is the right contraction order for Y?
    OxUt(i) = ((i == 1) ? (return OxW) : (return U_t.Wi[i]))

    @tensor W_t_2_mpo[lk, lb, u3, u4, d3, d4, rk, rb] := OxUt(1)[t1, lk, r, k1] * conj(U_t.Wi[1])[t1, lb, r, b1] * OxUt(2)[t2, k1, s, k2] * conj(U_t.Wi[2])[t2, b1, s, b2] * OxUt(3)[u3, k2, t, k3] * conj(U_t.Wi[3])[d3, b2, t, b3] * OxUt(4)[u4, k3, u, rk] * conj(U_t.Wi[4])[d4, b3, u, rb];
    W_t_2_mpo = W_t_2_mpo[1, 1, : , :, :, :, 1, 1];
    
    W_t = U_exact * X1 * U_exact';
    W_t_2 = sum([(I2 ⊗ b2[i]') * W_t * (I2 ⊗ b2[i] ) for i ∈ 1:2^2]);
    norm(reshape(W_t_2_mpo, 4, 4) - W_t_2)

end

begin #* W(t)
    @tensor OxW[u, l, d, r] := X[u, x]*U_t.Wi[1][x, l, d, r]; #! which is the right contraction order for Y?
    OxUt(i) = ((i == 1) ? (return OxW) : (return U_t.Wi[i]))

    @tensor W_t_2_mpo[lk, lb, u3, u4, d3, d4, rk, rb] := conj(U_t.Wi[1])[r, lk, t1, k1] * OxUt(1)[r, lb, t1, b1] * conj(U_t.Wi[2])[s, k1, t2, k2] * OxUt(2)[s, b1, t2, b2] * conj(U_t.Wi[3])[t, k2, u3, k3] * OxUt(3)[t, b2, d3, b3] * conj(U_t.Wi[4])[u, k3, u4, rk] * OxUt(4)[u, b3, d4, rb];
    W_t_2_mpo = W_t_2_mpo[1, 1, : , :, :, :, 1, 1];

    W_t = U_exact' * X1 * U_exact;
    W_t_2 = sum([(I2 ⊗ b2[i]') * W_t * (I2 ⊗ b2[i] ) for i ∈ 1:2^2]);
    norm(reshape(W_t_2_mpo, 4, 4) - W_t_2)

end


# density matrix test

begin
    @tensor OxW[u, l, d, r] := X[u, α] * U_t.Wi[1][α, l, d, r]; 
    OxUt(i) = ((i == 1) ? (return OxW) : (return U_t.Wi[i]))
    
    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_U[u, r2, d, r1] := Li[α, r2] * conj(U_t.Wi[i])[d, α, u, r1] # Contract results with next tensor of U†(t)
        @tensor Li[r1, r2] := Li_U[α, β, γ, r1] * OxUt(i)[γ, β, α, r2] # Contract Ô⋅U(t) with U(t)†
    end
    
    @tensor rho[u, r1, r2, d] := Li[α, β] * conj(U_t.Wi[D + 1])[γ, α, u, r1] * OxUt(D + 1)[γ, β, d, r2];
    
    for i ∈ D+2:L
        rho = reshape(rho, 2^(i - D - 1), U_t.D[i - 1], U_t.D[i - 1], 2^(i - D - 1));
        @tensor rho[u, un, r1, r2, d, dn] := rho[u, α, β, d] * conj(U_t.Wi[i])[γ, α, un, r1] * OxUt(i)[γ, β, dn, r2]
    end
    
    rho_mpo = reshape(rho[:, :, 1, 1, :, :], (16, 16))
    
    W_t = U_exact' * X1 * U_exact;
    W_t_4 = sum([(I4 ⊗ b2[i]') * W_t * (I4 ⊗ b2[i] ) for i ∈ 1:2^2]);
    
    println(norm(W_t_4 - rho_mpo))    
end


begin #* D
    @tensor OxW[u, l, d, r] := X[u, α] * U_t.Wi[5][α, l, d, r]; 
    OxUt(i) = ((i == 5) ? (return OxW) : (return U_t.Wi[i]))

    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_U[u, r2, d, r1] := Li[α, r2] * conj(U_t.Wi[i])[d, α, u, r1] # Contract results with next tensor of U†(t)
        @tensor Li[r1, r2] := Li_U[α, β, γ, r1] * OxUt(i)[γ, β, α, r2] # Contract Ô⋅U(t) with U(t)†
    end

    @tensor rho[u, r1, r2, d] := Li[α, β] * conj(U_t.Wi[D + 1])[γ, α, u, r1] * OxUt(D + 1)[γ, β, d, r2];
end

begin #* D†
    @tensor OxW[u, l, d, r] := X[d, x]*U_t.Wi[5][u, l, x, r]; 
    OxUt(i) = ((i == 5) ? (return OxW) : (return U_t.Wi[i]));

    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_U[u, r2, d, r1] := Li[x, r2]*OxUt(i)[u, x, d, r1] # Contract results with U(t).W_i
        @tensor Li[r1, r2] := Li_U[x, y, z, r1]*conj(U_t.Wi[i])[x, y, z, r2] # Contract U(t)U(t)†
    end

    @tensor rho_dag[u, r1, r2, d] := Li[x, y] * OxUt(D + 1)[u, x, z, r1] * conj(U_t.Wi[D + 1])[d, y, z, r2];

end


#### Operator density with exact unitary ####
#############################################
using SparseArrays
using operators_basis
include(srcdir("hamiltonians.jl"));

alpha = 2.5;
J = -1.0;
Bx = 1.05;
Bz = -0.5;
dt = 0.05;
L = 6;
s = 100;

function JPL(α, L)
    J = zeros(L,L);
    for i ∈ 1:L, j ∈ i+1:L
        J[i,j] = abs(i-j)^(-α)
    end
    kacn = sum(J)/(L-1)
    J += transpose(J)
    return J/kacn
    #return J
end

Jij = 4*J*JPL(2.5, L);
Bxe = 2*Bx;
Bze = 2*Bz;

TF = TransverseFieldIsing(Jij,[Bxe]);
TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
Hlong = Bze*spdiagm(sum([diag(Sᶻᵢ(i,L)) for i in 1:L]));
H_TMI = TF_M + Hlong;

U_exact = exp(-im * s * dt * collect(H_TMI));
U_exact_mpo = operator_to_mpo(U_exact);
U_t = deepcopy(U_exact_mpo);

## Apply operator to Ut at l=1
Y = im*[0 -1; 1 0];
X = [0 1; 1 0]
I = [1 0 ; 0 1]
Y1 = σʸᵢ(1, L); # kron(I, I, I, I, I, Y)
X1 = σˣᵢ(1, L); # kron(I, I, I, I, I, Y)

loc_O = 1;
O = X;

## Calculates Ô ⋅ U(t) and U(t) ⋅ Ô
@tensor OxW[u, l, d, r] := O[u, α] * U_t.Wi[loc_O][α, l, d, r]; 
OxUt(i) = ((i == loc_O) ? (return OxW) : (return U_t.Wi[i]))

@tensor WxO[u, l, d, r] := X[α, d] * U_t.Wi[loc_O][u, l, α, r]; 
UtxO(i) = ((i == loc_O) ? (return WxO) : (return U_t.Wi[i]));



rho_l = [];

using UnicodePlots
scatterplot(real.(rho_l))

for Λ ∈ 1:L
    ## Calculate Wₖ(t) = tr_D W(t)
    D = L - Λ;
    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_U[u, r2, d, r1] := Li[α, r2] * conj(U_t.Wi[i])[d, α, u, r1] # Contract results with next tensor of U†(t)
        @tensor Li[r1, r2] := Li_U[α, β, γ, r1] * OxUt(i)[γ, β, α, r2] # Contract Ô⋅U(t) with U(t)†
    end

    ## Calculate Wₖ† = tr_D W†(t)    
    Li_dag = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_dag_U[u, r2, d, r1] := Li_dag[x, r2]*UtxO(i)[u, x, d, r1] # Contract results with U(t).W_i
        @tensor Li_dag[r1, r2] := Li_dag_U[x, y, z, r1]*conj(U_t.Wi[i])[x, y, z, r2] # Contract U(t)U(t)†
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

    push!(rho_l, Li[1, 1, 1, 1]/(2^(2D)))
end


#### Operator density with WII approx ####
##########################################


function rebuild_Ut(address)
    Ut_h5 = h5open(address);
    tensors = Vector{Array{ComplexF64,4}}();
    for i ∈ 1:10
        push!(tensors, read(Ut_h5["Tensors/Wi_$(i)"]));
    end
    return MPO(tensors)
end

L = 10;
alpha = 2.5;
tf = 5.0;
dt = 0.05;
tol_compr = 5.0e-6;
JOB_ID = 1;
s = 100;
Dmax = 500;

obelix_folder = "/mnt/obelix/TMI/WII/D_$(Dmax)/"
address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r_id$(JOB_ID)_step=$(s).h5"
U_t_mpo = rebuild_Ut(address);
U_t = mpo_compress(U_t_mpo; METHOD = COMPRESSOR(1), direction = "right", final_site = 1, Dmax = 30);


#Ucm = cast_mps(U_t;  normalize = true);
#Ulm = cast_mps(U_t_mpo;  normalize = true);
#overlap(Ucm, Ulm)

loc_O = 1;
O = X;

## Calculates Ô ⋅ U(t) and U(t) ⋅ Ô
@tensor OxW[u, l, d, r] := O[u, α] * U_t.Wi[loc_O][α, l, d, r]; 
OxUt(i) = ((i == loc_O) ? (return OxW) : (return U_t.Wi[i]))

@tensor WxO[u, l, d, r] := X[α, d] * U_t.Wi[loc_O][u, l, α, r]; 
UtxO(i) = ((i == loc_O) ? (return WxO) : (return U_t.Wi[i]));


rho_l = [];
scatterplot(real.(rho_l))
rho_l

for Λ ∈ 1:L
    ## Calculate Wₖ(t) = tr_D W(t)
    D = L - Λ;
    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_U[u, r2, d, r1] := Li[α, r2] * conj(U_t.Wi[i])[d, α, u, r1] # Contract results with next tensor of U†(t)
        @tensor Li[r1, r2] := Li_U[α, β, γ, r1] * OxUt(i)[γ, β, α, r2] # Contract Ô⋅U(t) with U(t)†
    end

    ## Calculate Wₖ† = tr_D W†(t)    
    Li_dag = ones(1,1); # dummy identity matrix
    for i ∈ 1:D
        @tensor Li_dag_U[u, r2, d, r1] := Li_dag[x, r2]*UtxO(i)[u, x, d, r1] # Contract results with U(t).W_i
        @tensor Li_dag[r1, r2] := Li_dag_U[x, y, z, r1]*conj(U_t.Wi[i])[x, y, z, r2] # Contract U(t)U(t)†
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

    push!(rho_l, Li[1, 1, 1, 1])
end


#### Compression of W(t) ####
#############################
using HDF5
using Plots
using LaTeXStrings
include(plotsdir("plotting_functions.jl"))
pgfplotsx()


## dt = 0.1
dt = 0.1;
tol = 1.0e-6;
data_folder = "/mnt/obelix/TMI/WII/D_190/"
root_file = "U_alpha=2.5_N=10_t=5.0_dt=$(dt)_tol=$(tol)_kac_corr_step="

e_c = zeros(49,2);

for step ∈ 2:50
    file = h5open("$(data_folder)$(root_file)$(step).h5");
    e_c[step-1,1] = read(file["Diagnosis/ϵ_c"]);
    e_c[step-1,2] = read(file["Diagnosis/Dmax"]);
end

time_axis = collect(0.2:0.1:5.0)

scatter(time_axis, abs.(e_c[:,1]), label=L"\epsilon_c", yscale = :log10)
Wt_comp = scatter(time_axis, abs.(e_c[:,1]), label=L"\epsilon_c", legend = :topleft, yscale = :log10, xlabel = L"t/J", ylabel = L"\epsilon_c")
scatter!(twinx(), xticks = :none, time_axis, abs.(e_c[:,2]), label=L"D_\textrm{max}", mc = :red, ylabel = L"D");
scatter!(thickness_scaling=1.5)
savelatexfig(Wt_comp, plotsdir("WII/op_dens/Wt_compression_dt=0.1_Dmax=190"))

## dt = 0.05
dt = 0.05;
tol = 1.0e-6;
data_folder = "/mnt/obelix/TMI/WII/D_190/"
root_file = "U_alpha=2.5_N=10_t=5.0_dt=$(dt)_tol=$(tol)_kac_corr_step="

e_c_dt_0_5 = zeros(99,2);

for step ∈ 2:100
    file = h5open("$(data_folder)$(root_file)$(step).h5");
    e_c_dt_0_5[step-1,1] = read(file["Diagnosis/ϵ_c"]);
    e_c_dt_0_5[step-1,2] = read(file["Diagnosis/Dmax"]);
end

time_axis = collect(0.1:0.05:5.0)

scatter(time_axis, abs.(e_c_dt_0_5[:,1]), label=L"\epsilon_c")
Wt_comp = scatter(time_axis, abs.(e_c_dt_0_5[:,1]), label=L"\epsilon_c", legend = :topleft, yscale = :log10, xlabel = L"t/J", ylabel = L"\epsilon_c")
scatter!(twinx(), xticks = :none, time_axis, abs.(e_c_dt_0_5[:,2]), label=L"D_\textrm{max}", mc = :red, ylabel = L"D");
scatter!(thickness_scaling=1.5)
savelatexfig(Wt_comp, plotsdir("WII/op_dens/Wt_compression_dt=0.05_Dmax=190"))


effect_dt = scatter(0.1:0.05:5.0, abs.(e_c_dt_0_5[:,1]), label=L"dt=0.05", legend = :topleft, yscale = :log10, xlabel = L"t/J")
scatter!(0.2:0.1:5.0, abs.(e_c[:,1]), label=L"dt=0.1", yscale = :log10)
ylabel!(L"\epsilon_c")
scatter!(thickness_scaling=1.5)
savelatexfig(effect_dt, plotsdir("WII/op_dens/Wt_compression_dt_effect"))


#### Memory cost reduced density matrix ####
############################################

mem_c = [38149.63053894043,55854.15173339844,79105.29263305664,108955.92297363281,146550.42303466797,193124.77053833008,250006.47833251953,318614.6144104004,400459.7984008789,497144.2305908203]

mem_c./1024

D = 100:10:190

mem_D = scatter(D, mem_c./1024, label = :none, xlabel = "D", ylabel = "GB")
savelatexfig(mem_D, plotsdir("WII/op_dens/reduced_density_mem_cost"))

#### Operator density with Kac correction ####
##############################################

using Plots, LaTeXStrings
op_size(rho_t) = [sum([l * rho_t[s, l] for l ∈ axes(rho_t, 2)]) for s ∈ axes(rho_t, 1)];


## dt = 0.05
datafolder = "/mnt/obelix/TMI/WII/results/"
filename = "rho_alpha=2.5_N=10_t=5.0_dt=0.05_tol=1.0e-6_Dmax=190_kac_corr.dat"

rawdata = readlines("$(datafolder)$(filename)")

rho_lines = rawdata[7:2:37]
step = collect(2:5:77)

rho_l = zeros(16, 10);
for (i, line) in enumerate(rho_lines)
    rho_l[i, :] = eval(Meta.parse(line[1:end-7]))
end


function normalize_rho(rho)
    rho_n = copy(rho)
    for s ∈ 1:size(rho)[1]
        rho_n[s, :] = rho[s, :]/sum(rho[s, :])
    end
    return rho_n
end

op_size(rho_t) = [sum([n * rho_t[s,n] for n ∈ 1:size(rho_t)[2]]) for s ∈ 1:size(rho_t)[1]]
rho_l_n = normalize_rho(rho_l)
Y_size = op_size(rho_l_n)

op_size_mpo = plot(0.05*step, op_size(rho_l_n), label = :none)
xlabel!("tJ")
ylabel!(L"L[\mathcal{W}(t)]")
plot!(thickness_scaling=1.5)
savelatexfig(op_size_mpo, plotsdir("WII/op_dens/op_size_dt=0.05_Dmax=190_kac_corr"))

## dt = 0.1

filename = "rho_alpha=2.5_N=10_t=5.0_dt=0.1_tol=1.0e-6_Dmax=190_kac_corr.h5"
datafolder = "/mnt/obelix/TMI/WII/results/"

f = h5open("$(datafolder)$(filename)");

rho_l_dt_0_1 = zeros(10, 10);
for (i, s) ∈ enumerate(2:5:50)
    rho_l_dt_0_1[i, :] = abs.(read(f["rho_l/step_$s"]))
end


step_dt_0_1 = 0.1*collect(2:5:50);
op_size_mpo = scatter(0.05*step, op_size(rho_l_n), label = "norm dt=0.05")
scatter!(0.05*step, op_size(rho_l), label = "unnorm dt=0.05")
scatter!(step_dt_0_1, op_size(normalize_rho(rho_l_dt_0_1)), label = "norm dt=0.1");
scatter!(step_dt_0_1, op_size(rho_l_dt_0_1), label = "unnorm dt=0.1")

savelatexfig(op_size_mpo, plotsdir("WII/op_dens/op_size_dt_comp_Dmax=190_kac_corr"))


## alpha = 2.5

datafolder = "/mnt/obelix/TMI/WII/results/"
filename = "rho_alpha=2.5_N=10_t=5.0_dt=0.05_tol=1.0e-6_Dmax=500_r_id36.h5"
f = h5open("$(datafolder)$(filename)");

rho_a_2_5 = zeros(99, 10);
for (i, s) ∈ enumerate(2:100)
    rho_a_2_5[i, :] = abs.(read(f["rho_l/step_$s"]))
end


## Reference alpha = 1.0
datafolder = "/mnt/obelix/TMI/WII/results/"
filename = "rho_alpha=1.0_N=10_t=5.0_dt=0.05_tol=1.0e-6_Dmax=500_r_id30.h5"
f = h5open("$(datafolder)$(filename)");

rho_a_1 = zeros(99, 10);
for (i, s) ∈ enumerate(2:100)
    rho_a_1[i, :] = abs.(read(f["rho_l/step_$s"]))
end

## Updated version: alpha = 1.0
datafolder = "/mnt/lisa/TMI/WII/results/"
filename = "rho_alpha=1.0_N=10_t=5.0_dt=0.05_tol=1.0e-6_Dmax=500_r_id4.h5"
f = h5open("$(datafolder)$(filename)");

rho_a_1_new = zeros(99, 10);
for (i, s) ∈ enumerate(2:100)
    rho_a_1_new[i, :] = abs.(read(f["rho_l/step_$s"]))
end


## Fast version: alpha = 1.0

datafolder = "/mnt/lisa/TMI/WII/results/"
filename = "rho_alpha=1.0_N=10_t=5.0_dt=0.05_tol=1.0e-6_Dmax=500_fast_code.h5"
f = h5open("$(datafolder)$(filename)");

rho_a_1_fast = zeros(99, 10);
for (i, s) ∈ enumerate(2:100)
    rho_a_1_fast[i, :] = abs.(read(f["rho_l/step_$s"]))
end


t_steps = collect(2:100)*0.05;
op_size_n10 = plot(t_steps, op_size(rho_a_1), label = L"\textrm{Ref:} \alpha = 1");
plot!(t_steps, op_size(rho_a_1_fast), label = L"\textrm{New:} \alpha =1.0")
xlabel!(L"tJ");
ylabel!(L"L[\mathcal{W}(t)]");
plot!(thickness_scaling=1.25, legend = :topleft)
savelatexfig(op_size_n10, plotsdir("WII/op_dens/op_size_alpha_1.0_Dmax=500_tol_1e-6"))




## Compare against ED results

fo = h5open(datadir("op_density/op_dens_mfi_n10_a2.5.hdf5"));
rho_ED = read(fo["op_dens_Y"]);
Y_size_ED = abs.(op_size(rho_ED))
t_ed = read(fo["time"]);

op_size_n10 = plot(t_steps, op_size(rho_a_2_5), label = L"\alpha =2.5 \quad (\textrm{MPO}, \epsilon = 10^{-6})");
plot!(t_ed[1:50], Y_size_ED[1:50], label = L"\alpha = 2.5 \quad (\textrm{ED})");
xlabel!(L"tJ");
ylabel!(L"L[\mathcal{W}(t)]");
plot!(thickness_scaling=1.25, legend = :topleft)
savelatexfig(op_size_n10, plotsdir("WII/op_dens/op_size_alpha_2.5_Dmax=500_vs_ED"))
