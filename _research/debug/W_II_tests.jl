using Revise
using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
#using LinearAlgebra
using SparseArrays
using TensorOperations

#using mps_compressor
using dmrg_methods
using operators_basis

include(srcdir("hamiltonians.jl"));

⊗ = kron


##### Parameters #####
######################

alpha = 2.5;
J = -1.0;
Bx = 1.05;
Bz = -0.5;
Ni = Nj = 5;
N = 10;
dt = 0.01;
t = 1.0;
M = t/dt;

##### W_II propagator #####
###########################

W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj; kac_norm=false);
W_II_dag = WII(alpha, N, J, Bx, Bz, -dt, Ni, Nj);

Wi = reshape(W_II.Wi, 2, 6, 2, 6);
W1 = reshape(W_II.W1, 2, 2, 6);
WN = reshape(W_II.WN, 2, 6, 2);

Wi_dag = reshape(W_II_dag.Wi, 2, 6, 2, 6);
W1_dag = reshape(W_II_dag.W1, 2, 2, 6);
WN_dag = reshape(W_II_dag.WN, 2, 6, 2);



## Calculate tr(U*U†)
@tensor tr_norm[akn, abn] := W1[x, y, akn]*W1_dag[y, x, abn]
@tensor begin
    for i ∈ 1:8
        tr_norm[akn, abn] := tr_norm[ak, ab]*Wi[x, ak, y, akn]*Wi_dag[y, ab, x, abn]
    end
end
@tensor norm_mpo = tr_norm[ak, ab]*WN[x, ak, y]*WN_dag[y, ab, x];
norm_mpo/2^10

## Calculate matrix U(dt)
Ui = reshape(W_II.W1, 2, 2, 6);
for i ∈ 1:N-2
    @tensor Ui[ui, un, di, dn, ai] := Ui[ui, di, x]*W_II.Wi[un, x, dn, ai]
    Ui = reshape(Ui, 2^(i+1), 2^(i+1), 6)
end 
@tensor U[ui, un, di, dn] := Ui[ui, di, x]*reshape(W_II.WN, 2, 6, 2)[un, x, dn];

U_WII = reshape(U, 2^N, 2^N);

tr(U_WII'*U_WII)

## Calculate matrix U(-dt)
Ui_dag = reshape(W_II_dag.W1, 2, 2, 6);
for i ∈ 1:N-2
    @tensor Ui_dag[ui, un, di, dn, ai] := Ui_dag[ui, di, x]*W_II_dag.Wi[un, x, dn, ai]
    Ui_dag = reshape(Ui_dag, 2^(i+1), 2^(i+1), 6)
end 
@tensor U_dag[ui, un, di, dn] := Ui_dag[ui, di, x]*reshape(W_II.WN, 2, 6, 2)[un, x, dn];

U_WII_dag = reshape(U_dag, 2^N, 2^N);

## Calculate tr(U(dt)U(-dt))
tr(U_WII*U_WII_dag)/2^N

##### Exact propagator #####
############################

function JPL(α, L)
    J = zeros(L,L);
    for i ∈ 1:L, j ∈ i+1:L
        J[i,j] = abs(i-j)^(-α)
    end
    kacn = sum(J)/(L-1)
    J += transpose(J)
    #return J/kacn
    return J
end

Jij = 4*J*JPL(2.5, N);
Bxe = 2*Bx;
Bze = 2*Bz;

TF = TransverseFieldIsing(Jij,[Bxe]);
TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
Hlong = Bze*spdiagm(sum([diag(Sᶻᵢ(i,N)) for i in 1:N]));
H_TMI = TF_M + Hlong;

# tr(U_exact*U_WII_dag)/2^N
# tr(U_exact*adjoint(U_WII))/2^N

##### Action on |→→→..→⟩ #####
##############################

psi_plus = 1/sqrt(2)*[im, 1];
psi_0 = ⊗([psi_plus for i ∈ 1:N]...);
psi_0_mps = MPS([reshape(psi_plus, 1, 2, 1) for _ ∈ 1:N]);


# Exact t=0.4
U_exact = exp(-im*4*dt*collect(H_TMI));
psi_exact = U_exact*psi_0;

# W_II t=0.4

function calc_psit(psi_initial::MPS, U, steps::Int)
    psi_dt = psi_initial;
    for m ∈ 1:steps
        psi_dt = mpo_mps_product(psi_dt, U);
    end
    normalize!(psi_dt)
    return psi_dt  
end

psi_dt_mps = calc_psit(psi_0_mps, W_II, 4);

psi_exact_mps = vector_to_mps(psi_exact);

overlap(psi_exact_mps, psi_dt_mps) #! 0.9999

# Left canonize
psi_dt_lc = MPS(copy(psi_dt_mps.Ai));
canonize!(psi_dt_lc);

# Expectation values
𝟙 = [1 0; 0 1]
X = [0 1; 1 0]
Z = [1 0; 0 -1]
Y = [0 -im; im 0]

#psi_dt_mc =  MPS(copy(psi_dt_lc.Ai)); 

exp_x = zeros(2,10);
exp_y = zeros(2,10);
exp_z = zeros(2,10);
for n = 9:-1:2
    psi_dt_mc =  MPS(copy(psi_dt_lc.Ai)); 
    n != N && canonize!(psi_dt_mc; final_site=n, direction = "right");
    exp_x[1,n] = abs(calc_expval(psi_dt_mc, X, n; mixed_canonical=true))
    exp_y[1,n] = abs(calc_expval(psi_dt_mc, Y, n; mixed_canonical=true))
    exp_z[1,n] = abs(calc_expval(psi_dt_mc, Z, n; mixed_canonical=true))
    exp_x[2,n] = abs(psi_exact'*σˣᵢ(n,10)*psi_exact)
    exp_y[2,n] = abs(psi_exact'*σʸᵢ(n,10)*psi_exact)
    exp_z[2,n] = abs(psi_exact'*σᶻᵢ(n,10)*psi_exact)
end

println("σₓ expectaction values")
display(exp_x[1,:] - exp_x[2,:])

println("σʸ expectaction values")
display(exp_y[1,:] - exp_y[2,:])

println("σᶻ expectaction values")
display(exp_z[1,:] - exp_z[2,:])

##### Compression of U(t=0.04) #####
####################################
using IterativeSolvers: cg, cg!, gmres, gmres!


W_3dt = calc_Ut(W_II, 0.03);
W_4dt = calc_Ut(W_II, 0.04);

U3dt_mps =  cast_mps(W_3dt; L=10, normalize = true);
U4dt_mps =  cast_mps(W_4dt; L=10, normalize = true);

solver_parms = Dict(
    :fav_solver=>cg!,
    :alt_solver=>gmres!,
    :abstol=>Real(1e-5),
    :reltol=>Real(1e-7),
    :verb_compressor=>0,
    :maxiter=>Int(216*5),
    :log=>true,
    :max_solver_runs=>20
)

U4dt_comp =  mps_compress_cg(U4dt_mps, U3dt_mps, 1e-6; solver_parms...); #! if the MPO is translation invariant, can I just simply optimize the central tensors?
U4dt =  cast_mpo(U4dt_mps);
U4dt_comp =  cast_mpo(U4dt_comp);


##### Comparison compressed, uncompressed and exact U(dt=0.04) #####
#####################################################

psi_t = calc_psit(psi_0_mps, U4dt, 1);
psi_t_comp = calc_psit(psi_0_mps, U4dt_comp, 1);

# Left canonize
canonize!(psi_t);
canonize!(psi_t_comp);

# Expectation values
𝟙 = [1 0; 0 1]
X = [0 1; 1 0]
Z = [1 0; 0 -1]
Y = [0 -im; im 0]


overlap(psi_t_comp, psi_t) #! 0.999999

exp_x = zeros(3,10);
exp_y = zeros(3,10);
exp_z = zeros(3,10);
for n = 9:-1:2
    psi_copy =  MPS(copy(psi_t.Ai)); 
    psi_comp_copy =  MPS(copy(psi_t_comp.Ai)); 
    canonize!(psi_copy; final_site=n, direction = "right");
    canonize!(psi_comp_copy; final_site=n, direction = "right");
    
    exp_x[1,n] = abs(calc_expval(psi_copy, X, n; mixed_canonical=true))
    exp_y[1,n] = abs(calc_expval(psi_copy, Y, n; mixed_canonical=true))
    exp_z[1,n] = abs(calc_expval(psi_copy, Z, n; mixed_canonical=true))
    
    exp_x[2,n] = abs(calc_expval(psi_comp_copy, X, n; mixed_canonical=true))
    exp_y[2,n] = abs(calc_expval(psi_comp_copy, Y, n; mixed_canonical=true))
    exp_z[2,n] = abs(calc_expval(psi_comp_copy, Z, n; mixed_canonical=true))

    exp_x[3,n] = abs(psi_exact'*σˣᵢ(n,10)*psi_exact)
    exp_y[3,n] = abs(psi_exact'*σʸᵢ(n,10)*psi_exact)
    exp_z[3,n] = abs(psi_exact'*σᶻᵢ(n,10)*psi_exact)
end

exp_x[3,:] - exp_x[2,:]
exp_y[3,:] - exp_y[2,:]
exp_z[3,:] - exp_z[2,:]


##### Compress general U(t) #####
#################################

tf = 1.0;
M = Int(tf / dt);

cg_params = Dict(
    :fav_solver=>gmres!,
    :alt_solver=>cg!,
    :abstol=>5e-7,
    :reltol=>1e-7,
    :tol_compr=>1e-6,
    :verb_compressor=>0,
    :maxiter=>Int(216*10), #* Increase for higher tolerances
    :log=>true,
    :max_solver_runs=>15 #* Increase for higher tolerances
)

W_3dt = calc_Ut(W_II, 0.03);
Dmax = 400;
W1_0 = rand(ComplexF64, (2, 1, 2, Dmax));
Wi_0 = rand(ComplexF64, (2, Dmax, 2, Dmax));
WN_0 = rand(ComplexF64, (2, Dmax, 2, 1));


U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));
U_mdt = MPO(copy([W_3dt.Wi[1], fill(W_3dt.Wi[2], N-2)..., W_3dt.Wi[3]]));
U_seed = MPO([W1_0, fill(Wi_0, N-2)..., WN_0]);

for m ∈ 4:M
    printstyled("\n##### Calculating U($(m)dt) ##### \n")
    U_mdt = prod(U_mdt, U_dt; compress = true, seed = U_mdt, cg_params...)
    #U_seed = U_mdt
end



##### Operator density #####
############################

########## Test projection
# e.g. 𝟙^2(X+Y+Z)(X+Y+Z+1)^2

𝟙 = [1 0; 0 1];     
ℤ = [1 0; 0 -1];    
𝕐	= im*[0 -1; 1 0];   
𝕏 = [0 1; 1 0];
P = [[𝟙] [𝟙] [(𝕏+𝕐+ℤ)] [(𝕏+𝕐+ℤ+𝟙)] [(𝕏+𝕐+ℤ+𝟙)] [(𝕏+𝕐+ℤ+𝟙)]]


W1 = collect(reshape(W_2_5.W1, (2,2,Ni+1)))
Wi = collect(reshape(W_2_5.Wi, (2, Ni+1, 2, Ni+1)))
WN = collect(reshape(W_2_5.WN, (2, Ni+1, 2)))

@tensor PW1[a] := W1[x,y,a]*P[1][x,y]

@tensor PW2[a,b] := Wi[x,a,y,b]*P[2][x,y]
@tensor PW3[a,b] := Wi[x,a,y,b]*P[3][x,y]
@tensor PW4[a,b] := Wi[x,a,y,b]*P[4][x,y]
@tensor PW5[a,b] := Wi[x,a,y,b]*P[5][x,y]

@tensor PW6[a] := WN[x,a,y]*P[6][x,y]

@tensor PH = PW1[a]*PW2[a,b]*PW3[b,c]*PW4[c,d]*PW5[d,e]*PW6[e]




########## Operator density

U_t_mpo = [];
@inbounds for s ∈ 4:100
    address = obelix_folder*"U_alpha=$(alpha)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_r1_step=$(s).h5"
    push!(U_t_mpo, rebuild_Ut(address));
end

includet(srcdir("observables.jl"))

D = 10;
N = 10;
W1 = rand(2, 1, 2, D);
WN = rand(2, D, 2, 1);
Wi = rand(2, D, 2, D);

U_dt = MPO([W1, fill(Wi, N-2)..., WN]);
X = [0 1; 1 0];

operator_density_trace(U_dt, X, 1, 3)








