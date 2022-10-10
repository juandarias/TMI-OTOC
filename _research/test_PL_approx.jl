
Wi_dag = reshape(W_II_dag.Wi, 2, 6, 2, 6);
W1_dag = reshape(W_II_dag.W1, 2, 2, 6);
WN_dag = reshape(W_II_dag.WN, 2, 6, 2);

@tensor Ei[u, l1, l2, d, r1, r2] := Wi[u, l1, x, r1]*Wi_dag[x, l2, d, r2]
Ei = reshape(Ei, 2, 36, 2, 36)

for i ∈ 1:36
    display(sparse(round.(real.(Ei[:,i,:,i]); digits=4)))
end

C_1 = W_II.C[1]
D = W_II.D

dtau = -im*dt;

cdag = [0 0; 1 0];
c = [0 1; 0 0];

⊗(𝟙, c, C_1)

basis0 =  ⊗([1,0], [0,1], [1,0])
basis1 =  ⊗([1,0], [0,1], [0,1])
gs_0 = insert!(zeros(7),1,1)
gs_1 = insert!(zeros(7),2,1)


W_C_1 = exp(√dtau*⊗(𝟙, cdag, C_1) + dtau*⊗(𝟙, 𝟙, D))

adjoint(basis0)*(W_C_1*gs_0) #! Construction is oke

W_II.WC[1:2,1:2]



############################## Power-law Hamiltonian MPO ##############################

𝟙 = [1 0; 0 1]; ℤ = [1 0; 0 -1]; 𝕏 = [0 1; 1 0]; 𝟘 = [0 0; 0 0];
H_Wi = zeros(2^N, 2^N);
A = zeros(2*Ni, 2*Ni);
B = zeros(2*Ni, 2);
C = zeros(2, 2*Ni);

for i ∈ 1:Ni, j ∈ 1:Ni
    A[2*i-1:2i,2*j-1:2j] = W_II.A[i,j]
    B[2*i-1:2*i,:] = W_II.B[i]
    C[:,2*i-1:2*i] = W_II.C[i]
end
D = W_II.D #! set to zero for debugging
D = 𝟘 #! set to zero for debugging

H_Wi = reshape(vcat(hcat(𝟙, C, D), hcat(zeros(2*Ni, 2), A, B), hcat(𝟘, zeros(2, 2*Ni), 𝟙)), (2, 7, 2, 7));
H_W1 = reshape(hcat(𝟙, C, D), (2, 2, 7));
H_WN = reshape(vcat(D, B , 𝟙), (2, 7, 2));

Hi = H_W1;
@tensor begin
    for i in 1:8
        Hi[u1, u2, d1, d2, a] := Hi[u1, d1, x]*H_Wi[u2, x, d2, a]
        Hi = reshape(Hi, 2^(i+1), 2^(i+1), 7)
    end
end
@tensor HPL[u1, u2, d1, d2] := Hi[u1, d1, x]*H_WN[u2, x, d2];
H_app = reshape(HPL, 2^10, 2^10);



Jij = 4*J*JPL(2.5, N);
Bxe = 0.0; #2*Bx;
Bze = 0.0; #2*Bz;

TF = TransverseFieldIsing(Jij,[Bxe], term="I");
TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);


diff = H_app-TF_M #! diagonal is correct
diag(diff)


############################## Simple NN Ising Hamiltonian ##############################

    Wi = reshape(vcat(hcat(𝟙, ℤ, -𝕏), hcat(𝟘, 𝟘, ℤ), hcat(𝟘, 𝟘, 𝟙)), (2, 3, 2, 3));
    W1 = reshape(hcat(𝟙, ℤ, -𝕏), 2, 2, 3)
    WN = reshape(vcat(-𝕏, ℤ, 𝟙), 2, 3, 2)

    @tensor H[u1,u2,u3,u4,d1,d2,d3,d4] := W1[u1,d1,x1]*Wi[u2,x1,d2,x2]*Wi[u3,x2,d3,x3]*WN[u4,x3,d4]

    H_app = reshape(H, 16,16)
    NN = [(i+1 == j ? 4.0 : 0.0) for i ∈ 1:4, j ∈ 1:4]
    NN += transpose(NN)

    TF = TransverseFieldIsing(Array(NN),[-2.0]);
    TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals)

    sparse(H_app - TF_M) #! they are equal

############################## Exponentially decaying Hamiltonian ##############################

    ξ = 3;
    λ = exp(-1/ξ);

    function Jij_exp(J, λ, N)
        Jij = zeros(N,N)
        for i ∈ 1:N, j ∈ i+1:N
            Jij[i, j] = J*λ^(abs(j-i))
        end
        return Jij + transpose(Jij)
    end

    Jexp = Jij_exp(4, λ, 5)
    TF = TransverseFieldIsing(Jexp, [0.0]; term="I");
    TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals)

        
    Wi = reshape(vcat(hcat(𝟙, ℤ, 𝟘), hcat(𝟘, λ*𝟙, λ*ℤ), hcat(𝟘, 𝟘, 𝟙)), (2, 3, 2, 3));
    W1 = reshape(hcat(𝟙, ℤ, 𝟘), 2, 2, 3)
    WN = reshape(vcat(𝟘, λ*ℤ, 𝟙), 2, 3, 2)

    @tensor H[u1,u2,u3,u4,u5,d1,d2,d3,d4,d5] := W1[u1,d1,x1]*Wi[u2,x1,d2,x2]*Wi[u3,x2,d3,x3]*Wi[u4,x3,d4,x4]*WN[u5,x4,d5]

    H = reshape(H, 2^5, 2^5);
    sum(diag(sparse(TF_M-H))) #! they are equal


#####
##### Exact propagator
#####

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

N = 10;
Jij = -4*J*JPL(2.5, N);
Bxe = 0.0; #2*Bx;
Bze = 0.0; #2*Bz;

TF = TransverseFieldIsing(Jij,[Bxe]);
TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
Hlong = Bze*spdiagm(sum([diag(Sᶻᵢ(i,N)) for i in 1:N]));
H_TMI = TF_M + Hlong;
U_exact = exp(-im*dt*collect(H_TMI));

for field in propertynames(W_II)[1:5]
    println(getfield(W_II, field))
end

