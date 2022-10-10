########## TODO
# - write a new structure that calculates and fills Wi matrices upon creation
########## 

@variables 𝕏 𝕐 ℤ 𝟙;

########## Hack to allow evaluating complex expression with complex variables
# See: https://github.com/JuliaSymbolics/Symbolics.jl/issues/534
Base.Complex{T}(r::Number, i::Complex) where {T<:Real} = r + im * i
Base.Complex{T}(r::Complex, i::Number) where {T<:Real} = r + im * i
Base.Complex{T}(r::Complex, i::Complex) where {T<:Real} = r + im * i
########## 

"""
    W_II(α::Float64,J::Float64,Bx::Float64,Bz::Float64,dt::Float64,Ni::Int)

Type for ``W_{II}`` representation of time-evolution operator.
Loads the coefficients of an order `Ni` approximation of the power-law Ising couplings.
Creates the ``A``, ``B``, ``C`` and ``D`` operator valued matrices of the MPO representation of ``H = J\\sum_{i,j}|r_i-r_j|^{-α}ℤ_i ℤ_j + \\sum_i B_x 𝕏_i + B_z ℤ_i``

### Arguments
- `α` : power-law exponent
- `J` : Ising strength
- `Bx, Bz` : field strengths
- `dt` : time-step
- `Ni` : MPO bond dimension
    
"""
mutable struct WII
    J::Float64;
    Bx::Float64;
    Bz::Float64;
    dt::Float64;
    
    alpha::Float64;
    betas::Array{Float64};
    lambdas::Array{Float64};
    Ni::Int64;
    Nj::Int64;
    
    A::Array{Num, 2};
    B::Array{Num, 1};
    C::Array{Num, 1};
    D::Num;

    WA::Any;
    WB::Any;
    WC::Any;
    WD::Any;
    Wi::Any;
    W1::Any;
    WN::Any;

    
    function WII(α::Float64,J::Float64,Bx::Float64,Bz::Float64,dt::Float64,Ni::Int)
        @variables 𝕏 𝕐 ℤ 𝟙 𝟘;
        @variables sA[1:Ni,1:Ni] sB[1:Ni] sC[1:Ni] D;
        A = Symbolics.scalarize(sA); # converts into an array of symbols
        B = Symbolics.scalarize(sB);
        C = Symbolics.scalarize(sC);

        pl_MPO = h5open(datadir("pl_Ham_MPO_Mmax=$(Ni).h5"), "r"); # load fit data
        βs = read(pl_MPO["alpha=$(α)/betas"]);
        λs = read(pl_MPO["alpha=$(α)/lambdas"]);

        # Fills A,B,C,D operator valued matrices of Hamiltonian
        A .=  𝟘;
        for m in 1:Ni
            A[m,m] = λs[m]*𝟙;
            B[m] = λs[m]*ℤ;
            C[m] = J*βs[m]*ℤ;
        end
        D = Bz*ℤ + Bx*𝕏;

        new(J, Bx, Bz, dt, α, βs, λs, Ni, Ni, A, B, C, D); # set fields as order inside struct
    end
end

"""
    function calc_WII_series!(oWII::WII; [order]::Int=20)

Calculates the ``W_{II}`` representation of the time-evolution operator ``\\hat{U}=exp(-i dt \\hat{H})``. Uses a Taylor expansion to approximate ``\\hat{U}=exp(-i dt \\hat{H})``

### Arguments

- `oWII` : an instance of the type [`WII`](@ref)
- `order` : order of Taylor expansion

### Returns

- Sets the fields ``W_A``, ``W_B``, ``W_C``, ``W_D`` of the `oWII` input instance

"""
function calc_WII_series!(oWII::WII; order=20)

    #println("New update 2")

    @variables F Φ 𝕏 𝕐 ℤ 𝟙 dτ 𝕚 # because of limitations to handle complex substitutions in Symbolics, need to define 𝕚
    Ni = oWII.Ni; Nj = oWII.Nj;
    WA = zeros(ComplexF64, 2Ni, 2Nj);
    WB = zeros(ComplexF64, 2Ni, 2);
    WC = zeros(ComplexF64, 2, 2Nj);
    
    pauli_M = Dict(𝟙 => [1 0; 0 1], ℤ => [1 0; 0 -1], 𝕐 => 𝕚*[0 -1; 1 0], 𝕏 => [0 1; 1 0]);
    toArray(sA) = Symbolics.value.(Symbolics.scalarize(substitute(substitute(sA, pauli_M),𝕚=>im))); # Converts the operator valued matrix element into is numerical values

    dτ = -𝕚*oWII.dt;
    A = oWII.A; B = oWII.B; C = oWII.C; D = oWII.D;
    for aj ∈ 1:oWII.Ni, aj_c ∈ 1:oWII.Nj
        
        # Calc exp of matrix elements of F symbolically
        F = [
            dτ*D       0         0     0;
            √dτ*C[aj_c]  dτ*D       0     0;
            √dτ*B[aj]   0        dτ*D   0;
            A[aj,aj_c]  √dτ*B[aj]  √dτ*C[aj_c] dτ*D;
            ]; #! A[i,j] = 0 if i ≠ j?
        
        Φ = sum([F^n/factorial(n) for n ∈ 1:order]);
        
        # Fill in matrix elements of WII with numerical values
        WA[2aj-1:2aj, 2aj_c-1:2aj_c] = toArray(Φ[4,1]);
        if aj_c == 1
            WB[2aj-1:2aj,:] = toArray(Φ[3,1]);
        end
        if aj == 1
            WC[:,2aj_c-1:2aj_c] = toArray(Φ[2,1]);
        end

    end

    WD = toArray(sum([(dτ*D)^n/factorial(n) for n ∈ 1:order]));
    
    oWII.WA = WA;    oWII.WB = WB;    oWII.WC = WC;    oWII.WD = WD;    
    oWII.Wi = vcat(hcat(WD,WC), hcat(WB, WA));    oWII.W1 = vcat(WD, WB);    oWII.WN = hcat(WD, WC);
end

"""
    function calc_WII!(oWII::WII)

Calculates the ``W_{II}`` representation of the time-evolution operator ``\\hat{U}=exp(-i dt \\hat{H}) ``

### Arguments

- `oWII` : an instance of the type [`WII`](@ref)

### Returns

- Sets the fields ``W_A``, ``W_B``, ``W_C``, ``W_D`` of the `oWII` input instance

"""
function calc_WII!(oWII::WII)

    @variables 𝕏 𝕐 ℤ 𝟙 𝟘
    Ni = oWII.Ni; Nj = oWII.Nj;
    WA = zeros(ComplexF64, 2Ni, 2Nj);
    WB = zeros(ComplexF64, 2Ni, 2);
    WC = zeros(ComplexF64, 2, 2Nj);
    
    # rules to replace symbols in A, B, C, D moperator valued matrices
    pauli_M = Dict(𝟙 => [1 0; 0 1], ℤ => [1 0; 0 -1], 𝕐 => im*[0 -1; 1 0], 𝕏 => [0 1; 1 0], 𝟘 => [0 0; 0 0]);
    #toArray(sA) = Symbolics.value.(Symbolics.scalarize(substitute(substitute(sA, pauli_M),𝕚=>im))); # Converts the operator valued matrix element into is numerical values
    toArray(sA) = Symbolics.value.(Symbolics.scalarize(substitute(sA, pauli_M)));

    dτ = -im*oWII.dt;
    A = toArray.(oWII.A); B = toArray.(oWII.B); C = toArray.(oWII.C); D = toArray.(oWII.D); 𝟘 = zeros(2,2); #! check if broadcasting works.
    
    @inbounds for aj ∈ 1:Ni, aj_c ∈ 1:Nj
        
        # Calc exp of matrix elements of F
        F = [
            dτ*D          𝟘         𝟘          𝟘;
            √dτ*C[aj_c]  dτ*D       𝟘          𝟘;
            √dτ*B[aj]     𝟘        dτ*D        𝟘;
            A[aj,aj_c]  √dτ*B[aj]  √dτ*C[aj_c] dτ*D;
            ]; #! A[i,j] = 0 if i ≠ j?
        
        Φ = exp(F);
        
        # Fill in matrix elements of WII
        WA[2aj-1:2aj, 2aj_c-1:2aj_c] = Φ[7:8,1:2];
        
        if aj_c == 1
            WB[2aj-1:2aj,:] = Φ[5:6,1:2];
        end
        
        if aj == 1
            WC[:,2aj_c-1:2aj_c] = Φ[3:4,1:2];
        end

    end

    WD = exp(dτ*D)
    
    oWII.WA = WA;    oWII.WB = WB;    oWII.WC = WC;    oWII.WD = WD;    
    oWII.Wi = vcat(hcat(WD,WC), hcat(WB, WA));    oWII.W1 = vcat(WD, WB);    oWII.WN = hcat(WD, WC);
end


"""
    function calc_WII_symb!(WII::WII; [order]::Int=20)

Calculates the ``W_{II}`` representation of the time-evolution operator ``\\hat{U}=exp(-i dt \\hat{H}) ``

### Arguments

- `oWII` : an instance of the type `WII`

### Returns

- The operator valued operators WA, WB, WC, WD with symbolic representation of Pauli matrices
"""
function calc_WII_symb!(oWII::WII; order=20) # Returns symbolic versions of WA, WB, WC, WD
    Ni = oWII.Ni; Nj = oWII.Nj;
    @variables F Φ 𝕏 𝕐 ℤ 𝟙 aWA[1:Ni,1:Nj] aWB[1:Ni] aWC[1:Nj] WD
    WA = Symbolics.scalarize(aWA); # converts into an array of symbols
    WB = Symbolics.scalarize(aWB);
    WC = Symbolics.scalarize(aWC);
    
    dt = -1im*oWII.dt;
    A = oWII.A; B = oWII.B; C = oWII.C; D = oWII.D;
    for aj ∈ 1:Ni, aj_c ∈ 1:Nj
        
        # Calc exp of matrix elements of F
        F = [
            dt*D       0         0     0;
            √dt*C[aj_c]  dt*D       0     0;
            √dt*B[aj]   0        dt*D   0;
            A[aj,aj_c]  √dt*B[aj]  √dt*C[aj_c] dt*D;
            ]; #! A[i,j] = 0 if i ≠ j?
        
        Φ = sum([F^n/factorial(n) for n ∈ 1:order]);
        
        # Fill in matrix elements of WII
        WA[aj, aj_c] = Φ[4,1];
        if aj_c == 1
            WB[aj] = Φ[3,1];
        end
        if aj == 1
            WC[aj_c] = Φ[2,1];
        end

    end

    WD = sum([(dt*D)^n/factorial(n) for n ∈ 1:order]);
    oWII.WA = WA;    oWII.WB = WB;    oWII.WC = WC;    oWII.WD = WD;
    oWII.Wi = vcat(hcat(WD,WC'), hcat(WB, WA));
    oWII.W1 = vcat(WD, WB);    
    oWII.WN = hcat(WD, WC');
end

"""
    function calc_WII(Ni::Int, Nj::Int, A, B, C, D, dt; [order]=20)

Calculates the ``W_{II}`` representation of the time-evolution operator ``\\hat{U}=exp(-i dt \\hat{H}) ``

### Arguments

- `Ni, Nj` : left and right auxiliary bond dimensions of MPO
- `A, B, C, D` : operator valued matrices of parent Hamiltonian
- `dt` : time-step

### Returns

- The operator valued operators WA, WB, WC, WD

"""
function calc_WII(Ni::Int, Nj::Int, A, B, C, D, dt; order=20)
    @variables F Φ 𝕏 𝕐 ℤ 𝟙
    WA = spzeros(2Ni, 2Nj);
    WB = spzeros(2Ni,2);
    WC = spzeros(2,2Nj);

    pauli_M = Dict(𝟙 => [1 0; 0 1], ℤ => [1 0; 0 -1], 𝕐 => im*[0 -1; 1 0], 𝕏 => [0 1; 1 0]);
    toArray(sA) = Symbolics.value.(Symbolics.scalarize(substitute(sA, pauli_M))); # Converts the operator valued matrix element into is numerical values


    for aj ∈ 1:Ni, aj_c ∈ 1:Nj
        
        # Calc exp of matrix elements of F symbolically
        F = [
            dt*D       0         0     0;
            √dt*C[aj_c]  dt*D       0     0;
            √dt*B[aj]   0        dt*D   0;
            A[aj,aj_c]  √dt*B[aj]  √dt*C[aj_c] dt*D;
            ]; #! A[i,j] = 0 if i ≠ j?
        
        Φ = sum([F^n/factorial(n) for n ∈ 1:order]);
        
        # Fill in matrix elements of WII with numerical values
        WA[2aj-1:2aj, 2aj_c-1:2aj_c] = toArray(Φ[4,1]);
        if aj_c == 1
            WB[2aj-1:2aj,:] = toArray(Φ[3,1]);
        end
        if aj == 1
            WC[:,2aj_c-1:2aj_c] = toArray(Φ[2,1]);
        end

    end

    WD = toArray(sum([(dt*D)^n/factorial(n) for n ∈ 1:order]));

    return WA, WB, WC, WD
end

"""
    function calc_Ut(oWII::WII, t::Float64)

Calculates the time-evolution unitary ``U(t)``  by contracting MPO layers of the ``W_{II}`` representation of ``U(dt)``

### Arguments
- `oWII` : an instance of the type [`WII`](@ref)
- `t` : final time

### Returns 
- ``U(t)`` as an instance of the `MPO` type

"""
function calc_Ut(oWII::WII, t::Float64)
    dt = oWII.dt;
    M = Int(ceil(t/dt));
    Ni = oWII.Ni;

    W1 = collect(reshape(oWII.W1, (2, 2, Ni+1))); # phys up, phys down, aux right
    Wi = collect(reshape(oWII.Wi, (2, Ni+1, 2, Ni+1))); # phys up, aux left, phys down, aux right
    WN = collect(reshape(oWII.WN, (2, Ni+1, 2))); # phys up, aux left, phys down

    Idt = zeros(2,1,2,1);    
    Idt[:,1,:,1] = [1 0; 0 1];
    W1m = Idt[:,1,:,:]; 
    Wim = Idt; 
    WNm = Idt[:,:,:,1];

    for m ∈ 1:M
        @tensor begin
            W1m[a,b,c,d] := W1m[a,x,c]*W1[x,b,d]; # phys up, phys down, aux left N-1, aux left 1
            Wim[a,c,d,b,e,f] := Wim[a,c,x,e]*Wi[x,d,b,f]; # phys up, aux left N-1, aux left 1, phys down, aux rigt N-1, aux right 1
            WNm[a,c,d,b] := WNm[a,c,x]*WN[x,d,b]; # phys up, aux left N-1, aux left 1, phys down
        end
        W1m = reshape(W1m, (2,2,(Ni+1)^m));
        Wim = reshape(Wim, (2,(Ni+1)^m,2,(Ni+1)^m));
        WNm = reshape(WNm, (2,(Ni+1)^m,2));
    end

    return MPO([W1m, Wim, WNm])
end



#==========================================================================================#
########## OLD: keep for debugging

#= 
"""
Calculates the matrix element ``F_{j;a_j,\\bar{a}_j}`` where ``a_j = {1,...,N_j}`` labels the bosonic auxiliary fields (``\\bar{a}_j`` the complex conjugate of the fields) and ``N_j`` is the (non-trivial) bond dimension of the MPO tensor at each site ``j``


# Arguments
- `i,j` : indices corresponding to ``a_j`` and ``\\bar{a}_j``
- `A,B,C,D` : operator valued matrices of Hamiltonian

# Returns
- A ``4d`` matrix (where ``d`` is the physical site dimension)

"""
function F_ij(i, j, A, B, C, D, dt) #! if i ≠ j, then Aᵢⱼ = 0??
    @variables M
    M = [
        dt*D       0         0     0;
        √dt*C[j]  dt*D       0     0;
        √dt*B[i]   0        dt*D   0;
        A[i,j]  √dt*B[i]  √dt*C[j] dt*D;
        ]; 
    
    return M
end


mF = F_ij(1, 1, A, B, C, D, 0.05)

########## WII matrix elements


function exp_F(F; order=20) #! largerst factorial supported without bignumber
    @variables Φ
    Φ = sum([F^n/factorial(n) for n ∈ 1:order])
    return Φ
end


function exp_F(i, j, A, B, C, D, dt; order=20) #! largest factorial supported without bignumber
    @variables F Φ
    F = [
        dt*D       0         0     0;
        √dt*C[j]  dt*D       0     0;
        √dt*B[i]   0        dt*D   0;
        A[i,j]  √dt*B[i]  √dt*C[j] dt*D;
        ]; 
    Φ = sum([F^n/factorial(n) for n ∈ 1:order])
    return Φ
end

 =#