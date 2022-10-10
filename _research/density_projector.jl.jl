using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using LinearAlgebra
using Symbolics
using TensorOperations

𝟙 = [1 0; 0 1];
ℤ = [1 0; 0 -1];
𝕐	= im*[0 -1; 1 0];
𝕏 = [0 1; 1 0];

⊗ = kron


########## Case: H = XXII + IXXI + IIXX

𝕎 = zeros(2*3,2*3);
𝕎[1:2,1:2] = 𝟙;
𝕎[end-1:end,end-1:end] = 𝟙;
𝕎[1:2,3:4] = 𝕏;
𝕎[3:4,end-1:end] = 𝕏;

𝕎0 = [𝟙 𝕏 zeros(2,2)]
𝕎f = [zeros(2,2); 𝕏; 𝟙]

𝕎0r = reshape(𝕎0, (2,2,3));
𝕎fr = reshape(𝕎f, (2,3,2));
𝕎r = reshape(𝕎,(2,3,2,3));

##### Projector: IIXX

@tensor PW1[a] := 𝕎0r[x,y,a]*𝟙[x,y]
@tensor PW2[a,b] := 𝕎r[x,a,y,b]*𝟙[x,y]
@tensor PW3[a,b] := 𝕎r[x,a,y,b]*𝕏[x,y]
@tensor PW4[a] := 𝕎fr[x,a,y]*𝕏[x,y]
@tensor PH = PW1[a]*PW2[a,b]*PW3[b,c]*PW4[c]


########## Case: H = XXII + IXXI + IIXX + ZZII + IZZI + IIZZ


𝕎0 = [𝟙 𝕏 ℤ zeros(2,2)]
𝕎f = [zeros(2,2); 𝕏; ℤ; 𝟙]
𝕎 = zeros(2*4,2*4);
𝕎[1:2,:] = 𝕎0
𝕎[:,end-1:end] = 𝕎f

𝕎0r = reshape(𝕎0, (2,2,4));
𝕎fr = reshape(𝕎f, (2,4,2));
𝕎r = reshape(𝕎,(2,4,2,4));

##### Projector: IIXX

@tensor PW1[a] := 𝕎0r[x,y,a]*𝟙[x,y]
@tensor PW2[a,b] := 𝕎r[x,a,y,b]*𝟙[x,y]
@tensor PW3[a,b] := 𝕎r[x,a,y,b]*𝕏[x,y]
@tensor PW4[a] := 𝕎fr[x,a,y]*𝕏[x,y]
@tensor PH = PW1[a]*PW2[a,b]*PW3[b,c]*PW4[c]



##### Projector: IIZZ

@tensor PW1[a] := 𝕎0r[x,y,a]*𝟙[x,y]
@tensor PW2[a,b] := 𝕎r[x,a,y,b]*𝟙[x,y]
@tensor PW3[a,b] := 𝕎r[x,a,y,b]*ℤ[x,y]
@tensor PW4[a] := 𝕎fr[x,a,y]*ℤ[x,y]
@tensor PH = PW1[a]*PW2[a,b]*PW3[b,c]*PW4[c]


##### Projector: IIZZ + IIXX

@tensor PW1[a] := 𝕎0r[x,y,a]*𝟙[x,y]
@tensor PW2[a,b] := 𝕎r[x,a,y,b]*𝟙[x,y]
@tensor PW3[a,b] := 𝕎r[x,a,y,b]*(ℤ[x,y]+ 𝕏[x,y])
@tensor PW4[a] := 𝕎fr[x,a,y]*(ℤ[x,y]+ 𝕏[x,y])
@tensor PH = PW1[a]*PW2[a,b]*PW3[b,c]*PW4[c]



########## Case: H = YYII + IYYI + IIYY + ZZII + IZZI + IIZZ

𝟘 = zeros(ComplexF64, 2, 2);
𝕀 = (𝕏+𝕐+ℤ+𝟙); # operator space identity
𝕏𝕐ℤ = 𝕏+𝕐+ℤ; 

𝕎0 = [𝟙 𝕐 ℤ 𝟘]
𝕎f = [𝟘; 𝕐; ℤ; 𝟙]
𝕎 = zeros(ComplexF64, 2*4,2*4);
𝕎[1:2,:] = 𝕎0
𝕎[:,end-1:end] = 𝕎f

𝕎0r = reshape(𝕎0, (2,2,4));
𝕎fr = reshape(𝕎f, (2,4,2));
𝕎r = reshape(𝕎,(2,4,2,4));

##### Projector: IIAA

@tensor PW1[a] := 𝕎0r[x,y,a]*𝟙[y,x]
@tensor PW2[a,b] := 𝕎r[x,a,y,b]*𝟙[y,x]
@tensor PW3[a,b] := 𝕎r[x,a,y,b]*𝕏𝕐ℤ[y,x]
@tensor PW4[a] := 𝕎fr[x,a,y]*𝕀[y,x]
@tensor PH = PW1[a]*PW2[a,b]*PW3[b,c]*PW4[c]





########## Symbolic construction

@variables 𝕎1 𝕎2 𝕎0 𝕎f
@variables 𝕏1 𝕐1 ℤ1 𝟙1 𝕏2 𝕐2 ℤ2 𝟙2 𝕏3 𝕐3 ℤ3 𝟙3 𝕏4 𝕐4 ℤ4 𝟙4

𝕎1 = [𝟙2 𝕏2 ℤ2 0;
      0 0 0 𝕏2;
      0 0 0 ℤ2;
      0 0 0 𝟙2];

𝕎2 = [𝟙3 𝕏3 ℤ3 0;
      0 0 0 𝕏3;
      0 0 0 ℤ3;
      0 0 0 𝟙3];


𝕎0 = [𝟙1 𝕏1 ℤ1 0];
𝕎f = [0; 𝕏4; ℤ4; 𝟙4];

