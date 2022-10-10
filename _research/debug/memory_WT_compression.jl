using BenchmarkTools, TimerOutputs

## Load test Wt

function load_tensors(input_file)
#    data = h5open("$(obelix_folder)/D_$(Dmax)/$(input_file).h5", "r");
    data = h5open(input_file, "r");
    Wi = Vector{Array{ComplexF64, 4}}();
    for n ∈ 1:10
        push!(Wi, read(data["Tensors/Wi_$(n)"]))
    end
    return MPO(Wi)
end

step = 10;
filename = "Wt_alpha=1.0_N=32_t=5.0_dt=0.1_tol=1.0e-6_r_id29_step=4.h5";
datafolder = "/mnt/lisa/TMI/WII/D_1000/"

W_t, loaded = load_tensors(datafolder*filename);





function load_tensors(input_file)
    try
        data = h5open(input_file, "r");
        Wi = Vector{Array{ComplexF64, 4}}();
        for n ∈ 1:N
            push!(Wi, read(data["Tensors/Wi_$(n)"]))
        end
        return MPO(Wi), true
    catch e
        return 0, false
    end
end



## Test a compression step

const J = get_param!(args_dict, "J", -1.0);
const alpha = get_param!(args_dict, "alpha", 2.5);
const Bx = get_param!(args_dict, "Bx", 1.05);
const Bz = get_param!(args_dict, "Bz", -0.5);
const N = get_param!(args_dict, "N", 10);
const Ni = get_param!(args_dict, "Ni", 5);
const Nj = get_param!(args_dict, "Nj", 5);
const dt = get_param!(args_dict, "dt", 0.05);

svd_params = Dict(:Dmax => 200, :ϵmax => 1e-7);
var_params = Dict(:METHOD => SIMPLE, :tol_compr => 1e-6, :Dmax => 300, :rate => 1.1);


W_II = WII(alpha, N, J, Bx, Bz, dt, Ni, Nj);
U_dt = MPO(copy([W_II.W1, fill(W_II.Wi, N-2)..., W_II.WN]));

## Initial W(t)
𝕐 = im*[0.0 -1.0; 1.0 0.0]
W_ti = calc_Wt(U_dt, 𝕐, 10);

WU_dt = prod(W_ti, U_dt); # W(t_i)*U(dt)
W_t = prod(conj(U_dt), WU_dt); # U(dt)†*W(t_i)*U(dt);

seed_mps = cast_mps(W_t; normalize = false); #* Seed is unnormalized
ϵ_svd =  mps_compress_svd!(seed_mps; svd_params...); #* Seed is normalized
var_params[:seed] = seed_mps; 

## Variational compression
W_t, ϵ_c = @btime mpo_compress(W_t; var_params...); # 


###
mps_mpo = cast_mps_timed(W_t; normalize = false);
reset_timer!(to)

size(mps_mpo.Ai[2])

sweep_qr_timed!(mps_mpo)

reset_timer!(to)
to


const to = TimerOutput()

function cast_mps_timed(mpo::MPO{T}; L::Int = mpo.L, normalize = false, Dmax::Int = 100) where {T}
    
    mps = MPS([zeros(T, 0,0,0) for i ∈ 1:L]); # initializes MPS of appropiate type
    
    @timeit to "copy/reshape tensors" begin
        update_tensor!(mps, reshape(permutedims(mpo.Wi[1], (2,1,3,4)), 1, 4, mpo.D[1]), 1);
        for i ∈ 2:L-1
            @timeit to "copy/reshape tensors $(i)" update_tensor!(mps, reshape(permutedims(mpo.Wi[i], (2,1,3,4)), mpo.D[i-1], 4, mpo.D[i]), i);
        end
        update_tensor!(mps, reshape(permutedims(mpo.Wi[end], (2,1,3,4)), mpo.D[end], 4, 1), L);
    
    end

    mps.physical_space = BraKet();
    mps.d = mps.d^2;
    
    if normalize == true
        if maximum(mps.D) > Dmax # For large tensors, reduces the memory cost of calculating the norm at the price of doing a QR sweep of the MPS
            @timeit to "qr sweep" sweep_qr!(mps);
            println("Doing a sweep")
        end
        @timeit to "norm calc" n = norm(mps);
        nxs = n^(-1/L);
        dmrg_methods.prod!(nxs, mps);
        #for n ∈ 1:L
        #    mps.Ai[n] = mps.Ai[n]/nxs;
        #end
    end

    return mps
end


function update_tensor!(mpo::MPO, tensor::Array{T, 4}, loc::Int) where {T}
    mpo.Wi[loc] = tensor;
    loc == mpo.L && (mpo.D[loc-1] = size(tensor)[1]);
    loc != mpo.L && (mpo.D[loc] = size(tensor)[end]);
    return nothing    
end

function update_tensor!(mps::MPS, tensor::Array{T, 3}, loc::Int) where {T}
    mps.Ai[loc] = tensor;
    loc == mps.L && (mps.D[loc-1] = size(tensor)[1]);
    loc == 1 && (mps.D[1] = size(tensor)[3]);
    loc != mps.L && (mps.D[loc] = size(tensor)[end]);
    loc != 1 && (mps.D[loc-1] = size(tensor)[1]);
    return nothing    
end


@btime collect(Base.ReshapedArray(a, (2^2, 2^8), ()))

@btime reshape(a, 2^2, :)

@btime reshape(view(a,:), (2^2, 2^8))

Base.ReshapedArray(a, (2^2, :), ())

function sweep_qr_timed!(mps::MPS; final_site::Int = mps.L, direction::String = "left")
    #Ai_new = Vector{Array{ComplexF64,3}}();
    L = mps.L;
    d = mps.d;
    
    if direction == "left" && mps.oc != L
        mps.canonical == None() && (mps.oc = 1;) 
        @assert mps.oc < final_site "New orthogonality center must be right from current one"
        
        Atilde =  @timeit to "reshape" reshape(mps.Ai[mps.oc], (:, mps.D[mps.oc]));
        
        for i ∈ mps.oc:final_site-1
            Qi, Ri = @timeit to "qr tensor $(i)" qr(Atilde);
            @timeit to "collect Q$(i)" Qi = Matrix(Qi);
            println(size(Qi))
            @timeit to "update tensor $(i)" update_tensor!(mps, reshape(Qi, (Int(size(Qi,1)/d), d, size(Qi,2))), i);

            @timeit to "reshape tensor $(i)" begin
                Atilde = reshape(Ri * reshape(mps.Ai[i+1], (size(Ri,2), :)), (d * mps.D[i], mps.D[i+1])); 
                #Atilde = reshape(Atilde, (d * mps.D[i], :));    
            end
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

comprket = @btime deepcopy(mps_mpo);

mps_mpo, ϵ_c = mps_compress_var(mps_mpo, seed, :tol_compr => 1e-6, :Dmax => 300, :rate => 1.1); # Seed must be normalized
mpo_comp = cast_mpo(mps_mpo);


### Bond dimension vs error compresssion

dt = 0.1;
tol = 1.0e-6;
ID = "r_id12"
datafolder = "/mnt/lisa/TMI/WII/D_1000/"
root_file = "Wt_alpha=1.0_N=32_t=5.0_dt=$(dt)_tol=$(tol)_$(ID)_step="

e_c = zeros(49,2);

for step ∈ 2:50
    file = h5open("$(datafolder)$(root_file)$(step).h5");
    e_c[step-1,1] = read(file["Diagnosis/ϵ_c"]);
    e_c[step-1,2] = read(file["Diagnosis/Dmax"]);
end

time_axis = collect(0.2:0.1:5.0)

scatter(time_axis, abs.(e_c[:,1]), label=L"\epsilon_c", yscale = :log10)
Wt_comp = scatter(time_axis, abs.(e_c[:,1]), label=L"\epsilon_c", legend = :topleft, yscale = :log10, xlabel = L"t/J", ylabel = L"\epsilon_c")
scatter!(twinx(), xticks = :none, time_axis, abs.(e_c[:,2]), label=L"D_\textrm{max}", mc = :red, ylabel = L"D", legend = :topright);
scatter!(thickness_scaling=1.5)
savelatexfig(Wt_comp, plotsdir("WII/op_dens/Wt_comp_error_alpha=1.0_N=32_dt=0.1_Dmax=1000"))




function Dcutoff(s::Vector{Float64}, Dmax::Int, ϵ::Float64)
    D = length(s);
    sum_disc = 0.0;
    n = 0;
    if ϵ != 0.0
        while sum_disc < ϵ
            sum_disc += s[end - n]^2
            n += 1;
            println("Removing sv $(n). Total sum $(sum_disc)")
        end
        n += -1; # to cancel the last step
    end
    Dkeep = D - n;
    println(Dkeep)
    return min(Dkeep, Dmax)
end


U,S,V = svd(rand(2^10, 2^10));

Dcutoff(S, 2^8, 50000.0)