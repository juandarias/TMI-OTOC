using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using TensorOperations
using BenchmarkTools


using argparser


##### Parameters #####
######################
args_dict = collect_args(ARGS)
const D = get_param!(args_dict, "D", 20);
const Dmin = get_param!(args_dict, "Dmin", 10);
const Dmax = get_param!(args_dict, "Dmax", 50);
const Dstep = get_param!(args_dict, "Dstep", 5);
global JOB_ID = get_param!(args_dict, "JOB_ID", "NA");

include(srcdir("logger.jl"))

##### Contraction sequence test #####
#####################################


########## Method A: Contraction cost = 𝑶(D⁸d⁴). Memory = D⁴
#* By moving the canonical center to the edge of the partition trace out, the bond dimension in the remaining partition is smaller

sizeTGB(D) = 2^4*D^4/1e9 # Size in memory of largest rank-4 tensor 



function contractionA(L::Array{ComplexF64,4}, A::Array{ComplexF64,4})
    LA = zeros(ComplexF64, size(L));
    @tensor LA[a, b, c, d] = L[r, s, t, u]*A[d1, r, d2, a]*A[d2, s, d3, b]*A[d3, t, d4, c]*A[d4,  u, d1, d];
    return nothing
end

function contractionAnA(L::Array{ComplexF64,4}, A::Array{ComplexF64,4})
    #LA = zeros(ComplexF64, size(L));
    @tensor LA[a, b, c, d] := L[r, s, t, u]*A[d1, r, d2, a]*A[d2, s, d3, b]*A[d3, t, d4, c]*A[d4,  u, d1, d];
    return nothing
end

function runDscan(Dmin::Int64, Dmax::Int64, Dstep::Int64)
    m_mean = Float64[];
    t_mean = Float64[];
    m_minimum = Float64[];
    t_minimum = Float64[];
    local L
    local A
    for D in Dmin:Dstep:Dmax
        log_message("\n Running benchmarks for D = $(D) \n")

        L = rand(ComplexF64, D, D, D, D);
        A = rand(ComplexF64, 2, D, 2, D); #! move this into functions
    
        brA = @benchmark contractionAnA($L, $A); #! requires interpolation of variables
        display(brA)
        flush(stdout)
        
        push!(m_mean, Float64(memory(mean(brA)))/1024^2);
        push!(t_mean, Float64(time(mean(brA)))*1e-9);

        push!(m_minimum, Float64(memory(minimum(brA)))/1024^2);
        push!(t_minimum, Float64(time(minimum(brA)))*1e-9);
        log_message("\n min_time,$(t_minimum[end])\n")
        log_message("\n min_mem,$(m_minimum[end])\n")
    end

    return m_mean, t_mean, m_minimum, t_minimum
end

println("Number of threads : ", Threads.nthreads())
m_mean, t_mean, m_minimum, t_minimum = runDscan(Dmin, Dmax, Dstep);


open(datadir(JOB_ID*".dat"), "w") do io
    write(io, "\nTime(s):\n")
    write(io, string(t_mean.*1e-9))
    write(io, "\nMem(MB):\n")
    write(io, string(m_mean./1024^2))
end;