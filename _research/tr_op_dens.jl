using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using TensorOperations
using Tullio
using BenchmarkTools
using UnicodePlots: scatterplot, scatterplot!
using argparser

##### Parameters #####
######################
args_dict = collect_args(ARGS)
const D = get_param!(args_dict, "D", 20);
const Dmin = get_param!(args_dict, "Dmin", 10);
const Dmax = get_param!(args_dict, "Dmax", 50);

##### Contraction sequence test #####
#####################################


########## Method A: Contraction cost = 𝑶(D⁸d⁴). Memory = D⁴
#* By moving the canonical center to the edge of the partition trace out, the bond dimension in the remaining partition is smaller

#sizeTGB(D) = 2^4*D^4/1e9 

function contractionA(L::Array{ComplexF64,4}, A::Array{ComplexF64,4})
    LA = zeros(ComplexF64, size(L));
    @tensor LA[a, b, c, d] = L[r, s, t, u]*A[d1, r, d2, a]*A[d2, s, d3, b]*A[d3, t, d4, c]*A[d4,  u, d1, d];
    return nothing
end

function contractionA2(L::Array{ComplexF64,4}, A::Array{ComplexF64,4})
    LA = zeros(ComplexF64, size(L));
    @tensoropt (a=>χ, b=>χ, c=>χ, d=>χ, r=>χ, s=>χ, t=>χ, u=>χ, d1=>2, d2=>2, d3=>2, d4=>2) LA[a, b, c, d] = L[r, s, t, u]*A[d1, r, d2, a]*A[d2, s, d3, b]*A[d3, t, d4, c]*A[d4,  u, d1, d];
    return nothing
end

function contractionAT(L::Array{ComplexF64,4}, A::Array{ComplexF64,4})
    LA = zeros(ComplexF64, size(L));
    #@tensoropt (a=>χ, b=>χ, c=>χ, d=>χ, r=>χ, s=>χ, t=>χ, u=>χ, d1=>2, d2=>2, d3=>2, d4=>2) LA[a, b, c, d] = L[r, s, t, u]*A[d1, r, d2, a]*A[d2, s, d3, b]*A[d3, t, d4, c]*A[d4,  u, d1, d];
    @tullio LA[a, b, c, d] = L[r, s, t, u]*A[d1, r, d2, a]*A[d2, s, d3, b]*A[d3, t, d4, c]*A[d4,  u, d1, d];
    return nothing
end

########## Method B: Contraction cost = 𝑶(D⁵d³). Memory = D⁴d²

function contractionB(L::Array{ComplexF64,4}, A::Array{ComplexF64,4})
    LA = zeros(ComplexF64, size(L));
    @tensor LAi[d1, a, d2, b, c, d] := L[r, b, c, d]*A[d1, r, d2, a]
    #permutedims!(LA, (1, 3, 2, 4, 5, 6)); 
    @tensor LAi[d1, a, b, d3, c, d] := LAi[d1, a, d2, r, c, d]*A[d2, r, d3, b]
    #permutedims!(LA, (1, 2, 3, 5, 4, 6))
    @tensor LAi[d1, a, b, c, d4, d] := LAi[d1, a, b, d3, r, d]*A[d3, r, d4, c]
    @tensor LA[a, b, c, d] = LAi[d1, a, b, c, d4, r]*A[d4, r, d1, d]
    return nothing
end


function contractionB2(L::Array{ComplexF64,4}, A::Array{ComplexF64,4})
    D = size(L)[end];
    LAi = zeros(ComplexF64, (2, D, D, D, D, 2));
    LA = zeros(ComplexF64, (D, D, D, D));
    @tensor LAi[d1, a, b, c, d, d2] = L[r, b, c, d]*A[d1, r, d2, a]
    #permutedims!(LA, (1, 3, 2, 4, 5, 6)); 
    @tensor LAi[d1, a, b, c, d, d3] = LAi[d1, a, r, c, d, d2]*A[d2, r, d3, b]
    #permutedims!(LA, (1, 2, 3, 5, 4, 6))
    @tensor LAi[d1, a, b, c, d, d4] = LAi[d1, a, b, r, d, d3]*A[d3, r, d4, c]
    @tensor LA[a, b, c, d] = LAi[d1, a, b, c, r, d4]*A[d4, r, d1, d]
    return nothing
end


##### Run tests #####
#####################

########## Single run


println("Running benchmarks for D = $(D)")

const L = rand(ComplexF64, D, D, D, D);
const A = rand(ComplexF64, 2, D, 2, D); #! move this into functions

println("\n Method A \n")
brA = @benchmark contractionA($L, $A)
display(brA)

#println("\n Method A Tullio \n")
#brA = @benchmark contractionAT($L, $A)
#display(brA)




#=
println("Running benchmarks for D = $(D)")

const L = rand(ComplexF64, D, D, D, D);
const A = rand(ComplexF64, 2, D, 2, D); #! move this into functions

println("\n Method A \n")
brA = @benchmark contractionA(L, A)
#display(brA)

 println("\n Method B \n")
brB = @benchmark contractionB(L, A)
display(brB)

println("\n Method B 2")
brB2 = @benchmark contractionB2(L, A)
display(brB2)
=#

########## Run scan

function runDscan(Dmin::Int64, Dmax::Int64)
    m_mean = Float64[];
    t_mean = Float64[];
    m_minimum = Float64[];
    t_minimum = Float64[];
    local L
    local A
    for D in Dmin:5:Dmax
        println("Running benchmarks for D = $(D)")

        L = rand(ComplexF64, D, D, D, D);
        A = rand(ComplexF64, 2, D, 2, D); #! move this into functions
    
        brA = @benchmark contractionB2($L, $A); #! requires interpolation of variables
        display(brA)
        push!(m_mean, Float64(memory(mean(brA))));
        push!(t_mean, Float64(time(mean(brA))));

        push!(m_minimum, Float64(memory(minimum(brA))));
        push!(t_minimum, Float64(time(minimum(brA))));
    end

    return m_mean, t_mean, m_minimum, t_minimum
end

#= 
println("Number of threads : ", Threads.nthreads())
m_mean_B2, t_mean_B2, m_minimum_B2, t_minimum_B2 = runDscan(Dmin, Dmax);

memplot = scatterplot(collect(Dmin:5:Dmax), m_mean, name = "Mem_mean", xlabel = "D")
scatterplot!(memplot, collect(Dmin:5:Dmax), m_minimum, name = "Mem_min")

println("Memory usage")
display(memplot)

time_plot = scatterplot(collect(Dmin:5:Dmax), t_mean, name = "Time_mean", xlabel = "D")
scatterplot!(time_plot, collect(Dmin:5:Dmax), t_minimum, name = "Time_min")

println("Computational time")
display(time_plot)
 =#
