using MKL
using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());

using BSON, HDF5, SparseArrays, LinearAlgebra
println("Number of threads BLAS = ", BLAS.get_num_threads());
include(srcdir()*"/hamiltonians.jl");
using operators_basis

folder = "/mnt/c/Users/Juan/surfdrive/QuantumSimulationPhD/Code/TDVP/_research/data/unlabelled/"
params_tdvp = load(folder*"L14_JPLparameters.bson")

JPL = hcat(params_tdvp[:coupling_matrix]...)
JNN = diagm(-1 => -4*ones(13), 1 => -4*ones(13))

Nsites = 14;
Bx = [2.1];
Bz = -1;

TF = TransverseFieldIsing(JPL,Bx);
TF_M = convert(SparseMatrixCSC, TF);
Hlong = Bz*spdiagm(sum([diag(Sᶻᵢ(i,Nsites)) for i in 1:Nsites]));
H_TMI = TF_M + Hlong;
evals, evecs = eigen(collect(H_TMI));


TF_NN = TransverseFieldIsing(JNN,Bx);
TF_NN_M = convert(SparseMatrixCSC, TF_NN);
Hlong = Bz*spdiagm(sum([diag(Sᶻᵢ(i,Nsites)) for i in 1:Nsites]));
H_NN = TF_NN_M + Hlong;
evals_NN, evecs_NN = eigen(collect(H_NN));



function unfold_spectrum(evals; f_out=0.1)
    dims = length(evals)
    evals_filtered = evals[floor(Int, dims*f_out ÷ 2):floor(Int, dims*(1-f_out/2))]
    s_unfolded = Float64[];
    for i ∈ 1:11:floor(Int, dims*(1-f_out)-10)
        evals_set = evals_filtered[i:i+10];
        s_set = [evals_set[n+1]-evals_set[n] for n ∈ 1:10];
        s_set_mean = (evals_set[end]-evals_set[1])/10;
        s_set = s_set./s_set_mean
        push!(s_unfolded, s_set...)
    end
    return s_unfolded
end



s_uf =  unfold_spectrum(evals_NN, f_out=0.05)


#* Plots
using Plots, LaTeXStrings
include(plotsdir("plotting_functions.jl"));

pgfplotsx()
s_uf_NN =  histogram(s_uf, label=:none, normalize=:true, fc=nothing);
xgrid!(:none); ygrid!(:none);title!(L"NN");xlabel!("s");
plot!(thickness_scaling=1.5)
savelatexfig(s_uf_NN, plotsdir("LMU/level_spacing_alpha_NN"), tex=true)
