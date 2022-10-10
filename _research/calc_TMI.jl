using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5, LinearAlgebra, SparseArrays, Plots, LaTeXStrings
pgfplotsx();
include(srcdir("MPS_conversions.jl"));
include(srcdir("load_methods.jl"));
include(srcdir("ncon.jl"));
include(plotsdir("plotting_functions.jl"));

L=32;
NSV=512;

state_i = datadir("single_site_forward_state_step=1.h5")
data_state = h5open(state_i, "r")
mps_state = read(data_state["mps"])
psi_32 = reconstructcomplexState(mps_state)