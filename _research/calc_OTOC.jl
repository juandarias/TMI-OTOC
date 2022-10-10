using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());
using LsqFit

using HDF5, LinearAlgebra, SparseArrays, Plots, LaTeXStrings, ColorSchemes, LsqFit
pgfplotsx();
include(srcdir("MPS_conversions.jl"));
include(srcdir("load_methods.jl"));
include(srcdir("ncon.jl"));
include(plotsdir("plotting_functions.jl"));

####* Load previous workspace
using Interpolations, LsqFit
using JLD2

@load "TMI_benasque.jld2"

#####* Parameters
L=32;
NSV=512;
location="/mnt/iop/Simulations/TMI" #to mount: sudo mount.cifs ''//iop-s0.science.uva.nl/Gerritsma'' -o user=JDARIAS /mnt/iop/


#####* Entropies
#! S_forward_3_25 = readEntanglement(L, 3.25, 400, location, "SS"); #* alpha = 3.25

S_hc_forward = [];
for α ∈ [2.5 2.75 3 3.5]
    push!(S_hc_forward, readEntanglement(L, α, 512, location, "SS")[1][15,1:100]);
end
push!(S_hc_forward, readEntanglement(L, 3.75, 400, location, "SS")[1][15,1:100]);

S_L32_norm = plot();
for (i,α) ∈ enumerate([2.5 2.75 3 3.5 3.75])
    plot!(collect(0.1:0.1:10), S_hc_forward[i]/log(2), label="$α",  lc=i+2, palette = :BuPu_9)
end
plot!(thickness_scaling=1.55,legend=:topleft)
xlabel!("tJ"); ylabel!(L"\tilde{S}_{16}")
savelatexfig(S_L32_norm, plotsdir("LMU/N32_SS_S16_updated"), tex=true)

plot!(collect(0.1:0.1:10), S_single_alpha1_5_NSV700[15,1:100], label="1.5",  lc=6+2, palette = :BuPu_9, ls=:dash)

default(fmt=:svg)

scatter!(collect(0.1:0.1:10), readEntanglement(L, 3.75, 400, location, "SS")[1][15,1:100], label="3.75 rep",  palette = :BuPu_9, ms=3)

#####* Entropy speed
model(t, a) = a[1]*ones(51) + a[2]*t;
a0 = [0.0, 0.5]

vEntropy=zeros(5);
for i in 1:5
    S_slice =  S_hc_forward[i][25:75]
    fit = curve_fit(model, collect(2.5:0.1:7.5), S_slice, a0)
    sol = fit.param
    vEntropy[i] = sol[2];
end


#####* OTOC A
#* α=2.5
OTOC_A_2_5 = readOTOC(2.5, "A");
OTOC_A_2_5_plot =  plotOTOC(OTOC_A_2_5, AorB="A")
OTOC_A_2_5_fit, OTOC_A_2_5_t_v, OTOC_A_2_5_tfitted =  velocityOTOC(OTOC_A_2_5, 22, 29)
OTOC_A_2_5_plot =  plotOTOC(OTOC_A_2_5, AorB="A", fitted_times = OTOC_A_2_5_tfitted)
C_t_2_5_A = interpolateOTOC(OTOC_A_2_5)

savelatexfig(OTOC_A_2_5_plot, plotsdir("Benasque/OTOC_A_2_5_fit"), tex=true)

#* α=2.75
OTOC_A_2_75 = readOTOC(2.75, "A");
OTOC_A_2_75_plot =  plotOTOC(OTOC_A_2_75, AorB="A")
OTOC_A_2_75_fit, OTOC_A_2_75_t_v, OTOC_A_2_75_tfitted =  velocityOTOC(OTOC_A_2_75, 21, 29)
OTOC_A_2_75_plot =  plotOTOC(OTOC_A_2_75, AorB="A", fitted_times = OTOC_A_2_75_tfitted)
savelatexfig(OTOC_A_2_75_plot, plotsdir("Benasque/OTOC_A_2_75_fit"), tex=true)
OTOC_A_2_75[:,11].=1.0
OTOC_A_2_75[:,11]= 0.5*(OTOC_A_2_75[:,10] +OTOC_A_2_75[:,12])


#* α=3.0
OTOC_A_3_0 = readOTOC(3.0, "A");
OTOC_A_3_0_plot =  plotOTOC(OTOC_A_3_0, AorB="A")
OTOC_A_3_0_fit, OTOC_A_3_0_t_v, OTOC_A_3_0_tfitted =  velocityOTOC(OTOC_A_3_0, 19, 29)
OTOC_A_3_0_plot =  plotOTOC(OTOC_A_3_0, AorB="A",fitted_times = OTOC_A_3_0_tfitted)
savelatexfig(OTOC_A_3_0_plot, plotsdir("Benasque/OTOC_A_3_0_fit"), tex=true)

#* α=3.5
OTOC_A_3_5 = readOTOC(3.5, "A");
OTOC_A_3_5_plot =  plotOTOC(OTOC_A_3_5, AorB="A")
OTOC_A_3_5_fit, OTOC_A_3_5_t_v, OTOC_A_3_5_tfitted =  velocityOTOC(OTOC_A_3_5, 19, 29)
OTOC_A_3_5_plot =  plotOTOC(OTOC_A_3_5, AorB="A",fitted_times = OTOC_A_3_5_tfitted)
savelatexfig(OTOC_A_3_5_plot, plotsdir("Benasque/OTOC_A_3_5_fit"), tex=true)

v_butterfly = [-1/OTOC_A_2_5_fit[2], -1/OTOC_A_2_75_fit[2],-1/OTOC_A_3_0_fit[2],-1/OTOC_A_3_5_fit[2]]


#####* OTOC B
#*α=2.5
OTOC_B_2_5 = readOTOC(2.5, "B");
OTOC_B_2_5_plot =  plotOTOC(OTOC_B_2_5, AorB="B")
OTOC_B_2_5_fit, OTOC_B_2_5_t_v, OTOC_B_2_5_tfitted =  velocityOTOC(OTOC_B_2_5, 5, 14, "B")
OTOC_B_2_5_plot =  plotOTOC(OTOC_B_2_5, AorB="B", fitted_times = OTOC_B_2_5_tfitted)
savelatexfig(OTOC_B_2_5_plot, plotsdir("Benasque/OTOC_B_2_5_fit"), tex=true)

#*α=3.0 
#!wrong
OTOC_B_3_0 = readOTOC(3.0, "B";step_end=95)[:,1:19];
OTOC_B_3_0_plot =  plotOTOC(OTOC_B_3_0, AorB="B", tfinal=9.5)
OTOC_B_3_0_fit, OTOC_B_3_0_t_v, OTOC_B_3_0_tfitted =  velocityOTOC(OTOC_B_3_0, 5, 14, "B")
OTOC_B_3_0_plot =  plotOTOC(OTOC_B_3_0, AorB="B", fitted_times = OTOC_B_3_0_tfitted)
savelatexfig(OTOC_B_3_0_plot, plotsdir("Benasque/OTOC_B_2_5_fit"), tex=true)

#*α=3.5
OTOC_B_3_5 = readOTOC(3.5, "B";step_end=95)[:,1:19];
OTOC_B_3_5_plot =  plotOTOC(OTOC_B_3_5, AorB="B", tfinal=9.5)
OTOC_B_3_5_fit, OTOC_B_3_5_t_v, OTOC_B_3_5_tfitted =  velocityOTOC(OTOC_B_3_5, 5, 14, "B", tfinal=9.5)
OTOC_B_3_5_plot =  plotOTOC(OTOC_B_3_5, AorB="B", fitted_times = OTOC_B_3_5_tfitted, tfinal=9.5)
savelatexfig(OTOC_B_3_5_plot, plotsdir("Benasque/OTOC_B_3_5_fit"), tex=true)


#####* Velocity comparison

alphas = [2.5, 2.75, 3, 3.5, 3.75]

vels_plot = plot(layout=(1,2),framestyle=:box);
scatter!(alphas, vEntropy, marker=:xcross, msc=:red, label=:none, ylabel=L"v_E",subplot=2, ylims=(0.4,0.65),xlims=(2,4));
scatter!(alphas[1:4], v_butterfly, marker=:cross, msc=:blue, label=:none, ylabel=L"v_B", subplot=1, ylims=(1.0,1.55),xlims=(2,4));
xlabel!(L"\alpha");
plot!(thickness_scaling=1.75)
savelatexfig(vels_plot, plotsdir("Benasque/vel_comparison"), tex=true)


#####* OTOC speed

function interpolateOTOC(OTOC; tfinal=10)
    t_range = 0.5:0.5:tfinal
    d_range = 1:1:31; 
    dmax=31;
    #AorB=="A" && (d_range = 1:1:31; dmax=31);
    #AorB=="B" && (d_range = 1:1:15; dmax=15);
    itp = interpolate(OTOC, BSpline(Cubic(Line(OnGrid()))));
    sitp = scale(itp, d_range, t_range)

    C_t = zeros(191,dmax);
    for (i,d) ∈ enumerate(d_range)
        for (j,t) ∈ enumerate(0.5:0.05:tfinal)
            C_t[j,i] = sitp(d,t)
        end
    end

    return C_t
end

C_t_2_5_A = interpolateOTOC(OTOC_A_2_5)
C_t_2_75_A = interpolateOTOC(OTOC_A_2_75)
C_t_3_0_A = interpolateOTOC(OTOC_A_3_0)
C_t_3_5_A = interpolateOTOC(OTOC_A_3_5)

i=2
C_T_OTOC_A = plot(0.5:0.05:10, -C_t_2_5_A[:,25].+1, label=L"2.5", lc=i+2, palette = :BuPu_9);
plot!(0.5:0.05:10, -C_t_2_75_A[:,25].+1, label=L"2.75", lc=i+3, palette = :BuPu_9);
plot!(0.5:0.05:10, -C_t_3_0_A[:,25].+1, label=L"3.0", lc=i+4, palette = :BuPu_9);
plot!(0.5:0.05:10, -C_t_3_5_A[:,25].+1, label=L"3.5", lc=i+5, palette = :BuPu_9);
ylims!(-0.05,0.5);
xlims!(0,7);
xlabel!("t/J"); ylabel!(L"C(t)");
plot!(thickness_scaling=1.5)
savelatexfig(C_T_OTOC_A, plotsdir("Benasque/C(t)_A_d=25"), tex=true)