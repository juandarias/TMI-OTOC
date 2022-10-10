using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());


using HDF5, LinearAlgebra, SparseArrays
using ColorSchemes, LaTeXStrings, Plots

include(srcdir("load_methods.jl"))
include(plotsdir("plotting_functions.jl"));

####* Paths
root_folder = "/mnt/obelix/TMI/"


####* Read observables
#* α = 3.5
root_folder = "/mnt/iop/Simulations/TMI/L32/alpha_3.5/"
S_single_alpha3_5_NSV512 = readEntaglement("$(root_folder)NSV_512/single_site_forward")


#* α = 3.0
root_folder = "/mnt/iop/Simulations/TMI/L32/alpha_3.0/"
S_single_alpha3_NSV512 = readEntaglement("$(root_folder)NSV_512/single_site_forward")


#* α = 2.5
root_folder = "/mnt/iop/Simulations/TMI/L32/alpha_2.5/"
S_single_alpha2_5_NSV512 = readEntaglement("$(root_folder)NSV_512/single_site_forward")


#* α = 1.5
N = 32;
α = 1.5;
NSVs = [512,700];
root_folder = "/mnt/obelix/TMI/"

S_M_1_5 =  [readEntanglement(32, α, NSVs[i], root_folder, "SS") for i ∈ 1:2]


#* α = 1.1
N = 32;
α = 1.1;
NSVs = [500,700];

S_M_1_1 =  [readEntanglement(32, α, NSVs[i], root_folder, "SS") for i ∈ 1:2]

#* α = 0.5
N = 32;
α = 0.5;
NSVs = [500,700];

S_M_0_5 =  [readEntanglement(32, α, NSVs[i], root_folder, "SS") for i ∈ 1:2]


####* Plot results

Sy_data_1_1 = [S_M_1_1[1][1][16,1:100], S_M_1_1[2][1][16,1:100]]
y_color = [4 6 8]
y_fill = fill(:transparent,(1,3))
y_shape = [:dtriangle :utriangle :ltriangle]
y_label = [L"1-site $D_{max}=500$" L"1-site $D_{max}=700$"]


plot_S_alpha1_1 = scatter(collect(0.1:0.1:10), Sy_data, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape)
#scatter!(collect(0.1:0.1:10), S_single_alpha1_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.5, legend=:topleft)
xlabel!("tJ"); ylabel!(L"S_{16}")
#title!(L"$N=32, \; \alpha=1.5$")
xlims!((8,10));ylims!(3.5,4.5)
savelatexfig(plot_S_alpha1_1, plotsdir("LMU/N32_alpha1_1_S16_vs_t"), tex=true)



Sy_data_0_5 = [S_M_0_5[1][1][16,1:100], S_M_0_5[2][1][16,1:100]]
y_label = [L"1-site $D_{max}=500$" L"1-site $D_{max}=700$"]


plot_S_alpha0_5 = scatter(collect(0.1:0.1:10), Sy_data, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape)
#scatter!(collect(0.1:0.1:10), S_single_alpha1_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.5, legend=:topleft)
xlabel!("tJ"); ylabel!(L"S_{16}")
#title!(L"$N=32, \; \alpha=1.5$")
xlims!((8,10));ylims!(3.5,4.5)
savelatexfig(plot_S_alpha0_5, plotsdir("LMU/N32_alpha0_5_S16_vs_t"), tex=true)


Sy_data_1_5 = [S_M_1_5[1][1][16,1:100], S_M_1_5[2][1][16,1:100]]
y_label = [L"1-site $D_{max}=512$" L"1-site $D_{max}=700$"]


plot_S_alpha1_5 = scatter(collect(0.1:0.1:10), Sy_data, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape)
#scatter!(collect(0.1:0.1:10), S_single_alpha1_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
scatter!(thickness_scaling=1.5, legend=:topleft)
xlabel!("tJ"); ylabel!(L"S_{16}")
#title!(L"$N=32, \; \alpha=1.5$")
savelatexfig(plot_S_alpha0_5, plotsdir("LMU/N32_alpha0_5_S16_vs_t"), tex=true)



#* Combined plot
S_nl_700 = [S_single_alpha3_5_NSV512[16,1:100] S_single_alpha3_NSV512[16,1:100] S_single_alpha2_5_NSV512[16,1:100] Sy_data_1_5[2] Sy_data_1_1[2] Sy_data_0_5[2]]
y_label = [L"$\alpha=3.5$" L"$\alpha=3.0$" L"$\alpha=2.5$" L"$\alpha=1.5$" L"$\alpha=1.1$" L"$\alpha=0.5$"]
y_shape = [:utriangle :dtriangle :cross :diamond :circle :xcross]
y_color = [4 5 6 7 8 9]

#plot_S_alpha_NL = plot(collect(0.1:0.1:10), S_nl_700, label=y_label, palette = :BuPu_9, msc=y_color, mc=y_fill, shape=y_shape, ms=4)

plot_S_alpha_NL = plot(collect(0.1:0.1:10), S_nl_700, label=y_label, palette = :PRGn_6, w=2)
#plot!(collect(0.1:0.1:10), S_single_alpha1_5_NSV400[16,1:100], label=L"1-site $D_{max}=400$",shape=:dtriangle, msc=:6,mc=:transparent);
plot!(thickness_scaling=1.5, legend=:topleft)
xlabel!("tJ"); ylabel!(L"S_{16}")
savelatexfig(plot_S_alpha_NL, plotsdir("LMU/N32_alpha_NL_line_S16_vs_t"), tex=true)
