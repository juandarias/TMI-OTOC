using HDF5, Plots, LaTeXStrings
using Plots.PlotMeasures

using DrWatson
@quickactivate

include(plotsdir("plotting_functions.jl"))

op_size(rho_t::Array) = [sum([l * rho_t[s, l] for l ∈ axes(rho_t, 2)]) for s ∈ axes(rho_t, 1)];
op_size_v(rho_t::Vector) = sum([l * rho_t[l] for l ∈ axes(rho_t, 1)]);
size_Haar(N) = N * (1 + 1/(4^N - 1)) - 1/3
op_size_norm(rho_t, N) = op_size(rho_t)/size_Haar(N)


##### ED results #####

ed_out_10 = h5open(datadir("exact/op_density/dat1.1/op_dens_mfi_n10_a1.1.hdf5"))
ed_out_12 = h5open(datadir("exact/op_density/dat1.1/op_dens_mfi_n12_a1.1.hdf5"))
ed_out_14 = h5open(datadir("exact/op_density/dat1.1/op_dens_mfi_n14_a1.1.hdf5"))
ed_out_16 = h5open(datadir("exact/op_density/dat1.1/op_dens_mfi_n16_a1.1.hdf5"))

t_ed_10 = read(ed_out_10["time"])
t_ed_12 = read(ed_out_12["time"])
t_ed_14 = read(ed_out_14["time"])
t_ed_16 = read(ed_out_16["time"])
rho_10_ed = read(ed_out_10["op_dens_Y"]);
rho_12_ed = read(ed_out_12["op_dens_Y"]);
rho_14_ed = read(ed_out_14["op_dens_Y"]);
rho_16_ed = read(ed_out_16["op_dens_Y"]);

op_size_10_ed = op_size(rho_10_ed);
op_size_12_ed = op_size(rho_12_ed);
op_size_14_ed = op_size(rho_14_ed);
op_size_16_ed = op_size(rho_16_ed);

##### MPO results #####

t_mpo = 0.1 * collect(1:26) .+ 2.4

#= N = 10 =#

N = 10;
alpha = 1.1;
tf = 5.0;
steps = 25;
tol_compr_min = 1e-8;
tol_compr_max = 1e-5;
outfile = "rho_exact_N=$(N)_alpha=$(alpha)_tf=$(tf).h5";
res_10 = h5open(datadir("exact/op_density/$(outfile)"));


r_max = floor(Int, log10(tol_compr_max/tol_compr_min));
tol_range = tol_compr_min * vcat([[10^n, 5 * 10^n] for n ∈ 0:r_max]...);


rho_10 = [];
op_size_10 = [];
bond_ent_10 = [];
for tol_compr ∈ tol_range
    push!(rho_10, read(res_10["rho/tol_$(tol_compr)"]))
    push!(op_size_10, op_size(rho_10[end]))
    push!(bond_ent_10, read(res_10["entropy/tol_$(tol_compr)"]))
end



op_size_10 = scatter(t_mpo, op_size_10[1], label = "MPO, ϵ = 5e-5")
plot!(t_ed_10[10:30], abs.(op_size_10_ed)[10:30], label = "ED")
title!("N=10, α = 1.1")
xlabel!("tJ")
ylabel!("L")
savelatexfig(op_size_10, plotsdir("exact/op_size_alpha=1.1_N=10_exact_MPO_vs_ED"))
plot()

#= N = 10, small dt =#
op_size_10_mps = [];
norm_rho_mps = [];


N = 10;
tf = 5.0;
alpha = 1.1;
dt = 0.005;
tol_compr = 5e-10;
ID = 60;
Dmax = 4096;
outfile = "rho_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_r_id$(ID).h5";
lisa_dir = "/mnt/lisa/TMI/WII/results/"
res_10 = h5open("$(lisa_dir)$(outfile)");

t_max = 4.0;
op_size_10 = []
norm_rho = []

for s in 2:Int(t_max/dt)
    rho_s = read(res_10["rho_l/step_$(s)"]);
    push!(op_size_10, op_size_v(rho_s));
    push!(norm_rho, sum(rho_s));
end


push!(op_size_10_mps, op_size_10);
push!(norm_rho_mps, norm_rho);


lt = [:dash, :dot, :dashdot, :solid, :auto];

plot(collect(2:400) * 0.01, op_size_10_mps[1], name = "dt=0.01, ϵ = 1e-8", left_margin=1cm, ls = lt[1])
plot!(collect(2:400) * 0.01, op_size_10_mps[2], name = "dt=0.01, ϵ = 1e-9", ls = lt[2])
plot!(collect(2:800) * 0.005, op_size_10_mps[3], name = "dt=0.005, ϵ = 1e-8", ls = lt[3])
plot!(collect(2:800) * 0.005, op_size_10_mps[4], name = "dt=0.005, ϵ = 1e-9", ls = lt[4])
plot!(collect(2:800) * 0.005, op_size_10_mps[5], name = "dt=0.005, ϵ = 5e-10", ls = lt[5])
plot!(t_ed_10[1:20], abs.(op_size_10_ed)[1:20], label = "ED", mode="markers")
title!("N=10, α = 1.1")
xlabel!("tJ")
ylabel!("L")


op_size_10_fig = plot(collect(2:800) * 0.005, op_size_10_mps[1], label = "dt=0.005, ϵ = 5e-10", ls = lt[1]);
scatter!(t_ed_10[1:20], abs.(op_size_10_ed)[1:20], label = "ED", markershape = :cross);
title!("N=10, α = 1.1");
xlabel!("tJ");
ylabel!("L")

savelatexfig(op_size_10_fig, plotsdir("WII/op_size_10_dt=0.005_vs_ED"))


#= N = 12, small dt =#
N = 12;
tf = 5.0;
alpha = 1.1;
dt = 0.005;
tol_compr = 1e-8;
ID = 82;
Dmax = 4096;
outfile = "rho_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_r_id$(ID).h5";
lisa_dir = "/mnt/lisa/TMI/WII/results/"
res_12 = h5open("$(lisa_dir)$(outfile)");

t_max = 2.8;
s_max = Int(t_max/dt);
op_size_12 = []
norm_rho_12 = []

for s in 2:Int(t_max/dt)
    rho_s = read(res_12["rho_l/step_$(s)"]);
    push!(op_size_12, op_size_v(rho_s));
    push!(norm_rho_12, sum(rho_s));
end



op_size_12_fig = plot(collect(2:s_max) * 0.005, op_size_12, label = "dt=0.005, ϵ = 1e-8", ls = lt[1])
scatter!(t_ed_12[1:15], abs.(op_size_12_ed)[1:15], label = "ED", markershape = :cross);
title!("N=12, α = 1.1");
xlabel!("tJ");
ylabel!("L")


savelatexfig(op_size_12_fig, plotsdir("WII/op_size_12_dt=0.005_vs_ED"))


#= N = 12 =#
N = 12;
outfile = "rho_exact_N=$(N)_alpha=$(alpha)_tf=$(tf).h5";
res_12 = h5open(datadir("exact/op_density/$(outfile)"));


rho_12 = [];
op_size_12 = [];
bond_ent_12 = [];
for tol_compr ∈ tol_range
    push!(rho_12, read(res_12["rho/tol_$(tol_compr)"]))
    push!(op_size_12, op_size(rho_12[end]))
    push!(bond_ent_12, read(res_12["entropy/tol_$(tol_compr)"]))
end


op_size_12 = scatter(t_mpo, op_size_12[1], label = "MPO, ϵ = 5e-5")
plot!(t_ed_12[10:30], abs.(op_size_12_ed)[10:30], label = "ED")
title!("N=12, α = 1.1")
xlabel!("tJ")
ylabel!("L")
savelatexfig(op_size_12, plotsdir("exact/op_size_alpha=1.1_N=12_exact_MPO_vs_ED"))



## N = 14
N = 14;
tf = 10.0;
outfile = "rho_exact_N=$(N)_alpha=$(alpha)_tf=$(tf).h5";
lisa_dir = "/mnt/lisa/TMI/exact/op_density/"
res_14 = h5open("$(lisa_dir)$(outfile)");


rho_14 = [];
op_size_14 = [];
bond_ent_14 = [];
for tol_compr ∈ tol_range
    push!(rho_14, read(res_14["rho/tol_$(tol_compr)"]))
    push!(op_size_14, op_size(rho_14[end]))
    push!(bond_ent_14, read(res_14["entropy/tol_$(tol_compr)"]))
end


t_mpo_14 = 9.5:0.1:10


scatter(op_size_14[1]-op_size_14[end])
op_size_14_plot = scatter(t_mpo_14, op_size_14[2], label = "MPO, ϵ = 5e-8")
plot!(t_ed_14[45:55], abs.(op_size_14_ed)[45:55], label = "ED")
title!("N=14, α = 1.1")
xlabel!("tJ")
ylabel!("L")
savelatexfig(op_size_14_plot, plotsdir("exact/op_size_alpha=1.1_N=14_exact_MPO_vs_ED"))


## N = 10
N = 10;
tf = 5.0;
alpha = 1.1;
dt = 0.05;
tol_compr = 1e-10;
Dmax = 4096;
ID = 64;
outfile = "rho_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_r_id$(ID).h5";
outfile_svd = "rho_svd_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_r_id$(ID).h5";
lisa_dir = "/mnt/lisa/TMI/WII/results/"
res_10 = h5open("$(lisa_dir)$(outfile)");
res_10_svd = h5open("$(lisa_dir)$(outfile_svd)");

op_size_10_svd_ht = []
norm_rho_svd_ht = []

op_size_10_ht = []
norm_rho_ht = []

for s in 2:100
    rho_s_svd = read(res_10_svd["rho_l/step_$(s)"]);
    push!(op_size_10_svd_ht, op_size_v(rho_s_svd));
    push!(norm_rho_svd_ht, sum(rho_s_svd));

    rho_s = read(res_10["rho_l/step_$(s)"]);
    push!(op_size_10_ht, op_size_v(rho_s));
    push!(norm_rho_ht, sum(rho_s));
end


op_size_10_svd_ht = op_size_10_svd_ht ./ norm_rho_svd_ht

t_mpo = collect(2:100)*0.05;

op_size_10_plot = scatter(t_mpo, op_size_10_ht, label = "MPO var, ϵ = 1e-6", markershape = :xcross)
scatter!(t_mpo, op_size_10_svd_ht, label = "MPO svd, ϵ = 1e-6", markershape = :cross, msc = 1)
plot!(t_ed_10[1:20], abs.(op_size_10_ed)[1:20], label = "ED")
title!("N=14, α = 1.1")
xlabel!("tJ")
ylabel!("L")
xlims!(0,2)
ylims!(1,4)


## N = 10 exact Ut

op_size_10_exact_all = []

N = 10;
tf = 5.0;
alpha = 1.1;
dt = 0.05;
tol_compr = 1e-8;
Dmax = 4096;
ID = 80; #80 -> 1e-6; 42 -> 1e-8
steps = 100;
outfile = "rho_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_r_id$(ID).h5";
lisa_dir = "/mnt/lisa/TMI/exact/results/"
res_10_exact = h5open("$(lisa_dir)$(outfile)");

settings_file = "Wt_alpha=1.1_N=10_t=5.0_dt=0.05_tol=1.0e-8_r_id$(ID).bson"
params_sim = load("/mnt/lisa/TMI/$(settings_file)")

op_size_10_exact = []
norm_rho_exact = []

for s in 2:steps
    rho_s = read(res_10_exact["rho_l/step_$(s)"]);
    push!(op_size_10_exact, op_size_v(rho_s));
    push!(norm_rho_exact, sum(rho_s));
end

push!(op_size_10_exact_all, op_size_10_exact)



t_mpo = collect(2:steps)*0.05;

markers = [:cross, :xcross, :diamond, :utriangle, :circle];

op_size_10_plot = scatter(t_mpo[1:49], op_size_10_exact_all[1], label = "MPO, ϵ(U(dt)) = 5e-10", markershape = markers[1], markercolor = nothing, msc = 1)
scatter!(t_mpo[1:49], op_size_10_exact_all[2], label = "MPO, ϵ(U(dt)) = 1e-9", markershape = markers[2], markercolor = nothing, msc = 2)
scatter!(t_mpo, op_size_10_exact_all[3], label = "MPO, ϵ(U(dt)) = 1e-8", markershape = markers[3], markercolor = nothing, msc = 3)
scatter!(t_mpo, op_size_10_exact_all[4], label = "MPO, ϵ(U(dt)) = 1e-7", markershape = markers[4], markercolor = nothing, msc = 4)
scatter!(t_mpo, op_size_10_exact_all[5], label = "MPO, ϵ(U(dt)) = 1e-6", markershape = markers[5], markercolor = nothing, msc = 5)

plot!(t_ed_10[1:30], abs.(op_size_10_ed)[1:30], label = "ED")
title!("N=10, α = 1.1")
xlabel!("tJ")
ylabel!("L")
xlims!(2,2.5)
ylims!(3.0,3.75)
savelatexfig(op_size_10_plot, plotsdir("exact/op_size_alpha_1.1_Udt_exact_vs_ED_low_comp"))



## N = 16
N = 16;
tf = 5.0;
alpha = 1.1;
dt = 0.1;
tol_compr = 1e-6;
Dmax = 4096;
ID = 31;
outfile = "rho_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_r_id$(ID).h5";
outfile = "rho_svd_alpha=$(alpha)_N=$(N)_t=$(tf)_dt=$(dt)_tol=$(tol_compr)_Dmax=$(Dmax)_r_id$(ID).h5";
lisa_dir = "/mnt/lisa/TMI/WII/results/"
res_16 = h5open("$(lisa_dir)$(outfile)");
res_16_svd = h5open("$(lisa_dir)$(outfile)");

op_size_16_svd = []
norm_rho_svd = []
for s in 2:50
    rho_s = read(res_16_svd["rho_l/step_$(s)"]);
    push!(op_size_16_svd, op_size_v(rho_s));
    push!(norm_rho_svd, sum(rho_s));
end

t_mpo = collect(2:50)*0.1

op_size_16_plot = scatter(t_mpo, op_size_16, label = "MPO var, ϵ = 1e-6", markershape = :xcross)
scatter!(t_mpo, op_size_16_svd, label = "MPO svd, ϵ = 1e-6", markershape = :cross)
plot!(t_ed_16[1:55], abs.(op_size_16_ed)[1:55], label = "ED")
title!("N=14, α = 1.1")
xlabel!("tJ")
ylabel!("L")


pl_coefs = h5open(projectdir("input/pl_Ham_MPO_Mmax=10_L=10_kac_norm.h5"))




espectrum = read(res_14["spectrum/tol_5.0e-7"])[1,:,:]
D = read(res_14["D/tol_5.0e-7"])[1,:]
S = read(res_14["entropy/tol_5.0e-7"])[1,:]

Dmax = 1000
scatter(1:Dmax, espectrum[1,1:Dmax] .+ 1e-10, xscale = :log10, yscale = :log10)
for s ∈ 2:13
    display(scatter!(1:Dmax, espectrum[s,1:Dmax] .+ 1e-10, xscale = :log10, yscale = :log10))
end
ylims!(1e-5, 1)

scatter(1:13, espectrum[:,1:Dmax] .+ 1e-10; yscale= :log10, label = :none, markershape = :cross, msc = 1, markercolor = nothing)
ylims!(1e-5, 1)
S

gr()
