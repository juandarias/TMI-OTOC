using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());


using HDF5, LinearAlgebra, SparseArrays
using ColorSchemes, LaTeXStrings, Plots

include(srcdir("load_methods.jl"))
include(plotsdir("plotting_functions.jl"));

#* α = 2.5
root_folder = "/mnt/iop/Simulations/TMI/L32/alpha_2.5/"
S_single_alpha2_5_NSV512 = readEntaglement("$(root_folder)NSV_512/single_site_forward")



obelix_folder = "/mnt/obelix/TDVP/"
S_2site = readEntaglement("$(obelix_folder)L32/alpha_2.5/NSV_512/two_site_forward")
S_2site_1e8 = readEntaglement("$(obelix_folder)L32/alpha_2.5/NSV_512/two_site_eps_1e-8_forward")
S_2site_5e9 = readEntaglement("$(obelix_folder)L32/alpha_2.5/NSV_512/two_site_eps_tw=5e-9_forward")
S_2site_1e9 = readEntaglement("$(obelix_folder)L32/alpha_2.5/NSV_512/two_site_eps_tw=1e-9_forward")

D_2site_1e6 = readBonddimension("$(obelix_folder)L32/alpha_2.5/NSV_512/two_site_forward")
D_2site_1e8 = readBonddimension("$(obelix_folder)L32/alpha_2.5/NSV_512/two_site_eps_1e-8_forward")

sdata =  h5open(datadir("LMU/entropy_comparison.h5"), "w")
create_group(sdata, "alpha_2.5")
sdata["alpha_2.5/S_1site"] = S_single_alpha2_5_NSV512[1:end-1];
sdata["alpha_2.5/S_2site_eps_tw=1e-6"] = S_2site[1:end-2];
sdata["alpha_2.5/S_2site_eps_tw=1e-8"] = S_2site_1e8[1:end-2];

close(sdata)

pars = load("$(obelix_folder)L32/alpha_2.5/NSV_512/two_site_parameters.bson")
pars[:Args]

using Plots
gr()

S_single_alpha2_5_NSV512[:,1]

s16 = plot(collect(0.05:0.05:5), S_2site_1e8[16,1:end-2], label="2TDVP, ϵ=1e-8")
#plot!(collect(0.05:0.05:5), S_2site_1e9[16,1:end-2])
#plot!(collect(0.05:0.05:5), S_2site_1e8[16,1:end-2])
plot!(collect(0.05:0.05:5), S_2site[16,1:end-2], label="2TDVP, ϵ=1e-6")
scatter!(collect(0.1:0.1:5), S_single_alpha2_5_NSV512[16,1:50], ms=1, label="1TDVP", mc=:black)
plot!(xlabel="tJ", ylabel="S₁₆")
savelatexfig(s16, plotsdir("LMU/S_16_1_site_vs_2_site"))


Dmax = plot(collect(0.05:0.05:5), D_2site_1e8[1:end-2], label="2TDVP, ϵ=1e-8")
plot!(collect(0.05:0.05:5), D_2site_1e6[1:end-2], label="2TDVP, ϵ=1e-6")
plot!(xlabel="tJ", ylabel="D")
savelatexfig(Dmax, plotsdir("LMU/D_1_site_vs_2_site"))



# alpha =2.5, N = 32 debug
"sudo mount.cifs //iop-s0.science.uva.nl/Gerritsma -o user=JDARIAS@uva.nl /mnt/iop/"  

"sshfs -o IdentityFile=/home/juandarias/.ssh/id_ed25519 jdarias@obelix-h0.science.uva.nl:/home/jdarias/TDVP/_research/data /mnt/obelix/TDVP"

root_folder = "/mnt/iop/Simulations/TMI/L32/alpha_2.5/"
S_single_alpha2_5_NSV512 = readEntaglement("$(root_folder)NSV_512/single_site_forward")

data = h5open("$(root_folder)NSV_512/single_site_forward_observables.h5")

Bx = -2.1;
Bz = -1.0;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_A =  readEntanglement(cbe_folder*file_name)

Bx = 2.1;
Bz = -1.0;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_B =  readEntanglement(cbe_folder*file_name)

Bx = -1.05;
Bz = 0.5;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_C =  readEntanglement(cbe_folder*file_name)

Bx = 1.05;
Bz = 0.5;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_D =  readEntanglement(cbe_folder*file_name)

Bx = -2.1;
Bz = 1.0;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_E =  readEntanglement(cbe_folder*file_name)

Bx = 2.1;
Bz = 1.0;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_F =  readEntanglement(cbe_folder*file_name)

Bx = 1.05;
Bz = -0.5;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_G =  readEntanglement(cbe_folder*file_name)

Bx = -1.05;
Bz = -0.5;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_forward"
S_H =  readEntanglement(cbe_folder*file_name)

Bx = -2.1;
Bz = 1.0;
cbe_folder = "/mnt/obelix/TDVP/CBE/"
file_name = "single_site_Bx=$(Bx)_Bz=$(Bz)_J=4.0_forward"
S_I =  readEntanglement(cbe_folder*file_name)

markers = [:cross, :xcross, :diamond, :utriangle, :circle];

scatter(1:31, S_A[:,1], label = L"h_x = -2.1; \quad h_z = -1.0", markershape = markers[1], markercolor = nothing, msc = 1)
scatter!(1:31, S_B[:,1], label = L"h_x = 2.1; \quad h_z = -1.0", markershape = markers[2], markercolor = nothing, msc = 2)
scatter!(1:31, S_C[:,1], label = L"h_x = -1.05; \quad h_z = 0.5", markershape = markers[3], markercolor = nothing, msc = 2)
scatter!(1:31, S_D[:,1], label = L"h_x = 1.05; \quad h_z = 0.5", markershape = markers[4], markercolor = nothing, msc = 4)
scatter!(1:31, S_E[:,1], label = L"h_x = -2.1; \quad h_z = 1.0", markershape = markers[5], markercolor = nothing, msc = 5)
scatter!(1:31, S_F[:,1], label = L"h_x = 2.1; \quad h_z = 1.0", markershape = markers[1], markercolor = nothing, msc = 6)
scatter!(1:31, S_G[:,1], label = L"h_x = 1.05; \quad h_z = -0.5", markershape = markers[2], markercolor = nothing, msc = 7)
scatter!(1:31, S_H[:,1], label = L"h_x = -1.05; \quad h_z = -0.5", markershape = markers[3], markercolor = nothing, msc = 8)
scatter!(1:31, S_H[:,1], label = L"h_x = -2.1; \quad h_z = 1.0; \quad J = 4", markershape = markers[4], markercolor = nothing, msc = 9)
ylabel!(L"S_{vN}")
xlabel!("Bond")

beo = h5open(datadir("benchmark_results.h5"))

function JPL(α, L)
    J = zeros(L,L);
    for i ∈ 1:L, j ∈ i+1:L
        J[i,j] = abs(i-j)^(-α)
    end
    kacn = sum(J)/(L-1)
    J += transpose(J)
    return J/kacn
    #return J
end

J32 = JPL(2.5, 32)

h5open(datadir("LMU/Jij_alpha=2.5_N=32.h5"), "w") do file;
    write(file, "Jij", J32)
end

read(h5open(datadir("LMU/Jij_alpha=2.5_N=32.h5"))["Jij"])

h5open()