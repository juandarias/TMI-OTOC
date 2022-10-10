using DrWatson
@quickactivate
using HDF5, LinearAlgebra, OrderedCollections


#TODO: read h5 files
#TODO: external linking of HDF5 files
#TODO: rebuild states from Mps
#TODO: read observables



#* ------------------------ No truncation ----------------------------------------
#source_file = h5open(tempname(), "w")
lisa_loc_trunc = "/mnt/lisa/PL_alpha_0.5/TRUNC/"
Ψ_all_trunc = h5open(lisa_loc_trunc*"PL_alpha_0.5_eps_svd=1e-6.h5", "w"); #creates the collection of states
create_group(Ψ_all_trunc, "mps")


for n in 5:5:320
    file_name = lisa_loc_trunc*"L=16_t=16_Bx=0.5_Bz=0.5_NSV=256_eps_svd=1e-6_state_step=$(n).h5"
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    HDF5.h5l_create_external(Ψ_i.filename, "mps", Ψ_all_trunc.id, "mps/step_$(n)", HDF5.H5P_DEFAULT,HDF5.H5P_DEFAULT)
    close(Ψ_i)
end



# create the link
Ψ_all_trunc = h5open(lisa_loc_trunc*"PL_alpha_0.5_eps_svd=1e-6.h5", "r") #creates the collection of states

length(Ψ_all_trunc["mps"])


#* ------------------------ Truncation ----------------------------------------

#source_file = h5open(tempname(), "w")
lisa_loc_notrunc = "/mnt/lisa/PL_alpha_0.5/NOTRUNC/"
Ψ_all_notrunc = h5open(lisa_loc_notrunc*"PL_alpha_0.5_eps_svd=0.0.h5", "w"); #creates the collection of states
create_group(Ψ_all_notrunc, "mps")

for n in 5:5:320
    file_name = lisa_loc_notrunc*"L=16_t=16_Bx=0.5_Bz=0.5_NSV=256_eps_svd=0.0_state_step=$(n).h5"
    Ψ_i = h5open(file_name, "r"); # opens state at step i
    HDF5.h5l_create_external(Ψ_i.filename, "mps", Ψ_all_notrunc.id, "mps/step_$(n)", HDF5.H5P_DEFAULT,HDF5.H5P_DEFAULT)
    close(Ψ_i)
end

read(Ψ_all_notrunc["mps/step_205/2_0_((),())Im"])



close(Ψ_all_notrunc)

Ψ_all_notrunc = h5open(lisa_loc_notrunc*"PL_alpha_0.5_eps_svd=0.0.h5", "r"); #creates the collection of states