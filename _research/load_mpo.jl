
mpo = h5open(datadir("MPO_alpha_1.5_L32.h5"), "r+")

mpo_onlyW = h5open(datadir("W_MPO_alpha1.5_L32.h5"), "r+")
create_group(mpo_onlyW, "Parameters")
mpo_onlyW["Parameters/Bond dimension"] = [900, 700]
mpo_onlyW["Parameters/Total time"] = 10
mpo_onlyW["Parameters/Time step"] = 0.1
mpo_onlyW["Parameters/Method"] = "1-site"
mpo_onlyW["Parameters/Lanczos tolerance"] = 1e-8

read(mpo_onlyW["W/loc=17,m=1,n=1"])

close(mpo_onlyW)

for l ∈ 0:31
    for m ∈ [0,1], n ∈ [0,1]
        matrix_name = "loc=$l,m=$m,n=$n,t=0,k'=0,k=0"
        new_matrix_name = "loc=$(l+1),m=$m,n=$n"
        mpo_onlyW["W/"*new_matrix_name] = Float64.(read(mpo["W/"*matrix_name]))
    end
end

close(mpo_onlyW)
