## Find power-law approximation

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
using LinearAlgebra
using Optim
using Plots

"""
Computes coefficients βᵢ, λᵢ that minimize ‖1/rᵅ -  ∑βᵢλᵢʳ⁻¹‖.

# Arguments
- `α`: power-law exponente
- `distance`: distance range on which to find the best fit
- `M_max` : number of coefficients to fit. Determines the bond dimension of the MPO

"""
function pl_mpo_coeff(α::Float64, N::Int, M_max::Int; kac_norm = true, show_fit = false)

    if kac_norm == true
        kac = sum([abs(i-j)^(-α) for i ∈ 1:N for j ∈ i+1:N])/(N-1);
    else
        kac = 1;
    end

    J(r) = (1/kac)*(1/r^α);
    J_fit(β, λ, r) = sum(β⋅(λ.^r)); # approximation ∑βᵢλᵢʳ
    ϵ_J(β, λ) = sum([(1 - J_fit(β, λ, r)/J(r))^2 for r in 1:N]); # error function ‖1 - rᵅ ∑βᵢλᵢʳ⁻¹‖

    seed_params = rand(2*M_max);
    in_opt = LBFGS();
    sol = Optim.optimize(parms -> ϵ_J(parms[1:M_max], parms[M_max+1:2*M_max]), seed_params, in_opt; autodiff = :forward);

    βfit = sol.minimizer[1:M_max];
    λfit = sol.minimizer[M_max+1:2*M_max];

    sol_fit = [J_fit(βfit, λfit, r) for r in 1:N];
    diff = abs.(J.(1:N) - sol_fit);

    if show_fit == true
        #plot(J.(1:N), yscale = :log10, label="exact");
        scatter(1:N, J.(1:N), label="exact", legend = :topleft, markershape = :cross, msc = :red);
        scatter!(1:N, sol_fit, label="fit", marker = :xcross, msc = :blue);
        scatter!(twinx(), xticks = :none, 1:N, log10.(diff), label="error",  legend = :none, marker = :circle, ylabel = "log(error)");
        title!("α=$(α)")
        display(current())
    end

    return βfit, λfit, diff
end


N = 20;
Mmax = 5;

#pl_mpo_coeff(1.0, 28, Mmax; kac_norm = false, show_fit = true)
h5open(projectdir("input/pl_Ham_MPO_Mmax=$(Mmax)_L=$(N)_kac_norm.h5"), "w") do pl_MPO;
    for α in 0.5:0.1:2
        create_group(pl_MPO, "alpha=$(α)");
        β, λ, ϵ = pl_mpo_coeff(α, N, Mmax; kac_norm = true, show_fit = false)
        pl_MPO["alpha=$(α)/betas"] = β;
        pl_MPO["alpha=$(α)/lambdas"] = λ;
        pl_MPO["alpha=$(α)/error"] = ϵ;
    end
end
