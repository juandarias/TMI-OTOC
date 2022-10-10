#### Operator density with exact unitary ####
#############################################
using MKL

using DrWatson
@quickactivate
push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using SparseArrays
using HDF5
using BSON
using TensorOperations
using UnicodePlots: barplot, stairs, scatterplot
using Term
using ExponentialAction

using dmrg_methods
using operators_basis
include(srcdir("hamiltonians.jl"));
include(srcdir("observables.jl"));

using argparser
args_dict = collect_args(ARGS)

##### Parameters #####
######################

## System
const alpha = get_param!(args_dict, "alpha", 1.1);
const N = get_param!(args_dict, "N", 10);
const J = get_param!(args_dict, "J", -1.0);
const Bx = get_param!(args_dict, "Bx", 1.05);
const Bz = get_param!(args_dict, "Bz", -0.5);
const ti = get_param!(args_dict, "ti", 0.1);
const tf = get_param!(args_dict, "tf", 2.5);
const dt = get_param!(args_dict, "dt", 0.1);

## Compression
const tol_compr_min = get_param!(args_dict, "tol_compr_min", 1e-8);
const tol_compr_max = get_param!(args_dict, "tol_compr_max", 1000 * tol_compr_min);
const Dmax = get_param!(args_dict, "Dmax", 2^50);


## Build Hamiltonian

function JPL(α, L)
    J = zeros(L,L);
    for i ∈ 1:L, j ∈ i+1:L
        J[i,j] = abs(i-j)^(-α)
    end
    kacn = sum(J)/(L-1)
    J += transpose(J)
    #return J/kacn
    return J
end

function rho_exact(H_exact, dt, step_initial, step_final, tol_range, outfile; kwargs...)

    L = Int(log2(size(H_exact, 1)));
    rho_s = zeros(length(tol_range), step_final - step_initial + 1, L);
    e_bond = zeros(length(tol_range), step_final - step_initial + 1, L - 1);
    D_t = zeros(length(tol_range), step_final - step_initial + 1, L - 1);
    spectra_bond = zeros(length(tol_range), step_final - step_initial + 1, L - 1, Int(4^(L/2)));

    for (i, step) ∈ enumerate(step_initial:step_final)
        t_sim = round(step * dt, digits = 2)
        println("Calculating unitary and W($(t_sim))")

        #= Instead of calculating the exponential at every step,  one can recursively
        calculate U(t+dt) = U(t)*U(t), requiring only matrix multiplications =#
        #U_exact = exp(-im * step * dt * H_exact);
        #U_exact = fastExpm(-im * step * dt * H_exact, threshold  = 1e-6, nonzero_tol = 1e-14);
        X = expv(im * step * dt, H_exact, σʸᵢ(L, L));
        W_t_exact = expv(im * step * dt, H_exact, X');
        flush(stdout)
        #W_t_exact = U_exact' * (σʸᵢ(L, L) * U_exact);
        #U_exact_mpo = operator_to_mpo(U_exact);
        #U_t = deepcopy(U_exact_mpo);
        #W_t = calc_Wt(U_exact_mpo, Y, L);
        W_t_mps = cast_mps(operator_to_mpo(W_t_exact, ϵmax=tol_range[1]));
        flush(stdout)
        println("Canonizing W($(t_sim))")
        sweep_qr!(W_t_mps, final_site = 1, direction = "right");

        for (n, tol_compr) ∈ enumerate(tol_range)

            print("\n-> Compressing W($(t_sim)) with ϵ = $(tol_compr). ")
            W_t_compr = deepcopy(W_t_mps);
            ϵ_svd, svs =  sweep_svd!(W_t_compr, spectrum = true, direction= "left", ϵmax = tol_compr; kwargs...); #* MPS is normalized
            print("Total compression error is $(ϵ_svd)\n")

            [spectra_bond[n, i, s, 1 : length(svs[s])] = svs[s] for s ∈ 1:L - 1];
            e_bond[n, i, :] = entanglement_entropy.(svs);
            rho_s[n, i, :] = operator_density(cast_mpo(W_t_compr); normalized = true);
            D_t[n, i, :] = W_t_compr.D;

        end

        ## Generates plots of the bond entropy and operator density at every step
        bond_entropy = barplot(D_t[1, i, :], e_bond[1, i, :], xlabel = "S", ylabel = "D", title = "Entanglement entropy per bond", compact = true);
        op_den = barplot(1:L, abs.(rho_s[1, i, :]), xlabel = "p", ylabel = "Site", title = "Operator density", compact = true, xflip = true)
        spec_plot = scatterplot(1 : L - 1, spectra_bond[1, i, :, :] .+ 1e-10, yscale = :log10, ylabel = "σ", xlabel = "D", title = "Entanglement spectrum", name = "", color = :blue)
        if i > 1
            change_entropy = stairs(1:L, append!((e_bond[1, i, :] - e_bond[1, i - 1, :])./e_bond[1, i - 1, :], 0), xlabel = "B", ylabel = "ΔS/S", title = "Relative entanglement change per bond", compact = true);
            display(print(panel(bond_entropy) * panel(op_den) / (panel(spec_plot) * panel(change_entropy, fit_plot = true))));
        elseif i == 1
            display(print(panel(bond_entropy) * panel(op_den) / panel(spec_plot)));
        end


        ## Export data
        h5open(datadir("exact/op_density/$(outfile)_t=$(t_sim).h5"), "w") do rho_results;
            create_group(rho_results, "rho");
            create_group(rho_results, "entropy");
            create_group(rho_results, "D");
            create_group(rho_results, "spectrum");

            for (n, tol_compr) ∈ enumerate(tol_range)
                rho_results["rho/tol_$(tol_compr)"] = rho_s[n, i, :];
                rho_results["entropy/tol_$(tol_compr)"] = e_bond[n, i, :];
                rho_results["D/tol_$(tol_compr)"] = D_t[n, i, :];
                rho_results["spectrum/tol_$(tol_compr)"] = spectra_bond[n, i, :, :];
            end
        end

    end


    #return nothing
end

panel(plot; fit_plot::Bool = true, kw...) = Panel(string(plot, color = true); fit = fit_plot, kw...)


function main()

    ## Save parameters
    outfile = "rho_exact_N=$(N)_alpha=$(alpha)_tf=$(tf)";
    bson(datadir("exact/op_density/$(outfile).bson"), args_dict);

    ## Build sparse Hamiltonian
    println("Calculating Hamiltonian")
    Jij = 4 * J * JPL(alpha, N);
    TF = TransverseFieldIsing(Jij, [2 * Bx]);
    TF_M = sparse(TF.row_ixs, TF.col_ixs, TF.nz_vals);
    Hlong = 2 * Bz * spdiagm(sum([diag(Sᶻᵢ(i, N)) for i in 1 : N]));
    H_TMI = TF_M + Hlong;

    println("Done Calculating Hamiltonian")
    flush(stdout)

    ## Calculate range calculation
    initial_step = Int(floor(ti/dt));
    final_step = Int(floor(tf/dt));

    if tol_compr_min != 0.0
        r_max = floor(Int, log10(tol_compr_max/tol_compr_min));
        tol_range = tol_compr_min * vcat([[10^n] for n ∈ 0:r_max]...);
    else
        tol_range = 0.0;
    end


    ## Calculate operator density
    rho_exact(H_TMI, dt, initial_step, final_step, tol_range, outfile; Dmax = Dmax);

end


main()
