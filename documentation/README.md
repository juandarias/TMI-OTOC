# Folder structure

The most relevant code is found in the following folders.

## Scripts

Under the folder scripts you can find some useful scripts:

* `compressor_Wt_2_order.jl` : Calculates $\mathcal{W}(t)$ using the $W_{II}$ approximation and a second-order Trotter-Suzuki expansion. The error of the approximation of the unitary scales as $\mathcal{O}(Lt^3)$. It uses a SVD compression to reduce the tensor dimensions of $\mathcal{W}(t)$ and if desired, a variational optimization is done afterwards to improve the compressed tensors. The error tolerance of these two steps are controlled by the parameters `eps_svd` and `tol_compr` respectively. Furthermore, the operator density $\rho_\ell(t)$ is calculated at every step. *Open issues*: the memory requirements of the method are high and might require a small step size for convergence.
* `compressor_Wt_2_order_SVD.jl` : Similar to `compressor_Wt_2_order.jl`, but calculates the SVD truncation on the fly, i.e. while updating each site tensor of $\mathcal{W}(t)$. Should lead to lower-memory cost and faster speed.
* `operator_density_exact_fast.jl` : Calculates $\mathcal{W}(t)$ using the exact unitary $\mathcal{U}(t)$. At every time step, the operator density $\rho_\ell(t)$, the entanglement entropy $S_A$, and the entanglement spetrum are calculated and plotted. This script is fast as it requires no minimal time-step, but is limited to system sizes for which $\mathcal{U}(t)$ can be calculated, which is currently for $N=14$
* `pl_MPO_fits.jl` : Generates the coefficients $\lambda, \beta$ that approximate the power-law decay of the couplings by minimizing the difference $\lVert\frac{1}{r^\alpha} - \sum_i \beta_i \lambda_i^{(r-1)} \rVert$


## Src

Contains all functions and methods.

## Input

Contains files with the coefficients $\lambda, \beta$ to approximate the power-law couplings.

# Methods

## `mps_methods.jl`

* `vector_to_mps` : Converts any ket into a MPS with a maximum bond dimension `Dmax`

* `reduced_density_matrix` : For a given state $|\Psi\rangle$, calculates the reduced density matrix of the contiguous partition between sites `loc_start` and `loc_end`. The contraction is done from left to right thus the method works optimally for states when the entanglement decreases in the same direction