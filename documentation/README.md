The most relevant code is found in the following folders.

## Scripts

Under the folder scripts you can find some useful scripts:

* `compressor_Wt_2_order.jl` : Calculates $\mathcal{W}(t)$ using the $W_{II}$ approximation and a second-order Trotter-Suzuki expansion. The error of the approximation of the unitary scales as $\mathcal{O}(Lt^3)$. It uses a SVD compression to reduce the tensor dimensions of $\mathcal{W}(t)$ and if desired, a variational optimization is done afterwards to improve the compressed tensors. The error tolerance of these two steps are controlled by the parameters `eps_svd` and `tol_compr` respectively. Furthermore, the operator density $\rho_\ell(t)$ is calculated at every step. *Open issues*: the memory requirements of the method are high and might require a small step size for convergence.
* `operator_density_exact_fast.jl` : Calculates $\mathcal{W}(t)$ using the exact unitary $\mathcal{U}(t)$. At every time step, the operator density $\rho_\ell(t)$, the entanglement entropy $S_A$, and the entanglement spetrum are calculated and plotted. This script is fast as it requires no minimal time-step, but is limited to system sizes for which $\mathcal{U}(t)$ can be calculated, which is currently for $N=14$
* `pl_MPO_fits.jl` : Generates the coefficients $\lambda, \beta$ that approximate the power-law decay of the couplings by minimizing the difference $\lVert\frac{1}{r^\alpha} - \sum_i \beta_i \lambda_i^{(r-1)} \rVert$

## Src

Contains all functions and methods.

## Input

Contains files with the coefficients $\lambda, \beta$ to approximate the power-law couplings.

