# Product Monte Carlo (PMC)

PMC estimates expectations under a product of densities. To see how, consider the following setup.

Suppose you can draw samples independently from $K$ distributions on $\mathcal X\subseteq\mathbb R^p$:

$$
X_k \overset{\text{ind.}}{\sim} \pi_k,
\qquad k=1,\ldots,K.
$$

We define the **product distribution** $\pi^\star$ as follows:

$$
\pi^\star(x) =
\frac{\prod_{k=1}^{K}\pi_k(x)}
{\int_{\mathcal X} \prod_{k=1}^{K}\pi_k(u)\thinspace\mathrm{d}u}
\propto \prod_{k=1}^{K}\pi_k(x),
\qquad \forall x \in \mathcal X.
$$

Given a function $h:\mathcal X\to\mathbb R$, we wish to compute its expectation under $\pi^\star$:

$$
\mathbb E_{X\sim\pi^\star}[h(X)] = \int_{\mathcal X}h(x)\thinspace\pi^\star(x)\thinspace\mathrm{d}x.
$$

We call a function $c:\mathcal X^K\to\mathcal X$ a **consensus** function if

$$
c(x,\ldots,x)=x,
\qquad \forall\text{ }x\in\mathcal X.
$$

We call a function $d:\mathcal X^K\to\mathbb R^m$ a **discrepancy** function if, for all $x_1,\ldots,x_K\in\mathcal X$,

$$
d(x_1,\ldots,x_K)=\mathbf 0
\quad\Longleftrightarrow\quad
x_1=\cdots=x_K.
$$

Let $C=c(X_1,\ldots,X_K)$ and $D=d(X_1,\ldots,X_K)$. Then, under suitable regularity conditions, we have that

$$
\mathbb E_{X\sim\pi^\star}[h(X)] = \mathbb E[h(C)\mid D=\mathbf 0_m].
$$

PMC then works as follows:

1. Draw $S$ independent samples per distribution.
2. For each set of $K$ draws, compute its consensus $C$ and discrepancy $D$.
3. Fit a regressor to predict $h(C)$ from $D$.
4. Estimate the target expectation by evaluating the fitted regressor at $D=\mathbf 0_m$.

## Install

To install PMC from PyPI, run:

```bash
pip install product-monte-carlo
```

## Example

The example below shows how to estimate an expectation with PMC.

```python
# Import PyTorch and PMC classes
import torch
from pmc import AnchoredDifferences, Mean, OLS, PMC

# Draw S = 1,000 samples from K = 2 univariate standard Gaussian distributions (p = 1)
samples = torch.randn(1_000, 2, 1)

# Configure PMC to estimate the mean under the product distribution
pmc_estimator = PMC(
    consensus=Mean(),
    discrepancy=AnchoredDifferences(),
    h=lambda x: x,
    regressor=OLS(),
)

# Compute and print the resulting estimate
pmc_estimate = pmc_estimator.estimate(samples)
print(pmc_estimate)
```
