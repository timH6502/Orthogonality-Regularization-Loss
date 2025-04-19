# Orthogonality-Regularization-Loss
Orthogonality Regularization Loss

This is a modification the algorithms presented in  
Junlin He, Jinxiao Du, Wei Ma: “Preventing Dimensional Collapse in Self-Supervised Learning via Orthogonality Regularization”, 2024; http://arxiv.org/abs/2411.00392'>arXiv:2411.00392

## Modification

Let $W \in \mathbb{R}^{m \times n}$, and define $G = \begin{cases} W^TW - I \text{ if m > n}\\ WW^T - I \text{ otherwise.} \end{cases}$ Note that $G = G^T$.

The original algorithm is as follows:  
$u^{(k)} = Gv^{(k - 1)}$, $\quad v^{(k)} = Gu^{(k)} = G^2v^{(k - 1)}$.
Then, $$\lim_{k \rightarrow \infty} \frac{\| v^{(k)} \|_2}{\| u^{(k)} \|_2} = \sigma(G),$$ where $\sigma(G)$ denotes the spectral norm (i.e., largest singular value) of $G$.

However, this approach can lead to numerical instability in practice. The following is a stabilized version of the algorithm and demonstrate that it yields the same result. The modified steps are:  
1. $\tilde{u}^{(k)} = G v^{(k - 1)}$  
2. $\sigma^{(k)} = \| \tilde{u}^{(k)} \|_2 = \| Gv^{(k - 1)} \|_2$  
3. $u^{(k)} = \frac{\tilde{u}^{(k)}}{\| \tilde{u}^{(k)} \|_2}$  
4. $v^{(k)} = \frac{Gu^{(k)}}{\| Gu^{(k)} \|_2} = \frac{Gv^{(k - 1)}}{\| Gv^{(k - 1)} \|_2}$.

Since for $k \rightarrow \infty$, $v^{(k)} = v^{(k - 1)}$, we get $\| Gv^{(k - 1)} \|_2 v^{(k)} = Gv^{(k - 1)}$. That means that $\| Gv^{(k - 1)} \|_2$ is the singular value  associated with the eigenvector $v^{(k)} (= v^{(k - 1)})$ of $G$. Moreover, we know that as $k \rightarrow \infty$, $\| Gv^{(k - 1)} \|_2^2 v^{(k)} = G^2v^{(k - 1)} = \| G^2v^{(k - 1)} \|_2v^{(k)}$ (plug in $G^2$ in the beginning of this and you'll obtain the RHS.). From the original algorithm, we get $\sigma(G) = \frac{\| G^2v^{(k - 1)} \|_2}{\| Gv^{(k - 1)} \|_2} = \| Gv^{(k - 1)} \|_2 = \sigma^{(k)}$ as $k \rightarrow \infty$.
