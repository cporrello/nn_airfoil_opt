import torch

"""
The undeformed box is
0.5 - Lx/2  (8)        0.5 - Lx/6  (7)         0.5 + Lx/6  (6)          0.5 + Lx/2  (5)         (y = Ly/2)

0.5 - Lx/2  (1)        0.5 - Lx/6  (2)         0.5 + Lx/6  (3)          0.5 + Lx/2  (4)         (y = -Ly/2)

basis function at node (i)   is Bᵢ   = Φᵢ(x) Ψ₁(y)    (1 ≤ i ≤ 4)
basis function at node (i+4) is Bᵢ₊₄ = Φᵢ(x) Ψ₂(y)    (1 ≤ i ≤ 4)

The map is
(x, y) -> (x, y) + dᵢ Bᵢ(x,  y)"""
def sdesign(theta, x, y, Lx=1.5, Ly=0.2):
    x1, x2, x3, x4 = 0.5 - Lx / 2, 0.5 - Lx / 6, 0.5 + Lx / 6, 0.5 + Lx / 2
    y1, y2 = - Ly / 2, Ly / 2

    phi1 = (x - x2) * (x - x3) * (x - x4) / ((x1 - x2) * (x1 - x3) * (x1 - x4))
    phi2 = (x - x1) * (x - x3) * (x - x4) / ((x2 - x1) * (x2 - x3) * (x2 - x4))
    phi3 = (x - x1) * (x - x2) * (x - x4) / ((x3 - x1) * (x3 - x2) * (x3 - x4))
    phi4 = (x - x1) * (x - x2) * (x - x3) / ((x4 - x1) * (x4 - x2) * (x4 - x3))

    psi1 = (y - y2) / (y1 - y2)
    psi2 = (y - y1) / (y2 - y1)

    B = torch.stack([phi2 * psi1, phi3 * psi1, phi4 * psi1, phi4 * psi2, phi3 * psi2, phi2 * psi2, phi1 * psi2], dim=0)
    return x, y + torch.matmul(theta, B)