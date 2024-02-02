"""
Meta-Learning for Fast and Accurate Domain Adaptation for Irregular Tensors
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

import torch

def kron(A, B):
    """
    Compute kronecker product of A and B
    """
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def khatri(A, B):
    """
    Compute Khatri-Rao product of A and B
    """
    return torch.cat([kron(A[:, i].unsqueeze(1), B[:, i].unsqueeze(1)) for i in range(A.size(1))], 1)
