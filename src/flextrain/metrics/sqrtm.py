import time
import torch
from typing import Tuple


def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 200, tol: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Square root of matrix using Newton-Schulz Iterative method

    See https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    
    Parameters:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    
    Returns:
        Square root of matrix
    """
    expected_num_dims = 2
    if matrix.dim() != expected_num_dims:
        raise ValueError(f'Input dimension equals {matrix.dim()}, expected {expected_num_dims}')

    if num_iters <= 0:
        raise ValueError(f'Number of iteration equals {num_iters}, expected greater than 0')

    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, requires_grad=False).to(matrix)
    Z = torch.eye(dim, dim, requires_grad=False).to(matrix)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1).to(matrix)

    for n in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.]).to(error), atol=tol):
            break

    return s_matrix
