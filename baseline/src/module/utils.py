import warnings
import torch
import math
from typing import *

__all__ = [
    "cuda_if_available",
    "log1mexp",
]

def cuda_if_available(use_cuda: Optional[bool] = None) -> torch.device:
    cuda_available = torch.cuda.is_available()
    _use_cuda = (use_cuda is None or use_cuda) and cuda_available
    if use_cuda is True and not cuda_available:
        warnings.warn("Requested CUDA but it is not available, running on CPU")
    if use_cuda is False and cuda_available:
        warnings.warn(
            "Running on CPU, even though CUDA is available. "
            "(This is likely not desired, check your arguments.)"
        )
    return torch.device("cuda" if _use_cuda else "cpu")


_log1mexp_switch = math.log(0.5)

def log1mexp(
    x: torch.Tensor,
    split_point: float = _log1mexp_switch,
    exp_zero_eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).
    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).
    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
    Args:
        x: input tensor
        split_point: Should be kept to the default of log(0.5)
        exp_zero_eps: Default 1e-7
    Returns:
        torch.Tensor: Elementwise log1mexp(x) = log(1-exp(x))
    """
    logexpm1_switch = x > split_point
    Z = torch.zeros_like(x)
    # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
    # ie. p=1
    logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
    Z[logexpm1_switch] = logexpm1.detach() + (
        logexpm1_bw - logexpm1_bw.detach()
    )
    # Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
    Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

    return Z