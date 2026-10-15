"""Minimal compatibility namespace for ATPO's optional padding helpers.

This does not provide FlashAttention CUDA kernels.  It only exposes the
``bert_padding`` helpers imported unconditionally by the upstream VeRL code.
"""

__version__ = "compat"
