"""Smoke-only native RMSNorm fallback for the old SGLang kernel on SM100."""

import os


if os.environ.get("EA_ATPO_NATIVE_RMSNORM", "0") == "1":
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("EA_ATPO_NATIVE_RMSNORM requires the selected CUDA GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 10:
        raise RuntimeError(
            "EA_ATPO_NATIVE_RMSNORM was enabled outside the intended SM100 path: "
            f"device capability is {capability}"
        )

    from sglang.srt.layers.layernorm import RMSNorm

    # SGLang's native implementation uses the same RMSNorm equation with PyTorch
    # ops. The bundled sgl-kernel==0.2.8 wheel has no SM100 cubin for rmsnorm.
    RMSNorm.forward_cuda = RMSNorm.forward_native
    print(
        "EA-ATPO smoke: enabled SGLang native RMSNorm fallback on SM100",
        flush=True,
    )
