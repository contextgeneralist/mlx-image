import warnings

import mlx.core as mx
from mlx import nn

from mflux.models.common.lora.layer.lokr_reconstruct import reconstruct_lokr_delta


class LokrLinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear | nn.QuantizedLinear,
        scale: float = 1.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            # QuantizedLinear packs values at `bits`-per-element into 32-bit words
            input_dims *= 32 // linear.bits
        lokr_lin = LokrLinear(
            input_dims=input_dims,
            output_dims=output_dims,
            scale=scale,
        )
        lokr_lin.linear = linear
        return lokr_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        self._mflux_lora_role: str | None = None

        # Pre-computed Kronecker product delta, set by the loader.
        # Individual factors (lokr_w1, lokr_w2, etc.) may also be set directly
        # for unit-test use or when bypassing the full loader pipeline.
        self.delta_w = None
        self.lokr_w1 = None
        self.lokr_w2 = None
        self.lokr_w1_a = None
        self.lokr_w1_b = None
        self.lokr_w2_a = None
        self.lokr_w2_b = None
        self.lokr_t2 = None

    def compute_delta(self, x):
        if self.delta_w is not None:
            # Fast path: use pre-computed delta from the loader.
            # LoKr delta_w is stored in (In, Out) orientation by preference.
            # MLX linear layer weights are stored as (Out, In).
            if self.delta_w.shape == self.linear.weight.shape[::-1]:
                return self.scale * mx.matmul(x, self.delta_w)
            elif self.delta_w.shape == self.linear.weight.shape:
                return self.scale * mx.matmul(x, self.delta_w.T)
            else:
                # Fallback for other shapes (e.g. Conv)
                return self.scale * mx.matmul(x, self.delta_w)

        # Slow path: compute the Kronecker product from individual factors.
        # This path is used when factors are set directly (e.g. in unit tests).
        delta_w, _ = reconstruct_lokr_delta(
            lokr_w1=self.lokr_w1,
            lokr_w2=self.lokr_w2,
            lokr_w1_a=self.lokr_w1_a,
            lokr_w1_b=self.lokr_w1_b,
            lokr_w2_a=self.lokr_w2_a,
            lokr_w2_b=self.lokr_w2_b,
            lokr_t2=self.lokr_t2,
        )

        if delta_w is None:
            warnings.warn(
                "LokrLinear has no usable weights. Returning base output only.",
                stacklevel=2,
            )
            return 0

        if delta_w.shape == self.linear.weight.shape:
            return self.scale * mx.matmul(x, delta_w.T)
        else:
            return self.scale * mx.matmul(x, delta_w)

    def __call__(self, x):
        return self.linear(x) + self.compute_delta(x)
