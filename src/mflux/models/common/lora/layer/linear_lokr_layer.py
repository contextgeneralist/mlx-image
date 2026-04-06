import warnings

import mlx.core as mx
from mlx import nn


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

    def __call__(self, x):
        base_out = self.linear(x)

        if self.delta_w is not None:
            # Fast path: use pre-computed delta from the loader.
            # LoKr delta_w is stored in (In, Out) orientation by preference.
            # MLX linear layer weights are stored as (Out, In).
            if self.delta_w.shape == self.linear.weight.shape[::-1]:
                return base_out + self.scale * mx.matmul(x, self.delta_w)
            elif self.delta_w.shape == self.linear.weight.shape:
                return base_out + self.scale * mx.matmul(x, self.delta_w.T)
            else:
                # Fallback for other shapes (e.g. Conv)
                return base_out + self.scale * mx.matmul(x, self.delta_w)

        # Slow path: compute the Kronecker product from individual factors.
        # This path is used when factors are set directly (e.g. in unit tests).
        # 1. Reconstruct W1
        if self.lokr_w1 is not None:
            w1 = self.lokr_w1
        elif self.lokr_w1_a is not None and self.lokr_w1_b is not None:
            w1 = mx.matmul(self.lokr_w1_a, self.lokr_w1_b)
        else:
            warnings.warn(
                "LokrLinear has no usable weights (delta_w, lokr_w1, or lokr_w1_a/b). "
                "Returning base output only.",
                stacklevel=2,
            )
            return base_out

        # 2. Reconstruct W2
        if self.lokr_w2 is not None:
            w2 = self.lokr_w2
        elif self.lokr_w2_a is not None and self.lokr_w2_b is not None:
            if self.lokr_t2 is not None:
                # Tucker decomposition
                w2 = mx.einsum("ijkl, jr, ip -> prkl", self.lokr_t2, self.lokr_w2_b, self.lokr_w2_a)
            else:
                w2 = mx.matmul(self.lokr_w2_a, self.lokr_w2_b)
        else:
            warnings.warn(
                "LokrLinear has no usable w2 weights (lokr_w2, lokr_w2_a/b, or lokr_t2). "
                "Returning base output only.",
                stacklevel=2,
            )
            return base_out

        # 3. Align dimensions for Conv layers
        if len(w2.shape) == 4:
            w1 = mx.expand_dims(mx.expand_dims(w1, 2), 3)

        delta_w = mx.kron(w1, w2)
        if delta_w.shape == self.linear.weight.shape:
            return base_out + self.scale * mx.matmul(x, delta_w.T)
        else:
            return base_out + self.scale * mx.matmul(x, delta_w)
