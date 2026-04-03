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

        # These will be populated by the loader
        self.lokr_w1 = None
        self.lokr_w2 = None
        self.lokr_w1_a = None
        self.lokr_w1_b = None
        self.lokr_w2_a = None
        self.lokr_w2_b = None
        self.lokr_t2 = None

    def __call__(self, x):
        base_out = self.linear(x)

        # 1. Compute W1
        if self.lokr_w1 is not None:
            w1 = self.lokr_w1
        elif self.lokr_w1_a is not None and self.lokr_w1_b is not None:
            w1 = mx.matmul(self.lokr_w1_a, self.lokr_w1_b)
        else:
            return base_out

        # 2. Compute W2
        if self.lokr_t2 is not None and self.lokr_w1_a is not None:
            # Handle t2 (CP decomposition style often found in LyCORIS)
            # This is a simplified version, usually involves einsum or specific reshapes
            # For now, we materialise or handle as standard if t2 is missing
            w2 = self.lokr_w2 if self.lokr_w2 is not None else None
        elif self.lokr_w2 is not None:
            w2 = self.lokr_w2
        elif self.lokr_w2_a is not None and self.lokr_w2_b is not None:
            w2 = mx.matmul(self.lokr_w2_a, self.lokr_w2_b)
        else:
            return base_out

        if w2 is None:
            return base_out

        # 3. Apply Kronecker Product delta
        # Delta W = w1 ⊗ w2
        # For linear layers: y = x (w1 ⊗ w2)^T
        delta_w = mx.kron(w1, w2)
        
        # Ensure delta_w matches the expected shape (output_dims, input_dims)
        # LyCORIS/AIToolkit might need a transpose depending on how they were saved
        if delta_w.shape != self.linear.weight.shape:
             # Try to reshape or transpose if there's a mismatch
             # In most cases, mx.kron(w1, w2) should result in (out, in)
             pass

        lokr_out = mx.matmul(x, delta_w.T)
        return base_out + self.scale * lokr_out
