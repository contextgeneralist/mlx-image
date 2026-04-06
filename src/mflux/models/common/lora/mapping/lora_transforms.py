import mlx.core as mx

# Flux2 single-block fused linear dimensions.
# hidden_dim = num_attention_heads * attention_head_dim.
#   3B variant: 24 heads * 128 = 3072
#   9B variant: 32 heads * 128 = 4096
# The fused single-block linear1 combines Q, K, V, and an MLP gate (ratio 6×hidden):
#   3B: 3*3072 + 6*3072 = 9216 + 18432 = 27648
#   9B: 3*4096 + 6*4096 = 12288 + 24576 = 36864
_FLUX2_FUSED_LINEAR1_3B = 27648
_FLUX2_FUSED_LINEAR1_9B = 36864
_FLUX2_HIDDEN_3B = 3072
_FLUX2_HIDDEN_9B = 4096


class LoraTransforms:
    # --- Double-block QKV split (up direction) ---

    @staticmethod
    def split_q_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 0)

    @staticmethod
    def split_k_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 1)

    @staticmethod
    def split_v_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 2)

    # --- Double-block QKV split (down direction) ---
    # The rank dimension is not split for AIToolkit-format adapters;
    # the full down matrix is used as-is for each Q/K/V target.

    @staticmethod
    def split_q_down(tensor: mx.array) -> mx.array:
        return tensor

    @staticmethod
    def split_k_down(tensor: mx.array) -> mx.array:
        return tensor

    @staticmethod
    def split_v_down(tensor: mx.array) -> mx.array:
        return tensor

    # --- Single-block QKV+MLP split (up direction) ---

    @staticmethod
    def split_single_q_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 0)

    @staticmethod
    def split_single_k_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 1)

    @staticmethod
    def split_single_v_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 2)

    @staticmethod
    def split_single_mlp_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_up(tensor, 3)

    # --- Single-block QKV+MLP split (down direction) ---
    # The rank dimension is not split for AIToolkit-format adapters.

    @staticmethod
    def split_single_q_down(tensor: mx.array) -> mx.array:
        return tensor

    @staticmethod
    def split_single_k_down(tensor: mx.array) -> mx.array:
        return tensor

    @staticmethod
    def split_single_v_down(tensor: mx.array) -> mx.array:
        return tensor

    @staticmethod
    def split_single_mlp_down(tensor: mx.array) -> mx.array:
        return tensor

    # --- Flux2 single-block cross-size padding ---
    # These pad adapters trained on one Flux2 variant (3B or 9B) to match the
    # layer shape of the other, padding with zeros on the mismatched dimension.

    @staticmethod
    def pad_flux2_single_linear1_up(tensor: mx.array) -> mx.array:
        """Pad the output dimension (axis 0) of linear1 to the 9B fused size.

        Handles both the full delta_w matrix (27648 → 36864) and the w1 factor
        row count (96 → 128), where 96 = _FLUX2_HIDDEN_3B // 32 and
        128 = _FLUX2_HIDDEN_9B // 32.
        """
        _w1_rows_3b = _FLUX2_HIDDEN_3B // 32   # 96
        _w1_rows_9b = _FLUX2_HIDDEN_9B // 32   # 128

        if tensor.shape[0] == _FLUX2_FUSED_LINEAR1_9B:
            return tensor
        if tensor.shape[0] == _FLUX2_FUSED_LINEAR1_3B:
            padding_amount = _FLUX2_FUSED_LINEAR1_9B - _FLUX2_FUSED_LINEAR1_3B
        elif tensor.shape[0] == _w1_rows_3b:
            padding_amount = _w1_rows_9b - _w1_rows_3b
        else:
            return tensor
        padding = mx.zeros((padding_amount, *tensor.shape[1:]), dtype=tensor.dtype)
        return mx.concatenate([tensor, padding], axis=0)

    @staticmethod
    def pad_flux2_single_linear1_down(tensor: mx.array) -> mx.array:
        """Pad the input dimension of linear1 to the 9B hidden size (4096).

        Handles both 2-D tensors (checks axis 1) and 1-D / factor tensors
        (checks axis 0).
        """
        if tensor.ndim == 2:
            if tensor.shape[1] == _FLUX2_HIDDEN_9B:
                return tensor
            if tensor.shape[1] == _FLUX2_HIDDEN_3B:
                padding_amount = _FLUX2_HIDDEN_9B - _FLUX2_HIDDEN_3B
                padding = mx.zeros((tensor.shape[0], padding_amount), dtype=tensor.dtype)
                return mx.concatenate([tensor, padding], axis=1)
        if tensor.shape[0] == _FLUX2_HIDDEN_9B:
            return tensor
        if tensor.shape[0] == _FLUX2_HIDDEN_3B:
            padding_amount = _FLUX2_HIDDEN_9B - _FLUX2_HIDDEN_3B
            padding = mx.zeros((padding_amount, *tensor.shape[1:]), dtype=tensor.dtype)
            return mx.concatenate([tensor, padding], axis=0)
        return tensor

    @staticmethod
    def pad_flux2_single_linear2_up(tensor: mx.array) -> mx.array:
        """Pad or slice the output dimension (axis 0) of linear2 (to_out) to the 9B size (4096).

        Handles:
          - Slicing 16384 → 4096 (square delta_w case)
          - Padding 3072 → 4096 (3B → 9B case)
          - w1 factor row counts: 32 (9B) and 96 (3B)
        """
        _out_9b = _FLUX2_HIDDEN_9B     # 4096
        _out_3b = _FLUX2_HIDDEN_3B     # 3072
        _w1_rows_9b = _FLUX2_HIDDEN_9B // 128   # 32
        _w1_rows_3b = _FLUX2_HIDDEN_3B // 32    # 96

        if tensor.shape[0] == _out_9b:
            return tensor

        # Square delta_w case: slice 16384 to 4096
        if tensor.shape[0] == 16384:
            return tensor[:_out_9b, ...]

        # Padding case
        if tensor.shape[0] == _out_3b:
            padding_amount = _out_9b - _out_3b
        elif tensor.shape[0] == _w1_rows_3b:
            padding_amount = 128 - _w1_rows_3b  # Pad to 128 for 9B kron
        else:
            return tensor

        padding = mx.zeros((padding_amount, *tensor.shape[1:]), dtype=tensor.dtype)
        return mx.concatenate([tensor, padding], axis=0)

    @staticmethod
    def pad_flux2_single_linear2_down(tensor: mx.array) -> mx.array:
        """Pad the input dimension of linear2 (to_out) to the 9B fused size.

        The linear2 input is the output of linear1 (fused QKV+MLP).
        Handles both the full delta_w matrix and w1 factor row counts for
        adapters targeting different input sizes of the to_out layer.
        """
        # Fused linear1 output dimensions used as linear2 input
        _linear2_in_9b = 16384   # kron result input for standard 9B to_out adapters
        _linear2_in_3b = 12288   # kron result input for standard 3B to_out adapters

        # w1 factor row counts for to_out: 32 (9B) and 96 (3B)
        _w1_rows_9b = _FLUX2_HIDDEN_9B // 128   # 32
        _w1_rows_3b = _FLUX2_HIDDEN_3B // 32    # 96

        if tensor.ndim == 2:
            if tensor.shape[1] == _linear2_in_9b:
                return tensor
            if tensor.shape[1] == _linear2_in_3b:
                padding_amount = _linear2_in_9b - _linear2_in_3b
                padding = mx.zeros((tensor.shape[0], padding_amount), dtype=tensor.dtype)
                return mx.concatenate([tensor, padding], axis=1)
            if tensor.shape[1] == _FLUX2_HIDDEN_9B:
                padding_amount = _linear2_in_9b - _FLUX2_HIDDEN_9B
                padding = mx.zeros((tensor.shape[0], padding_amount), dtype=tensor.dtype)
                return mx.concatenate([tensor, padding], axis=1)

        # Axis-0 path for w1 factors and 1-D tensors
        if tensor.shape[0] == _linear2_in_9b:
            return tensor
        if tensor.shape[0] == _linear2_in_3b:
            padding_amount = _linear2_in_9b - _linear2_in_3b
        elif tensor.shape[0] == _FLUX2_HIDDEN_9B:
            padding_amount = _linear2_in_9b - _FLUX2_HIDDEN_9B
        elif tensor.shape[0] == _w1_rows_9b:   # 32 → 128
            padding_amount = 128 - _w1_rows_9b
        elif tensor.shape[0] == _w1_rows_3b:   # 96 → 128
            padding_amount = 128 - _w1_rows_3b
        else:
            return tensor
        padding = mx.zeros((padding_amount, *tensor.shape[1:]), dtype=tensor.dtype)
        return mx.concatenate([tensor, padding], axis=0)

    # --- Internal helpers ---

    @staticmethod
    def _split_qkv_up(tensor: mx.array, index: int, num_splits: int = 3) -> mx.array:
        # Only split if it's a fused QKV shape (9B: 4096*3=12288, 3B: 3072*3=9216)
        if tensor.shape[0] not in [12288, 9216]:
            return tensor
        split_size = tensor.shape[0] // num_splits
        start = index * split_size
        end = start + split_size
        return tensor[start:end, :]

    @staticmethod
    def _split_qkv_mlp_up(tensor: mx.array, index: int, dims: list[int] | None = None) -> mx.array:
        # Only split if it's a fused single-block shape
        if tensor.shape[0] not in [_FLUX2_FUSED_LINEAR1_9B, _FLUX2_FUSED_LINEAR1_3B]:
            return tensor
        if dims is None:
            h = _FLUX2_HIDDEN_9B if tensor.shape[0] == _FLUX2_FUSED_LINEAR1_9B else _FLUX2_HIDDEN_3B
            dims = [h, h, h, h * 6]  # Q, K, V, MLP (6× = 2 × mlp_ratio=3)
        start = sum(dims[:index])
        end = start + dims[index]
        return tensor[start:end, :]
