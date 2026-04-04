import mlx.core as mx


class LoraTransforms:
    @staticmethod
    def split_q_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 0)

    @staticmethod
    def split_k_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 1)

    @staticmethod
    def split_v_up(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_up(tensor, 2)

    @staticmethod
    def split_q_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_down(tensor, 0)

    @staticmethod
    def split_k_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_down(tensor, 1)

    @staticmethod
    def split_v_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_down(tensor, 2)

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

    @staticmethod
    def split_single_q_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 0)

    @staticmethod
    def split_single_k_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 1)

    @staticmethod
    def split_single_v_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 2)

    @staticmethod
    def split_single_mlp_down(tensor: mx.array) -> mx.array:
        return LoraTransforms._split_qkv_mlp_down(tensor, 3)

    @staticmethod
    def pad_klein9b_single_linear1_up(tensor: mx.array) -> mx.array:
        if tensor.shape[0] == 36864:
            return tensor
        if tensor.shape[0] == 27648:
            padding_amount = 9216
        elif tensor.shape[0] == 96:
            padding_amount = 32
        else:
            return tensor
        padding = mx.zeros((padding_amount, *tensor.shape[1:]), dtype=tensor.dtype)
        return mx.concatenate([tensor, padding], axis=0)

    @staticmethod
    def pad_klein9b_single_linear1_down(tensor: mx.array) -> mx.array:
        if tensor.ndim == 2:
            if tensor.shape[1] == 4096:
                return tensor
            if tensor.shape[1] == 3072:
                padding_amount = 1024
                padding = mx.zeros((tensor.shape[0], padding_amount), dtype=tensor.dtype)
                return mx.concatenate([tensor, padding], axis=1)
        if tensor.shape[0] == 4096:
            return tensor
        if tensor.shape[0] == 3072:
            padding_amount = 1024
            padding = mx.zeros((padding_amount, *tensor.shape[1:]), dtype=tensor.dtype)
            return mx.concatenate([tensor, padding], axis=0)
        return tensor

    @staticmethod
    def pad_klein9b_single_linear2_down(tensor: mx.array) -> mx.array:
        if tensor.ndim == 2:
            if tensor.shape[1] == 16384:
                return tensor
            if tensor.shape[1] == 12288:
                padding_amount = 4096
                padding = mx.zeros((tensor.shape[0], padding_amount), dtype=tensor.dtype)
                return mx.concatenate([tensor, padding], axis=1)
            if tensor.shape[1] == 4096:
                padding_amount = 12288
                padding = mx.zeros((tensor.shape[0], padding_amount), dtype=tensor.dtype)
                return mx.concatenate([tensor, padding], axis=1)
        if tensor.shape[0] == 16384:
            return tensor
        if tensor.shape[0] == 12288:
            padding_amount = 4096
        elif tensor.shape[0] == 4096:
            padding_amount = 12288
        elif tensor.shape[0] == 32:
            padding_amount = 96
        elif tensor.shape[0] == 96:
            padding_amount = 32
        else:
            return tensor
        padding = mx.zeros((padding_amount, *tensor.shape[1:]), dtype=tensor.dtype)
        return mx.concatenate([tensor, padding], axis=0)

    @staticmethod
    def _transpose(tensor: mx.array) -> mx.array:
        return tensor.T

    @staticmethod
    def _split_qkv_up(tensor: mx.array, index: int, num_splits: int = 3) -> mx.array:
        # Only split if it's a fused shape (e.g., 12288 or 9216)
        # 9B: 4096*3 = 12288. 4B: 3072*3 = 9216.
        if tensor.shape[0] not in [12288, 9216]:
            return tensor
        split_size = tensor.shape[0] // num_splits
        start = index * split_size
        end = start + split_size
        return tensor[start:end, :]

    @staticmethod
    def _split_qkv_down(tensor: mx.array, index: int, num_splits: int = 3) -> mx.array:
        # We don't split Rank for AIToolkit.
        return tensor

    @staticmethod
    def _split_qkv_mlp_up(tensor: mx.array, index: int, dims: list[int] | None = None) -> mx.array:
        # Only split if it's a fused shape (36864 or 27648)
        if tensor.shape[0] not in [36864, 27648]:
            return tensor
        if dims is None:
            if tensor.shape[0] == 36864:
                dims = [4096, 4096, 4096, 24576]
            else:
                dims = [3072, 3072, 3072, 18432]
        start = sum(dims[:index])
        end = start + dims[index]
        return tensor[start:end, :]

    @staticmethod
    def _split_qkv_mlp_down(tensor: mx.array, index: int, num_splits: int = 4) -> mx.array:
        return tensor
