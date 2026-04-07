import mlx.core as mx

LOKR_WEIGHT_KEYS = ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_t2", "dora_scale")

def reconstruct_lokr_delta(
    *,
    lokr_w1=None, lokr_w2=None,
    lokr_w1_a=None, lokr_w1_b=None,
    lokr_w2_a=None, lokr_w2_b=None,
    lokr_t2=None,
) -> tuple[mx.array | None, int | None]:
    """Reconstruct LoKr delta_w and rank from factor matrices.
    Returns (delta_w, rank) or (None, None) if factors are missing.
    """
    # 1. Reconstruct W2
    rank = None
    if lokr_w2 is not None:
        w2 = lokr_w2
    elif lokr_w2_a is not None and lokr_w2_b is not None:
        if lokr_t2 is not None:
            # Tucker decomposition
            w2 = mx.einsum("ijkl, jr, ip -> prkl", lokr_t2, lokr_w2_b, lokr_w2_a)
        else:
            w2 = mx.matmul(lokr_w2_a, lokr_w2_b)
        rank = lokr_w2_b.shape[0]
    else:
        w2 = None

    # 2. Reconstruct W1
    if lokr_w1 is not None:
        w1 = lokr_w1
    elif lokr_w1_a is not None and lokr_w1_b is not None:
        w1 = mx.matmul(lokr_w1_a, lokr_w1_b)
        if rank is None:
            rank = lokr_w1_b.shape[0]
    else:
        w1 = None

    if w1 is None or w2 is None:
        return None, None

    # 3. Align dimensions for Conv layers
    if len(w2.shape) == 4:
        w1 = mx.expand_dims(mx.expand_dims(w1, 2), 3)

    delta_w = mx.kron(w1, w2)
    return delta_w, rank
