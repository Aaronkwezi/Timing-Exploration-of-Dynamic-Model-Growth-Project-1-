"""Three training-cost proxies, so the conclusions can be checked against the choice of proxy.

All are FLOPs per trained token (forward + backward). Evaluation is tracked separately.

  6pbt    6 * every parameter. The original proxy; counts the token/position embedding
          tables as matmul work although a lookup is nearly free.
  matmul  6 * parameters that sit in a matrix multiply (embedding tables excluded, the
          output head kept).
  full    matmul + attention-score work: 12 * layers * width * context
          (QK^T and attention*V, 4*T*C FLOPs per layer forward, x3 for fwd+bwd;
          counted dense, i.e. the causal mask is not credited).
"""

COST_NAMES = ("6pbt", "matmul", "full")


def describe(model, context):
    total = sum(p.numel() for p in model.parameters())
    lookup = model.token_embedding.weight.numel() + model.position_embedding.weight.numel()
    return {
        "total": total,
        "matmul": total - lookup,
        "layers": len(model.blocks),
        "width": model.token_embedding.embedding_dim,
        "context": context,
    }


def train_flops_per_token(model, context):
    d = describe(model, context)
    attn = 12 * d["layers"] * d["width"] * d["context"]
    return {
        "6pbt": 6 * d["total"],
        "matmul": 6 * d["matmul"],
        "full": 6 * d["matmul"] + attn,
    }


def eval_flops_per_token(model, context):
    d = describe(model, context)
    return 2 * d["matmul"] + 4 * d["layers"] * d["width"] * d["context"]
