import torch

# Copy every small weight into the top-left corner of the matching large weight.
# Anything the small model doesn't have (extra rows, columns, layers) keeps its random start.
def grow(small, large):
    small_weights = small.state_dict()
    with torch.no_grad():
        for name, big in large.state_dict().items():
            if name in small_weights:
                s = small_weights[name]
                if s.dim() == 1:
                    big[:s.shape[0]] = s
                else:
                    big[:s.shape[0], :s.shape[1]] = s
    return large
