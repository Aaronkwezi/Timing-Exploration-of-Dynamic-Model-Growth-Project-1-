"""YOUR PART: the model. Write it here (or in any other .py file and pass it with --model).

    python train.py --model model.py

train.py imports the file and calls build(vocab_size, block_size). It must return a torch module whose
    forward(idx, targets=None) -> (logits, loss)
takes (batch, seq) token ids and returns (batch, seq, vocab_size) logits and the mean cross-entropy.
For growth, write a second file for the large model (e.g. large_model.py) and use --then, see grow.py.
"""


def build(vocab_size, block_size):
    raise NotImplementedError("write the model in model.py")
