"""YOUR PART: how to turn the small model into the large one.

train.py calls grow() once, at the step given by --switch-at:

    python train.py --model small --then large --switch-at 1000

It's handed the partly trained small model and a brand-new randomly initialized large model.
Fill the large model's weights using the small one's, then return it. Training continues on it,
and its cost (TFLOPs) keeps adding to the small model's, so runs are compared fairly.

Handy: `for name, p in small.named_parameters(): print(name, p.shape)` shows what you can copy.
"""


def grow(small, large):
    # TODO: initialize `large` from `small` here.
    # As it stands nothing is copied, which makes it the "throw the small model away" baseline.
    return large
