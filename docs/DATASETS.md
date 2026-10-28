# Dataset

The assignment asks us to "collect various English sentence datasets". To keep the project simple
we use one: **WikiText-103** (`Salesforce/wikitext`, `wikitext-103-raw-v1` on the Hugging Face Hub),
tokenized with the GPT-2 BPE (`tiktoken`, 50,257 tokens) by `prepare.py` into `data/wikitext103/`.

| Split | Tokens |
| --- | ---: |
| train | 117,920,140 |
| val | 247,289 |
| test | 283,287 |

Token counts are from our own preprocessing run (2026-10-26); about 226 MB on disk. `data/` is
git-ignored, so each teammate regenerates it.

## Why this one

WikiText-103 is the standard language-modelling benchmark, built from verified Good and Featured
Wikipedia articles, with an official train/val/test split. At ~118M training tokens it is big enough
that short runs don't cycle through the data, so loss differences reflect the training method
rather than memorization.

## Preprocessing details

- Rows are lines of an article stream; each split is tokenized as one contiguous text.
- Validation and test loss are computed on fixed, non-overlapping windows from the start of the
  split (`Data.eval_batches` in `data.py`), so every run is scored on exactly the same tokens.

## Citations (for the report)

- S. Merity, C. Xiong, J. Bradbury, R. Socher. *Pointer Sentinel Mixture Models.* ICLR 2017. (WikiText)
- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised Multitask Learners.* OpenAI, 2019. (GPT-2 tokenizer)
