# Datasets

The assignment asks us to "collect various English sentence datasets" and find them ourselves.
We use three public English corpora from the Hugging Face Hub, all tokenized with the GPT-2 BPE
(`tiktoken`, vocabulary of 50,257 tokens) by `prepare.py`.

| Name | Source | Train tokens | Val tokens | Test tokens | Role |
| --- | --- | ---: | ---: | ---: | --- |
| `wikitext2` | `Salesforce/wikitext`, `wikitext-2-raw-v1` | 2,391,884 | 247,289 | 283,287 | quick smoke tests |
| `wikitext103` | `Salesforce/wikitext`, `wikitext-103-raw-v1` | 117,920,140 | 247,289 | 283,287 | main corpus |
| `tinystories` | `roneneldan/TinyStories` | 473,992,236 | 2,272,398 | 2,493,520 | simple-English corpus |

Token counts are from our own preprocessing run (2026-10-26). On disk: WikiText-103 ≈ 226 MB and
TinyStories ≈ 914 MB of uint16 tokens. `data/` is git-ignored, so each teammate regenerates it.

## Why these

- **WikiText-103** is the standard word-level language-modelling benchmark built from verified Good
  and Featured Wikipedia articles, with an official train/val/test split. At ~118M training tokens it is big
  enough that short training runs don't cycle through the data, so loss differences reflect the
  training method rather than memorization.
- **TinyStories** is short stories written in deliberately simple vocabulary. The project brief
  notes that small models are "only suitable for a simpler language"; TinyStories lets us test
  whether growth timing behaves differently on simple text, where a small model learns a lot
  quickly, compared with encyclopedic text.
  It has no test split, so `prepare.py` halves the official validation split into val/test.
- **WikiText-2** has the same distribution as WikiText-103 at about 2% of the size, so it suits
  quick end-to-end checks.

## Preprocessing details

- WikiText rows are lines of an article stream; each split is tokenized as one contiguous text.
- TinyStories rows are separate stories; an `<|endoftext|>` token is appended after each one.
- Validation and test loss are computed on fixed, non-overlapping windows from the start of the
  split (`Data.eval_batches` in `data.py`), so every run is scored on exactly the same tokens.

## Citations (for the report)

- S. Merity, C. Xiong, J. Bradbury, R. Socher. *Pointer Sentinel Mixture Models.* ICLR 2017. (WikiText)
- R. Eldan, Y. Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759, 2023.
- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised Multitask Learners.* OpenAI, 2019. (GPT-2 tokenizer)
