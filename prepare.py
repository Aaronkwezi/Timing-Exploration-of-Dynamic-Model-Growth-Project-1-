import torch
from datasets import load_dataset
from transformers import AutoTokenizer
dataset = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1")
print(dataset)

train_text = "\n".join(dataset["train"]["text"])
val_text = "\n".join(dataset["validation"]["text"])
test_text = "\n".join(dataset["test"]["text"])

print("Training characters:", len(train_text))
print("Validation characters:", len(val_text))
print("Test characters: ", len(test_text))
print(train_text[:500])

tokenizer = AutoTokenizer.from_pretrained("gpt2")
train_tokens = tokenizer.encode(train_text)
val_tokens = tokenizer.encode(val_text)
test_tokens = tokenizer.encode(test_text)
print(train_tokens[:20])
train_data = torch.tensor(train_tokens, dtype=torch.long)
val_data = torch.tensor(val_tokens, dtype=torch.long)
test_data = torch.tensor(test_tokens, dtype=torch.long)

torch.save(train_data, "train.pt")
torch.save(val_data,"val.pt")
torch.save(test_data,"test.pt")

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Testing tokens:", len(test_data))