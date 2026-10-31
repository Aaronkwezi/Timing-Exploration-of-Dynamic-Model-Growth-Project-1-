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

def tokenize_split(split):
    tokens = []
    for row in dataset[split]:
        text = row["text"] + "\n"
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(tokens, dtype=torch.long)


train_data = tokenize_split("train")
val_data = tokenize_split("validation")
test_data = tokenize_split("test")

torch.save(train_data, "train.pt")
torch.save(val_data,"val.pt")
torch.save(test_data,"test.pt")

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Testing tokens:", len(test_data))
