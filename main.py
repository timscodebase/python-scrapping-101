from typing import Iterator, Tuple
from torchtext.datasets import AG_NEWS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.optim as optim

tokenizer = AutoTokenizer.from_pretrained("gpt2")

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-5

def preprocess_data(data_iter: Iterator[Tuple[str, str]]) -> torch.Tensor:
    data = [tokenizer.encode(text, return_tensors="pt") for _, text in data_iter]
    return torch.cat(data)

train_iter = AG_NEWS(split='train')
train_data = preprocess_data(train_iter)
train_data = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for epoch in range(EPOCHS):
    model.train()
    with torch.enable_grad():
        for batch in train_data:
            max_length = max(len(seq) for seq in batch)
            padded_batch = [seq + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in batch]
            padded_batch_tensor = torch.tensor(padded_batch)
            attention_mask = (padded_batch_tensor != tokenizer.pad_token_id).long()
            outputs = model(padded_batch_tensor, attention_mask=attention_mask, labels=padded_batch_tensor)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_data)
    
    scheduler.step(val_loss)

prompt = tokenizer.encode("Write a summary of the new features in the latest release of the Julia Programming Language", return_tensors="pt")
generated = model.generate(prompt)
generated_text = tokenizer.decode(generated[0])

with open("generated.txt", "w") as f:
    f.write(generated_text)