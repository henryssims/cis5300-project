import os
from typing import List, Tuple
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from score import sari_score


class SimplificationDataset(Dataset):
    
    def __init__(self, sources: List[str], targets: List[str], tokenizer, max_length: int = 128):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        source = "simplify: " + self.sources[idx]
        target = self.targets[idx]
        
        source_enc = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_enc = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_enc['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_enc['input_ids'].squeeze(),
            'attention_mask': source_enc['attention_mask'].squeeze(),
            'labels': labels
        }


def train_model(
    model,
    tokenizer,
    train_sources: List[str],
    train_targets: List[str],
    val_sources: List[str],
    val_targets: List[str],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    device: str = 'cuda'
):
    train_dataset = SimplificationDataset(train_sources, train_targets, tokenizer)
    val_dataset = SimplificationDataset(val_sources, val_targets, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best model to {output_dir}")

    return model


def strong_baseline(
    model,
    tokenizer,
    source: str,
    max_length: int = 128,
    num_beams: int = 4,
    device: str = 'cuda'
) -> str:
    model.eval()

    with torch.no_grad():
        input_text = "simplify: " + source
        input_enc = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        output_ids = model.generate(
            input_ids=input_enc['input_ids'],
            attention_mask=input_enc['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text


def main():
    model_name = 't5-small'
    output_dir = './t5-simplification'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 3
    batch_size = 16
    learning_rate = 3e-5

    splits = {
        'train': 'wiki.full.aner.ori.train.95.tsv',
        'validation': 'wiki.full.aner.ori.valid.95.tsv',
        'test': 'wiki.full.aner.ori.test.95.tsv'
    }

    print("Loading WikiLarge dataset...")
    train = pd.read_csv(
        "hf://datasets/bogdancazan/wikilarge-text-simplification/" + splits["train"],
        sep="\t"
    )
    val = pd.read_csv(
        "hf://datasets/bogdancazan/wikilarge-text-simplification/" + splits["validation"],
        sep="\t"
    )
    test = pd.read_csv(
        "hf://datasets/bogdancazan/wikilarge-text-simplification/" + splits["test"],
        sep="\t"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    model = train_model(
        model,
        tokenizer,
        train_sources=train['Normal'].tolist(),
        train_targets=train['Simple'].tolist(),
        val_sources=val['Normal'].tolist(),
        val_targets=val['Simple'].tolist(),
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )

    model.to(device)
    outputs = []
    for source in tqdm(test['Normal'], desc="Generating simplifications"):
        output = strong_baseline(model, tokenizer, source, device=device)
        outputs.append(output)


    sari_scores = []
    keep_scores = []
    del_scores = []
    add_scores = []

    for output, source, reference in zip(outputs, test['Normal'], test['Simple']):
        sari, components = sari_score(source, output, [reference])
        sari_scores.append(sari)
        keep_scores.append(components['keep'])
        del_scores.append(components['delete'])
        add_scores.append(components['add'])

    print("\n" + "=" * 50)
    print("Strong Baseline SARI Score Results")
    print("=" * 50)
    print(f"Number of samples: {len(sari_scores)}")
    print()
    print(f"  SARI:        {sum(sari_scores) / len(sari_scores):.2f}")
    print(f"    - Keep:    {sum(keep_scores) / len(keep_scores):.2f}")
    print(f"    - Delete:  {sum(del_scores) / len(del_scores):.2f}")
    print(f"    - Add:     {sum(add_scores) / len(add_scores):.2f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
