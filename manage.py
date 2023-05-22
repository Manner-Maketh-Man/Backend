#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BERTClassifier(nn.Module):
    def __init__(self, bert, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.num_classes = num_classes
        self.classifier = nn.Linear(768, self.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


class BERTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.sentences = [data[0] for data in dataset]
        self.labels = [int(data[1]) for data in dataset]  # Convert label to int
        self.max_len = max_len

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "label": torch.tensor(label, dtype=torch.long).to(device)
        }

    def __len__(self):
        return len(self.labels)


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
