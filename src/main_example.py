#!/usr/bin/env python3
"""
Fine-tuning di DistilBERT su IMDb per text classification (sentiment analysis).
Ispirato alla guida ufficiale Hugging Face 'Text classification' e al dataset IMDb.  # :contentReference[oaicite:4]{index=4}
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)
import evaluate
from tqdm.auto import tqdm


def main():
    ############################
    # 1. Hyperparam e device
    ############################
    model_name = "distilbert/distilbert-base-uncased"  # DistilBERT è più leggero/faster di BERT.  # :contentReference[oaicite:5]{index=5}
    batch_size = 16
    num_epochs = 2          # per demo; puoi aumentare
    lr = 2e-5               # tipico LR per fine-tuning Transformer
    seed = 42

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ############################
    # 2. Caricamento dataset IMDb
    ############################
    # IMDb = recensioni di film con etichetta 0 (negativo) o 1 (positivo).
    # Split ufficiali: 'train' (25k esempi) e 'test' (25k esempi). :contentReference[oaicite:6]{index=6}
    raw_datasets = load_dataset("imdb")

    # Creiamo una validation split dal train (80/20)
    split = raw_datasets["train"].train_test_split(test_size=0.2, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = raw_datasets["test"]

    print(train_ds[0])

    ############################
    # 3. Tokenizer e preprocessing
    ############################
    # Usiamo il tokenizer di DistilBERT per convertire testo in input_ids + attention_mask. :contentReference[oaicite:7]{index=7}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        # truncation=True → taglia le frasi troppo lunghe alla max length del modello
        return tokenizer(examples["text"], truncation=True)

    # Applichiamo la tokenizzazione in batch per velocizzare
    train_ds = train_ds.map(preprocess_function, batched=True)
    val_ds = val_ds.map(preprocess_function, batched=True)
    test_ds = test_ds.map(preprocess_function, batched=True)

    # Il modello Hugging Face si aspetta l'argomento "labels" nel forward,
    # mentre IMDb usa la colonna "label". Rinominiamo.
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    # Teniamo solo le colonne che servono a DistilBERT:
    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type=None, columns=keep_cols)
    val_ds.set_format(type=None, columns=keep_cols)
    test_ds.set_format(type=None, columns=keep_cols)

    ############################
    # 4. DataCollator + DataLoader
    ############################
    # DataCollatorWithPadding fa padding dinamico al batch: ogni batch ha sequenze
    # tutte della stessa lunghezza, ma senza sprecare spazio sul dataset intero. :contentReference[oaicite:8]{index=8}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    ############################
    # 5. Modello
    ############################
    # AutoModelForSequenceClassification aggiunge automaticamente una classification head
    # sopra DistilBERT, con num_labels=2 per sentiment positivo/negativo. :contentReference[oaicite:9]{index=9}
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    ############################
    # 6. Optimizer, scheduler, metric
    ############################
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)

    # Linear LR decay, come nel Trainer Hugging Face/get_scheduler. :contentReference[oaicite:10]{index=10}
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Usiamo la metrica di accuracy di Hugging Face Evaluate. :contentReference[oaicite:11]{index=11}
    metric = evaluate.load("accuracy")

    ############################
    # 7. Training loop
    ############################
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Il data_collator ci ha già fatto padding e ci ha dato tensori PyTorch
            batch = {k: v.to(device) for k, v in batch.items()}  # input_ids, attention_mask, labels

            # Forward: il modello calcola automaticamente la loss se passiamo labels
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        # Valutazione sul validation set
        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=preds, references=labels)

        val_acc = metric.compute()["accuracy"]
        print(f"Validation accuracy after epoch {epoch+1}: {val_acc:.4f}")

    ############################
    # 8. Test finale
    ############################
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=preds, references=labels)

    test_acc = metric.compute()["accuracy"]
    print(f"Test accuracy: {test_acc:.4f}")

    ############################
    # 9. Inference su testo custom
    ############################
    # Stessa logica mostrata nella guida ufficiale: tokenizziamo una frase a mano,
    # facciamo forward e prendiamo l'argmax per ottenere POSITIVE / NEGATIVE. :contentReference[oaicite:12]{index=12}
    custom_text = "I really loved this movie. The performances were incredible and it kept me engaged."
    inputs = tokenizer(
        custom_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    print("Custom text:", custom_text)
    print("Predicted label:", model.config.id2label[pred_id])


if __name__ == "__main__":
    main()
