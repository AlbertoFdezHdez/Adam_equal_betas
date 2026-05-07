from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from .utils import configure_cache_env


def load_wikitext2(
    batch_size=8,
    block_size=256,
    tokenizer_name="gpt2",
    max_train_samples=None,
    max_val_samples=None,
    pin_mem=True,
    num_workers=0,
    nanogpt_pairs=False,          # True when the loop expects NanoGPT-style (x, y) pairs.
    val_split="validation",       # Use "test" here if the test split should be evaluated.
):
    configure_cache_env()
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # GPT-2 has no pad token by default; use EOS so the collator can pad.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_ds = ds["train"]
    val_ds = ds[val_split]

    # Remove empty lines.
    def nonempty(ex):
        text = ex["text"]
        return text is not None and len(text.strip()) > 0

    train_ds = train_ds.filter(nonempty)
    val_ds = val_ds.filter(nonempty)

    # Tokenize without padding; fixed-size blocks are built below.
    def tok_fn(examples):
        return tok(examples["text"], add_special_tokens=False)

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tok_fn, batched=True, remove_columns=val_ds.column_names)

    # Concatenate and split into fixed-size blocks.
    def group_texts(examples):
        # examples["input_ids"] is a list of token-id lists.
        concatenated_ids = sum(examples["input_ids"], [])
        total_len = (len(concatenated_ids) // block_size) * block_size
        if total_len == 0:
            # Return zero examples if this batch is shorter than one block.
            return {"x": [], "y": []} if nanogpt_pairs else {"input_ids": []}

        blocks = [
            concatenated_ids[i : i + block_size]
            for i in range(0, total_len, block_size)
        ]

        if nanogpt_pairs:
            x = [block[:-1] for block in blocks]
            y = [block[1:] for block in blocks]
            return {"x": x, "y": y}

        return {"input_ids": blocks}

    # Remove original tokenized columns before writing fixed-size blocks.
    train_lm = train_tok.map(
        group_texts,
        batched=True,
        remove_columns=train_tok.column_names,
    )
    val_lm = val_tok.map(
        group_texts,
        batched=True,
        remove_columns=val_tok.column_names,
    )

    if max_train_samples is not None:
        train_lm = train_lm.select(range(min(max_train_samples, len(train_lm))))
    if max_val_samples is not None:
        val_lm = val_lm.select(range(min(max_val_samples, len(val_lm))))

    if nanogpt_pairs:
        import torch

        def collate_pairs(batch):
            x = torch.tensor([item["x"] for item in batch], dtype=torch.long)
            y = torch.tensor([item["y"] for item in batch], dtype=torch.long)
            return x, y

        collate_fn = collate_pairs
    else:
        # HF CausalLM: labels=input_ids; the model performs the shift internally.
        collate_fn = DataCollatorForLanguageModeling(
            tokenizer=tok,
            mlm=False,
            return_tensors="pt",
        )

    train_loader = DataLoader(
        train_lm,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_lm,
        batch_size=max(batch_size * 2, 1),
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )

    vocab_size = tok.vocab_size
    return train_loader, val_loader, vocab_size
