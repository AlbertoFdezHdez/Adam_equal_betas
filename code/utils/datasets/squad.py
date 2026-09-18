from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    GPT2Config,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoConfig, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from .utils import configure_cache_env

def load_squad(model, batch_size=8, max_train_samples=None, max_val_samples=None, pin_mem=True):
    configure_cache_env()
    tok = AutoTokenizer.from_pretrained("t5-small")
    ds = load_dataset("squad")
    train_ds = ds["train"]
    val_ds   = ds["validation"]
    # construir pares entrada/salida simples
    def build_io(ex):
        q = ex["question"]; c = ex["context"]
        a = ex["answers"]["text"][0] if len(ex["answers"]["text"])>0 else ""
        return {"input_text": f"question: {q}  context: {c}", "target_text": a}
    train_ds = train_ds.map(build_io)
    val_ds   = val_ds.map(build_io)
    def tok_fn(ex):
        model_inputs = tok(ex["input_text"], truncation=True, padding="max_length", max_length=256)
        with tok.as_target_tokenizer():
            labels = tok(ex["target_text"], truncation=True, padding="max_length", max_length=64)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(tok_fn, batched=True, remove_columns=val_ds.column_names)
    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_val_samples is not None:
        val_ds   = val_ds.select(range(min(max_val_samples, len(val_ds))))
    collate = DataCollatorForSeq2Seq(tok, model=model, return_tensors="pt")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=max(batch_size*2,1), shuffle=False, collate_fn=collate, pin_memory=pin_mem)
    return train_loader, val_loader
