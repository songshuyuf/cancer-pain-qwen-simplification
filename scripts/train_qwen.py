import argparse
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


INSTRUCTION = (
    "你是肿瘤专科护士，请将以下癌症疼痛医学文本简化为初中文化水平患者可理解的语言。"
    "要求：1)保留核心医学事实 2)使用简单句式 3)不添加额外信息"
)


def create_prompt(input_text: str) -> str:
    return (
        f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_and_split_data(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = ["professional", "simplified"]
    df = df.dropna()
    df = df[df["professional"] != df["simplified"]]
    df = df[df["professional"].str.len() >= 10]
    df = df[df["simplified"].str.len() >= 5]
    df = df.rename(columns={"professional": "input_text", "simplified": "output_text"})

    train_df, temp = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)
    return train_df, val_df, test_df


def prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int = 512) -> Dataset:
    rows = []
    for _, row in df.iterrows():
        prompt = create_prompt(row["input_text"])
        output = row["output_text"] + "<|im_end|>"
        full_text = prompt + output

        full_encoded = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        prompt_encoded = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)

        input_ids = full_encoded["input_ids"]
        attention_mask = full_encoded["attention_mask"]
        prompt_len = len(prompt_encoded["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]

        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct LoRA训练")
    parser.add_argument("--model_path", type=Path, default=Path("models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--data_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("output/qwen_original"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_and_split_data(args.data_csv)
    test_df.to_csv(args.output_dir / "test_set.csv", index=False, encoding="utf-8-sig")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()

    final_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"训练完成，模型已保存：{final_dir}")


if __name__ == "__main__":
    main()
