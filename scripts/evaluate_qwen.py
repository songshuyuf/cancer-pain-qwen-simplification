import argparse
import json
from pathlib import Path
import jieba
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION = (
    "你是肿瘤专科护士，请将以下癌症疼痛医学文本简化为初中文化水平患者可理解的语言。"
    "要求：1)保留核心医学事实 2)使用简单句式 3)不添加额外信息"
)


def create_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_text(model, tokenizer, text: str, device):
    prompt = create_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    return response.strip()


def calculate_sari(sources, predictions, references):
    scores = []
    for src, pred, ref in zip(sources, predictions, references):
        src_words = set(jieba.lcut(src))
        pred_words = set(jieba.lcut(pred))
        ref_words = set(jieba.lcut(ref))

        add_pred = pred_words - src_words
        add_ref = ref_words - src_words
        add_p = len(add_pred & add_ref) / len(add_pred) if add_pred else 0
        add_r = len(add_pred & add_ref) / len(add_ref) if add_ref else 0
        add_f1 = 2 * add_p * add_r / (add_p + add_r) if (add_p + add_r) > 0 else 0

        keep_pred = pred_words & src_words
        keep_ref = ref_words & src_words
        keep_p = len(keep_pred & keep_ref) / len(keep_pred) if keep_pred else 0
        keep_r = len(keep_pred & keep_ref) / len(keep_ref) if keep_ref else 0
        keep_f1 = 2 * keep_p * keep_r / (keep_p + keep_r) if (keep_p + keep_r) > 0 else 0

        del_src = src_words - pred_words
        del_ref = src_words - ref_words
        del_p = len(del_src & del_ref) / len(src_words) if del_src and src_words else 0

        scores.append((add_f1 + keep_f1 + del_p) / 3)
    return float(np.mean(scores))


def calculate_rouge(predictions, references):
    from rouge import Rouge

    pred_seg = [" ".join(jieba.lcut(p)) for p in predictions]
    ref_seg = [" ".join(jieba.lcut(r)) for r in references]
    valid = [(p, r) for p, r in zip(pred_seg, ref_seg) if p.strip() and r.strip()]
    if not valid:
        return None
    pred_seg, ref_seg = zip(*valid)
    rouge = Rouge()
    return rouge.get_scores(list(pred_seg), list(ref_seg), avg=True)


def main():
    parser = argparse.ArgumentParser(description="评估Qwen LoRA模型")
    parser.add_argument("--base_model_path", type=Path, default=Path("models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--lora_model_dir", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.test_csv)
    if "input_text" in test_df.columns:
        test_df = test_df.rename(columns={"input_text": "professional", "output_text": "simplified"})

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(args.lora_model_dir))
    model.eval()
    device = next(model.parameters()).device

    sources, references, predictions = [], [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        source = row["professional"]
        reference = row["simplified"]
        pred = generate_text(model, tokenizer, source, device)
        sources.append(source)
        references.append(reference)
        predictions.append(pred)

    pd.DataFrame({"source": sources, "reference": references, "prediction": predictions}).to_csv(
        args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )

    metrics = {
        "SARI": calculate_sari(sources, predictions, references),
        "Compression_Ratio": sum(len(p) for p in predictions) / sum(len(s) for s in sources),
    }
    rouge = calculate_rouge(predictions, references)
    if rouge:
        metrics["ROUGE-1"] = rouge["rouge-1"]["f"]
        metrics["ROUGE-2"] = rouge["rouge-2"]["f"]
        metrics["ROUGE-L"] = rouge["rouge-l"]["f"]

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("评估完成")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
