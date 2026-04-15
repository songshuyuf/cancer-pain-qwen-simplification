import argparse
import json
import os
import time
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


INSTRUCTION = (
    "你是肿瘤专科护士，请将以下癌症疼痛医学文本简化为初中文化水平患者可理解的语言。"
    "要求：1)保留核心医学事实 2)使用简单句式 3)不添加额外信息。"
    "只输出简化后的文本，不要输出任何解释或其他内容。"
)


def call_openai(text: str):
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def call_kimi(text: str):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("KIMI_API_KEY", ""), base_url="https://api.moonshot.cn/v1")
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def call_qwen_api(text: str):
    import dashscope
    from dashscope import Generation

    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
    response = Generation.call(
        model="qwen-plus",
        messages=[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.output.text.strip()


def call_deepseek_v3(text: str):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("SILICON_API_KEY", ""), base_url="https://api.siliconflow.cn/v1")
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def calculate_sari(sources, predictions, references):
    scores = []
    for src, pred, ref in zip(sources, predictions, references):
        if not pred:
            continue
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
    return float(np.mean(scores)) if scores else 0.0


def run_one_model(test_df, model_name: str, call_fn, output_dir: Path, delay: float = 1.0):
    sources, references, predictions = [], [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=model_name):
        src = row["professional"]
        ref = row["simplified"]
        try:
            pred = call_fn(src)
        except Exception as e:
            print(f"{model_name} 调用失败: {e}")
            pred = ""
        sources.append(src)
        references.append(ref)
        predictions.append(pred)
        time.sleep(delay)

    pd.DataFrame({"source": sources, "reference": references, "prediction": predictions}).to_csv(
        output_dir / f"{model_name}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics = {
        "model": model_name,
        "valid_count": sum(1 for x in predictions if x and x.strip()),
        "SARI": calculate_sari(sources, predictions, references),
        "Compression_Ratio": (
            sum(len(p) for p in predictions if p) / sum(len(s) for s in sources) if sources else 0
        ),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="商业API评估")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("output/commercial_api"))
    args = parser.parse_args()

    load_dotenv()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.test_csv)
    if "input_text" in test_df.columns:
        test_df = test_df.rename(columns={"input_text": "professional", "output_text": "simplified"})

    models = [
        ("gpt4o", call_openai, 1.0),
        ("kimi", call_kimi, 2.0),
        ("qwen_api", call_qwen_api, 1.0),
        ("deepseek_v3", call_deepseek_v3, 1.0),
    ]

    all_metrics = []
    for name, fn, delay in models:
        m = run_one_model(test_df, name, fn, args.output_dir, delay=delay)
        all_metrics.append(m)
        print(m)

    with open(args.output_dir / "commercial_api_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("商业API评估完成")


if __name__ == "__main__":
    main()
