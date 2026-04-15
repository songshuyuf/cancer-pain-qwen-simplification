# 癌症疼痛文本简化（Qwen 版本）

这是整理后的精简项目版本,拥有癌症文本缩减功能。

## 1. 项目目标

将专业癌症疼痛医学文本简化为患者更容易理解的文本（面向中文场景）。

## 2. 项目结构

```text
qwen_clean_project/
├─ scripts/
│  ├─ prepare_data.py
│  ├─ train_qwen.py
│  ├─ evaluate_qwen.py
│  └─ evaluate_commercial_api.py
├─ data/
│  ├─ raw/                # 原始CSV放这里
│  └─ processed/          # 清洗和切分结果
├─ output/
│  ├─ qwen_original/
│  ├─ qwen_full_data/
│  └─ commercial_api/
├─ models/
│  └─ Qwen2.5-7B-Instruct/  # 需要你下载后放到这里
├─ .env.example
├─ .gitignore
└─ requirements.txt
```

## 3. Qwen 模型下载地址（重要）

- 模型网址：<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- 下载后放入目录：`qwen_clean_project/models/Qwen2.5-7B-Instruct`

> 如果你保持默认配置，脚本会从 `./models/Qwen2.5-7B-Instruct` 加载模型。

## 4. 快速开始

### 4.1 安装依赖

```bash
pip install -r requirements.txt
```

### 4.2 准备数据

将原始数据（两列：`professional,simplified`）放入：

- `data/raw/cancer_data.csv`（必需）
- `data/raw/cancer_data_Enhance.csv`（可选）

然后执行：

```bash
python scripts/prepare_data.py --raw_csv data/raw/cancer_data.csv --use_enhance false
```

### 4.3 训练 Qwen（原始数据版本）

```bash
python scripts/train_qwen.py --data_csv data/raw/cancer_data.csv --output_dir output/qwen_original
```

### 4.4 评估 Qwen（完整测试集）

```bash
python scripts/evaluate_qwen.py --lora_model_dir output/qwen_original/final_model --test_csv output/qwen_original/test_set.csv --output_dir output/qwen_original/evaluation
```

完整生成结果会保存到：

- `output/qwen_original/evaluation/predictions.csv`

## 5. 商业模型评估

1. 复制 `.env.example` 为 `.env`
2. 填入 API Key
3. 运行：

```bash
python scripts/evaluate_commercial_api.py --test_csv output/qwen_original/test_set.csv
```

## 6. 推理参数说明（默认）

本地模型（Qwen）默认使用：

- `do_sample=True`
- `max_new_tokens=150`
- `temperature=0.3`
- `top_p=0.9`
- `repetition_penalty=1.1`

商业 API 默认使用：

- `temperature=0.3`
- `max_tokens=200`
- 其余参数走平台默认值

