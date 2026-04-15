import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(raw_csv: Path, enhance_csv: Path | None, use_enhance: bool) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, encoding="utf-8")
    df.columns = ["professional", "simplified"]
    frames = [df]

    if use_enhance and enhance_csv and enhance_csv.exists():
        try:
            enhance_df = pd.read_csv(enhance_csv, encoding="utf-8")
        except UnicodeDecodeError:
            enhance_df = pd.read_csv(enhance_csv, encoding="gbk")
        enhance_df.columns = ["professional", "simplified"]
        frames.append(enhance_df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna()
    merged = merged[merged["professional"] != merged["simplified"]]
    merged = merged[merged["professional"].str.len() >= 10]
    merged = merged[merged["simplified"].str.len() >= 5]
    merged = merged.drop_duplicates(subset=["professional", "simplified"])
    return merged.reset_index(drop=True)


def split_data(df: pd.DataFrame, seed: int = 42):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="准备Qwen训练数据")
    parser.add_argument("--raw_csv", type=Path, required=True, help="原始数据CSV")
    parser.add_argument("--enhance_csv", type=Path, default=Path("data/raw/cancer_data_Enhance.csv"))
    parser.add_argument("--use_enhance", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    use_enhance = args.use_enhance.lower() == "true"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.raw_csv, args.enhance_csv, use_enhance)
    train_df, val_df, test_df = split_data(df)

    df.to_csv(args.output_dir / "all_cleaned.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(args.output_dir / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(args.output_dir / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(args.output_dir / "test.csv", index=False, encoding="utf-8-sig")

    print("数据准备完成")
    print(f"总样本: {len(df)}")
    print(f"训练集: {len(train_df)} 验证集: {len(val_df)} 测试集: {len(test_df)}")


if __name__ == "__main__":
    main()
