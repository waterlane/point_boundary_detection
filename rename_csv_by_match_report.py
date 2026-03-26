import argparse
import json
import shutil
from pathlib import Path


def rename_csv_by_match_report(report_path, csv_dir, output_dir):
    report_path = Path(report_path)
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    rows = []
    for item in report:
        source_ply = Path(item["source"])
        target_ply = Path(item["target"])
        source_csv = csv_dir / f"{source_ply.stem}.csv"
        target_csv = output_dir / f"{target_ply.stem}.csv"

        if not source_csv.exists():
            raise FileNotFoundError(f"未找到对应 csv: {source_csv}")

        shutil.copy2(source_csv, target_csv)
        rows.append((str(source_csv), str(target_csv), item["matched_mesh"], item["score"]))
        print(f"{source_csv} -> {target_csv}")

    print(f"已完成，共复制并重命名 {len(rows)} 个 csv 到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="根据匹配报告重命名对应的 csv 文件")
    parser.add_argument("report", help="match_and_rename_ply.py 生成的 JSON 报告")
    parser.add_argument("csv_dir", help="原始 csv 文件夹")
    parser.add_argument("output_dir", help="输出重命名后的 csv 文件夹")
    args = parser.parse_args()

    rename_csv_by_match_report(args.report, args.csv_dir, args.output_dir)


if __name__ == "__main__":
    main()
