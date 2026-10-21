#!/usr/bin/env python3
"""
2D Turnaround Sheet Preparation & Auto-Preprocess Pipeline (`scripts/prepare_views.py`)

1. 根據 --subject 與 --style 由 scripts/styles.py 產出標準 21:9 三視圖提示詞。
2. 接收 21:9 三視圖參考圖 (--sheet 或 ref/{prefix}_sheet.png)。
3. 自動呼叫 scripts/preprocess_views.py 將三視圖拆解裁切為 1024x1024 正、側、背三圖，
   並套用外輪廓純白保護與向內距離場，為 Hunyuan3D-2 MV 幾何推論做好準備。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

from styles import get_style, generate_prompt


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", cleaned)
    return slug[:24] if slug else "character"


def main():
    parser = argparse.ArgumentParser(description="2D 三視圖生成提示詞與全自動裁切前處理工具")
    parser.add_argument("--subject", required=True, help="角色特徵描述 (中英皆可)")
    parser.add_argument("--style", "-s", default="popmart", help="風格名稱 (預設: popmart)")
    parser.add_argument("--prefix", "-p", default=None, help="角色檔名前綴 (預設根據 subject 自動生成)")
    parser.add_argument("--sheet", default=None, help="指定 21:9 參考圖路徑 (預設尋找 ref/{prefix}_sheet.png)")
    parser.add_argument("--dry-run", action="store_true", help="僅顯示產出的 Prompt 與執行路徑")

    args = parser.parse_args()

    style_cfg = get_style(args.style)
    prefix = args.prefix or slugify(args.subject)
    ref_dir = BASE_DIR / "ref"
    inputs_dir = BASE_DIR / "assets" / "inputs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    default_sheet = ref_dir / f"{prefix}_sheet.png"
    sheet_path = Path(args.sheet).resolve() if args.sheet else default_sheet

    print("=" * 70)
    print("🎨 2D 三視圖前處理與風格提示詞流水線 (2D Preprocessor)")
    print(f"   角色前綴: {prefix}")
    print(f"   角色主體: {args.subject}")
    print(f"   目標風格: {style_cfg.name} (ID: {style_cfg.id})")
    print(f"   三視圖預期路徑: {sheet_path}")
    print("=" * 70)

    # 1. 產出 21:9 提示詞
    prompt = generate_prompt(style_cfg.id, args.subject)
    print("\n📋 推薦之 21:9 三視圖生成 Prompt:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)

    if args.dry_run:
        print("💡 [Dry-run] 模式已完成。")
        return

    # 2. 檢查三視圖檔案是否存在
    if not sheet_path.is_file():
        print(f"\n⚠️ 尚未找到三視圖檔案: {sheet_path}")
        print("💡 請使用上方的 Prompt 生成 21:9 三視圖後，將圖片存檔至:")
        print(f"   {sheet_path}")
        print(f"   然後重新執行本指令加上: --sheet {sheet_path}\n")
        sys.exit(1)

    # 3. 自動執行三視圖裁切與向內距離場計算 (Stage 1)
    print(f"\n✂️  [Stage 1] 正在執行三視圖自動裁切與外輪廓保護 (scripts/preprocess_views.py)...")
    preprocess_script = BASE_DIR / "scripts" / "preprocess_views.py"
    cmd = [
        sys.executable,
        str(preprocess_script),
        "--sheet", str(sheet_path),
        "--out-dir", str(inputs_dir),
        "--prefix", prefix,
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"❌ 前處理裁切失敗 (Exit code: {res.returncode})")
        sys.exit(res.returncode)

    print("\n🎉 三視圖前處理完成！已輸出至:")
    print(f"   - 前視圖: {inputs_dir / f'{prefix}_front.png'}")
    print(f"   - 側視圖: {inputs_dir / f'{prefix}_side.png'}")
    print(f"   - 背視圖: {inputs_dir / f'{prefix}_back.png'}")


if __name__ == "__main__":
    main()
