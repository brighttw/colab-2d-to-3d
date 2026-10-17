#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
使用說明: run_colab_inference.sh --front FRONT.png --back BACK.png --side SIDE.png --output MODEL.glb [選項]

透過 Google Colab CLI 在遠端 GPU (預設 L4) 自動執行 Hunyuan3D-2 MV，將三視圖轉換為相容 Godot 4 的 .glb 3D 模型。

必要參數:
  -f, --front PATH       正面視角圖片路徑
  -b, --back PATH        背面視角圖片路徑
  -s, --side PATH        側面視角圖片路徑
  -o, --output PATH      輸出的 .glb 模型儲存路徑

選用參數:
  --faces COUNT          目標三角面數 (預設: 15000)
  --gpu MODEL            Colab GPU 型號 (預設: L4；亦支援 A100)
  --session NAME         指定或重用 Colab 會話名稱
  --keep-session         推論完成後保持會話運行 (預設自動停止以節省點數)
  --timeout SECONDS      執行逾時上限秒數 (預設: 3600)
  -h, --help             顯示本說明
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="$SCRIPT_DIR/scripts/runner.py"

FRONT_IMG=""
BACK_IMG=""
SIDE_IMG=""
OUTPUT_GLB=""
FACES="15000"
GPU="L4"
SESSION=""
KEEP_SESSION=0
TIMEOUT="3600"

while (($#)); do
  case "$1" in
    -f|--front)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      FRONT_IMG="$2"
      shift 2
      ;;
    -b|--back)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      BACK_IMG="$2"
      shift 2
      ;;
    -s|--side)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      SIDE_IMG="$2"
      shift 2
      ;;
    -o|--output)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      OUTPUT_GLB="$2"
      shift 2
      ;;
    --faces)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      FACES="$2"
      shift 2
      ;;
    --gpu)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      GPU="$2"
      shift 2
      ;;
    --session)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      SESSION="$2"
      shift 2
      ;;
    --keep-session)
      KEEP_SESSION=1
      shift
      ;;
    --timeout)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知選項: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$FRONT_IMG" || -z "$BACK_IMG" || -z "$SIDE_IMG" || -z "$OUTPUT_GLB" ]]; then
  echo "錯誤: 必須同時提供 --front, --back, --side 與 --output 參數！" >&2
  usage >&2
  exit 2
fi

ARGS=(
  --front "$FRONT_IMG"
  --back "$BACK_IMG"
  --side "$SIDE_IMG"
  --output "$OUTPUT_GLB"
  --faces "$FACES"
  --gpu "$GPU"
  --timeout "$TIMEOUT"
)

if [[ -n "$SESSION" ]]; then
  ARGS+=(--session "$SESSION")
fi
if (( KEEP_SESSION )); then
  ARGS+=(--keep-session)
fi

exec python3 "$RUNNER" "${ARGS[@]}"
