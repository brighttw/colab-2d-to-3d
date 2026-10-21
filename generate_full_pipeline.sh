#!/usr/bin/env bash
# ==============================================================================
# Full End-to-End 2D-to-3D Commercial Pipeline
# Usage: ./generate_full_pipeline.sh --subject "..." --style <name> [options]
# ==============================================================================
set -Eeuo pipefail

usage() {
  cat <<'EOF'
使用說明: ./generate_full_pipeline.sh --subject "角色特徵描述" [選項]

一鍵全自動執行端到端 2D-to-3D 生成管線：
  Stage 1: 根據風格產出 21:9 三視圖並自動拆解裁切為 1024x1024 正、側、背三圖。
  Stage 2: 於 Google Colab L4 GPU 生成 40,000 面高解析度 3D 幾何白模 (.orig.glb)。
  Stage 3: 本機 Blender 執行遮擋感知 PBR 貼圖烘焙 (依風格注入 Roughness / Metallic / Clearcoat / AO)。
  Stage 4: 自動導出為相容 Godot 4 的 .glb 模型，並渲染四視角檢視圖與 360 度旋轉影片。

必要參數:
  --subject "描述"       角色特徵描述 (例如: "遲滯樹懶，淺灰棕毛球身軀、憨厚慢吞吞表情...")

選用參數:
  -s, --style NAME       風格名稱: popmart (預設), anime, ceramic, handpainted, mecha, clay
  -p, --prefix PREFIX    角色檔案前綴代號 (預設根據 subject 自動生成)
  --sheet PATH           使用現有的 21:9 三視圖 (跳過線上生圖)
  --engine ENGINE        3D 生成引擎: hunyuan (預設, 40000面) 或 trellis
  --faces COUNT          目標三角面數 (預設: 40000)
  --gpu GPU              Colab GPU 型號 (預設: L4)
  --turntable            推論完成後渲染 360 度旋轉展示影片
  -h, --help             顯示本說明
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SUBJECT=""
STYLE="popmart"
PREFIX=""
SHEET=""
ENGINE="hunyuan"
FACES="40000"
GPU="L4"
TURNTABLE=0

while (($#)); do
  case "$1" in
    --subject)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      SUBJECT="$2"
      shift 2
      ;;
    -s|--style)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      STYLE="$2"
      shift 2
      ;;
    -p|--prefix)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      PREFIX="$2"
      shift 2
      ;;
    --sheet)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      SHEET="$2"
      shift 2
      ;;
    --engine)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      ENGINE="$2"
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
    --turntable)
      TURNTABLE=1
      shift
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

if [[ -z "$SUBJECT" ]]; then
  echo "錯誤: 必須提供 --subject 角色特徵描述！" >&2
  usage >&2
  exit 2
fi

if [[ -z "$PREFIX" ]]; then
  PREFIX="$(python3 -c "import re; print(re.sub(r'[^a-zA-Z0-9]+', '_', '''$SUBJECT''').strip('_').lower()[:20] or 'character')")"
fi

echo "========================================================================"
echo "🚀 啟動端到端全自動 2D-to-3D 商業級資產生成管線"
echo "   角色主體: $SUBJECT"
echo "   設定風格: $STYLE"
echo "   檔案前綴: $PREFIX"
echo "========================================================================"

# Stage 1: 2D 生成與拆解前處理
GEN_ARGS=(
  --subject "$SUBJECT"
  --style "$STYLE"
  --prefix "$PREFIX"
)
if [[ -n "$SHEET" ]]; then
  GEN_ARGS+=(--sheet "$SHEET")
fi

python3 "$SCRIPT_DIR/scripts/prepare_views.py" "${GEN_ARGS[@]}"

FRONT_IMG="$SCRIPT_DIR/assets/inputs/${PREFIX}_front.png"
SIDE_IMG="$SCRIPT_DIR/assets/inputs/${PREFIX}_side.png"
BACK_IMG="$SCRIPT_DIR/assets/inputs/${PREFIX}_back.png"
OUTPUT_GLB="$SCRIPT_DIR/godot_project/assets/models/${PREFIX}.glb"

if [[ ! -f "$FRONT_IMG" || ! -f "$SIDE_IMG" || ! -f "$BACK_IMG" ]]; then
  echo "❌ 錯誤: 三視圖未完成準備，管線終止。" >&2
  exit 1
fi

# Stage 2 + Stage 3: 雲端幾何推論 + 本機 PBR 烘焙
INFER_ARGS=(
  --front "$FRONT_IMG"
  --side "$SIDE_IMG"
  --back "$BACK_IMG"
  --output "$OUTPUT_GLB"
  --engine "$ENGINE"
  --faces "$FACES"
  --gpu "$GPU"
  --style "$STYLE"
  --auto-bake
)

"$SCRIPT_DIR/run_colab_inference.sh" "${INFER_ARGS[@]}"

# Stage 4: 渲染驗證
RENDER_DIR="$SCRIPT_DIR/scratch/renders_${PREFIX}"
mkdir -p "$RENDER_DIR"
RENDER_ARGS=(
  --glb "$OUTPUT_GLB"
  --out-dir "$RENDER_DIR"
  --prefix "$PREFIX"
)
if (( TURNTABLE )); then
  RENDER_ARGS+=(--turntable)
fi

python3 "$SCRIPT_DIR/scripts/render_turntable.py" "${RENDER_ARGS[@]}"

echo "========================================================================"
echo "🎉 恭喜！全流程一鍵資產生成已完成！"
echo "   - 3D 模型: $OUTPUT_GLB"
echo "   - 四視角檢驗圖: $RENDER_DIR/render_${PREFIX}_textured.png"
echo "   - Godot 4 專案已就緒！"
echo "========================================================================"
