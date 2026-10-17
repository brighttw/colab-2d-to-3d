#!/usr/bin/env bash
set -Eeuo pipefail

SKILL_NAME="colab-2d-to-3d"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_SKILL_DIR="$PROJECT_DIR/.agents/skills/$SKILL_NAME"
GLOBAL_TARGET_DIR="$HOME/.gemini/config/skills/$SKILL_NAME"

usage() {
  cat <<EOF
使用方式: ./install_skill.sh [選項]

將 colab-2d-to-3d AI Agent Skill 安裝至全域或指定專案目錄。

選項:
  -g, --global       安裝至使用者全域環境 (~/.gemini/config/skills/)
  -s, --symlink      使用軟連結 (Symlink) 安裝，便於即時同步修改
  -t, --target PATH  自訂安裝目標路徑
  -h, --help         顯示說明訊息
EOF
}

INSTALL_GLOBAL=0
USE_SYMLINK=0
CUSTOM_TARGET=""

while (($#)); do
  case "$1" in
    -g|--global)
      INSTALL_GLOBAL=1
      shift
      ;;
    -s|--symlink)
      USE_SYMLINK=1
      shift
      ;;
    -t|--target)
      (($# >= 2)) || { echo "缺少參數: $1" >&2; usage >&2; exit 2; }
      CUSTOM_TARGET="$2"
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

if [[ -n "$CUSTOM_TARGET" ]]; then
  DEST_DIR="$CUSTOM_TARGET"
elif (( INSTALL_GLOBAL )); then
  DEST_DIR="$GLOBAL_TARGET_DIR"
else
  # Default to global installation if run directly
  DEST_DIR="$GLOBAL_TARGET_DIR"
fi

if [[ ! -d "$SOURCE_SKILL_DIR" ]]; then
  echo "錯誤: 來源 Skill 目錄不存在: $SOURCE_SKILL_DIR" >&2
  exit 1
fi

echo "📦 準備安裝 AI Agent Skill: $SKILL_NAME"
echo "   來源: $SOURCE_SKILL_DIR"
echo "   目標: $DEST_DIR"

mkdir -p "$(dirname "$DEST_DIR")"

if [[ -e "$DEST_DIR" || -L "$DEST_DIR" ]]; then
  echo "⚠️ 目標路徑已存在，正在備份或替換..."
  rm -rf "$DEST_DIR"
fi

if (( USE_SYMLINK )); then
  ln -s "$SOURCE_SKILL_DIR" "$DEST_DIR"
  echo "🔗 已建立軟連結: $DEST_DIR -> $SOURCE_SKILL_DIR"
else
  cp -R "$SOURCE_SKILL_DIR" "$DEST_DIR"
  echo "✅ 已複製 Skill 內容至: $DEST_DIR"
fi

echo ""
echo "🎉 安裝完成！AI Agent 已可自動識別並載入 '$SKILL_NAME' 技能。"
