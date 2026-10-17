#!/usr/bin/env python3
"""
2D Multi-View Pre-Processing & Workflow Optimizer (`scripts/preprocess_views.py`)

針對輸入至 Hunyuan3D-2 MV 的 2D 三視圖進行自動化前處理與工作流調優，解決以下常見問題：
1. 自動從單張三視圖設定稿 (--sheet) 裁切出 Front / Side / Back，並自動剔除文字標籤 (FRONT/SIDE/BACK) 與底部殘影。
2. 自動抹除背景水平對齊輔助線 (--clean-lines)，防止 u2net / RMBG 去背殘留灰線造成 3D 破面。
3. 自動清除腳底灰黑接觸陰影 (--remove-ground-shadow)，防止腳底生成多餘底座。
4. 跨視角頭頂與腳底高度對齊與統一縮放 (--align-views)，確保三視圖在 3D 潛空間投影時比例完全一致。
5. 側面圖口鼻過度突出（長吻）液化回縮 (--side-snout-pull)，消除正面圓臉與側面長吻的幾何矛盾。
6. 暗部去陰影提亮 (--delight) 與細刺邊緣柔化 (--smooth-spikes)，防止暗部烘焙死黑與背刺耗盡 UV 空間。
7. 可選同步優化 workflow_api.json 推論參數 (--tune-workflow)。

用法範例:
  # 方式 1: 直接從單張三視圖設定稿切分並前處理輸出至 assets/inputs/
  python3 scripts/preprocess_views.py --sheet ref/Cute_hedgehog_standing_pose.jpg --out-dir assets/inputs --prefix hedgehog --tune-workflow workflow_api.json

  # 方式 2: 針對已切分的三視圖進行前處理優化
  python3 scripts/preprocess_views.py --front assets/inputs/hedgehog_front.png --side assets/inputs/hedgehog_side.png --back assets/inputs/hedgehog_back.png --out-dir assets/inputs --prefix hedgehog
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_runtime() -> None:
    """若當前 Python 缺少 numpy 與影像讀寫支援，自動轉發至 Blender 內建 Python 執行。"""
    try:
        import bpy  # noqa: F401
        import numpy  # noqa: F401
        return
    except ImportError:
        pass

    blender_candidates = [
        shutil.which("blender"),
        "/opt/homebrew/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/local/bin/blender",
    ]
    blender_bin = next((p for p in blender_candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)
    if not blender_bin:
        print("❌ 錯誤: 找不到 Blender 或具備 numpy 的 Python 環境。", file=sys.stderr)
        sys.exit(1)

    script_path = str(Path(__file__).resolve())
    cmd = [blender_bin, "-b", "-noaudio", "--python", script_path, "--", *sys.argv[1:]]
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


_ensure_runtime()

import bpy  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="2D 三視圖自動化前處理與 ComfyUI 工作流優化工具")
    parser.add_argument("--sheet", default=None, help="單張包含正/側/背三視圖的總表圖片路徑 (例: ref/Cute_hedgehog_standing_pose.jpg)")
    parser.add_argument("--front", default=None, help="正面圖路徑")
    parser.add_argument("--side", default=None, help="側面圖路徑")
    parser.add_argument("--back", default=None, help="背面圖路徑")
    parser.add_argument("--out-dir", default="assets/inputs", help="輸出目錄 (預設: assets/inputs)")
    parser.add_argument("--prefix", default="hedgehog", help="輸出檔名前綴 (預設: hedgehog -> hedgehog_front.png 等)")
    parser.add_argument("--canvas-width", type=int, default=680, help="輸出畫布寬度 (預設: 680)")
    parser.add_argument("--canvas-height", type=int, default=1000, help="輸出畫布高度 (預設: 1000)")
    parser.add_argument("--side-snout-pull", type=float, default=0.26, help="側面圖口鼻向內液化回縮比例 0.0~0.5 (預設: 0.26)")
    parser.add_argument("--delight", type=float, default=0.20, help="2D 固有暗部陰影提亮強度 0.0~0.5 (預設: 0.20)")
    parser.add_argument("--smooth-spikes", type=float, default=0.35, help="高頻背刺邊緣柔化強度 0.0~1.0 (預設: 0.35)")
    parser.add_argument("--tune-workflow", default=None, help="選用：自動優化指定的 workflow_api.json 檔案路徑")
    return parser.parse_args(argv)


def load_image_topdown(path: Path) -> np.ndarray:
    """讀取圖片並轉為 top-down (H, W, 3) float32 RGB 陣列 (0.0~1.0)。"""
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    # 翻轉 Y 軸使 index 0 為圖片頂部
    return px[::-1, :, :3].copy()


def save_image_topdown(rgb: np.ndarray, path: Path) -> None:
    """將 top-down (H, W, 3) float32 RGB 陣列存為 PNG。"""
    h, w, _ = rgb.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    # 翻轉回 Blender bottom-up 格式儲存
    rgba_bu = rgba[::-1, :, :]
    out = bpy.data.images.new("out_img", width=w, height=h, alpha=True)
    out.pixels[:] = rgba_bu.flatten().tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.filepath_raw = str(path.resolve())
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


def split_three_view_sheet(sheet_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    從單張三視圖設定表自動偵測主角色區間 (避開底部文字與倒影)，
    並透過垂直投影波谷自動找出 Front / Side / Back 之間的最佳分界線。
    """
    h, w, _ = sheet_rgb.shape
    luma = 0.299 * sheet_rgb[:, :, 0] + 0.587 * sheet_rgb[:, :, 1] + 0.114 * sheet_rgb[:, :, 2]
    sat = sheet_rgb.max(axis=2) - sheet_rgb.min(axis=2)

    # 角色本體具有較高飽和度或明顯低於背景的明度
    is_char = (luma < 0.88) & (sat > 0.05)
    row_counts = is_char.sum(axis=1)

    # 尋找包含所有細刺尖端的完整主角色垂直區間 (忽略文字標籤與底部殘影)
    active_rows = np.where(row_counts > 2)[0]
    if len(active_rows) > 0:
        splits = np.where(np.diff(active_rows) > 25)[0]
        segments = np.split(active_rows, splits + 1)
        main_seg = max(segments, key=lambda s: len(s))
        y0 = max(0, int(main_seg[0] - 32))
        y1 = min(h, int(main_seg[-1] + 24))
    else:
        y0, y1 = int(0.20 * h), int(0.70 * h)

    band = sheet_rgb[y0:y1, :, :]
    band_char = is_char[y0:y1, :]
    col_counts = band_char.sum(axis=0)

    # 在 28%~40% W 與 60%~72% W 區間尋找前景最少的垂直空白波谷作為切分線
    s1_min, s1_max = int(w * 0.28), int(w * 0.40)
    s2_min, s2_max = int(w * 0.60), int(w * 0.72)
    split1 = s1_min + int(np.argmin(col_counts[s1_min:s1_max]))
    split2 = s2_min + int(np.argmin(col_counts[s2_min:s2_max]))

    front = band[:, :split1, :].copy()
    side = band[:, split1:split2, :].copy()
    back = band[:, split2:, :].copy()
    return front, side, back


def clean_background_lines_and_ground_shadow(rgb: np.ndarray) -> np.ndarray:
    """
    1. 清除橫跨白色背景的細水平對齊灰線。
    2. 僅針對腳底接觸地面處 (底部 3.5% 高度) 淡化無彩色灰黑地板陰影，100% 保留臉頰與手臂亮部白毛。
    """
    h, w, _ = rgb.shape
    out = rgb.copy()
    luma = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    sat = out.max(axis=2) - out.min(axis=2)

    # 1. 偵測垂直方向厚度 <= 8px 的低飽和背景水平細線
    pad = 6
    luma_up = np.pad(luma[:-pad, :], ((pad, 0), (0, 0)), mode="edge")
    luma_dn = np.pad(luma[pad:, :], ((0, pad), (0, 0)), mode="edge")
    is_thin_line = (luma_up > 0.93) & (luma_dn > 0.93) & (luma < 0.995) & (sat < 0.065)
    line_mask = is_thin_line.copy()
    for shift in (-2, -1, 1, 2):
        if shift < 0:
            shifted = np.pad(is_thin_line[:shift, :], ((-shift, 0), (0, 0)), mode="constant")
        else:
            shifted = np.pad(is_thin_line[shift:, :], ((0, shift), (0, 0)), mode="constant")
        line_mask |= shifted
    out[line_mask] = 1.0

    # 2. 僅在腳底地板區域 (y_bot 附近) 淡化無彩色灰色地板陰影 (sat < 0.045)
    luma = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    sat = out.max(axis=2) - out.min(axis=2)
    char_rows = np.where(((luma < 0.85) & (sat > 0.06)).sum(axis=1) > 5)[0]
    if len(char_rows) > 0:
        y_bot = char_rows[-1]
        y_floor = max(0, int(y_bot - 0.03 * h))
        sub_luma = luma[y_floor:, :]
        sub_sat = sat[y_floor:, :]
        warmth = out[y_floor:, :, 0] - out[y_floor:, :, 2]
        floor_shadow_w = (
            np.clip((0.045 - sub_sat) / 0.020, 0.0, 1.0)
            * np.clip((0.050 - warmth) / 0.020, 0.0, 1.0)
            * np.clip((sub_luma - 0.50) / 0.15, 0.0, 1.0)
        )
        for c_i in range(3):
            out[y_floor:, :, c_i] = out[y_floor:, :, c_i] * (1.0 - floor_shadow_w) + 1.0 * floor_shadow_w

    bg_pure = (luma > 0.96) & (sat < 0.03)
    out[bg_pure] = 1.0
    return out


def warp_side_snout(rgb: np.ndarray, pull_strength: float = 0.26) -> np.ndarray:
    """
    對側面視角圖中向前(左)過度突出的長鼻吻進行平滑雙線性液化回縮，
    使側面圖的口鼻深度與正面圖的圓臉比例一致。
    """
    if pull_strength <= 0.0:
        return rgb

    h, w, _ = rgb.shape
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    fg = (luma < 0.88) & (sat > 0.06)

    rows = np.where(fg.sum(axis=1) > 3)[0]
    cols = np.where(fg.sum(axis=0) > 3)[0]
    if len(rows) == 0 or len(cols) == 0:
        return rgb

    y_top, y_bot = rows[0], rows[-1]
    x_left, x_right = cols[0], cols[-1]
    char_h = max(1, y_bot - y_top)
    char_w = max(1, x_right - x_left)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cy = y_top + 0.42 * char_h
    sy = 0.13 * char_h
    x_pivot = x_left + 0.36 * char_w

    wy = np.exp(-((yy - cy) ** 2) / (2 * sy**2))
    wx = np.clip((x_pivot - xx) / max(1.0, 0.36 * char_w), 0.0, 1.0)
    shift_x = pull_strength * (0.20 * char_w) * wx * wy

    src_x = np.clip(xx - shift_x, 0.0, w - 1.001)
    x0 = src_x.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    fx = (src_x - x0)[:, :, None]

    iy = np.arange(h)[:, None]
    warped = rgb[iy, x0] * (1.0 - fx) + rgb[iy, x1] * fx
    # 左側超出原邊界處補純白，防止邊緣像素拉長
    warped[xx - shift_x < 1.0] = 1.0
    return warped


def enhance_and_smooth_view(rgb: np.ndarray, delight: float = 0.20, smooth_spikes: float = 0.35) -> np.ndarray:
    """
    1. 適度提亮 2D 原圖中過深的下巴/腋下固有陰影，避免烘焙進 3D 貼圖後疊加光影變死黑。
    2. 對高頻尖刺區域進行邊緣柔化 (使用 edge padding 避免邊界環繞偽影)。
    """
    out = rgb.copy()
    luma = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
    sat = out.max(axis=2) - out.min(axis=2)
    is_fg = (luma < 0.92) & (sat > 0.05)

    if delight > 0.0:
        w_mid = np.clip((luma - 0.25) / 0.15, 0.0, 1.0) * np.clip((0.80 - luma) / 0.20, 0.0, 1.0) * is_fg
        lifted = np.power(np.clip(out, 1e-4, 1.0), 0.82)
        for c_i in range(3):
            out[:, :, c_i] = out[:, :, c_i] * (1.0 - delight * w_mid) + lifted[:, :, c_i] * (delight * w_mid)

    if smooth_spikes > 0.0:
        padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blurred = (
            out * 4.0
            + padded[:-2, 1:-1, :]
            + padded[2:, 1:-1, :]
            + padded[1:-1, :-2, :]
            + padded[1:-1, 2:, :]
        ) / 8.0
        is_spike = (out[:, :, 0] > out[:, :, 2] + 0.12) & (luma > 0.18) & (luma < 0.68)
        w_sp = is_spike.astype(np.float32) * smooth_spikes
        for c_i in range(3):
            out[:, :, c_i] = out[:, :, c_i] * (1.0 - w_sp) + blurred[:, :, c_i] * w_sp

    return np.clip(out, 0.0, 1.0)


def resize_bilinear(rgb: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """純 NumPy 雙線性縮放 (H, W, 3) -> (new_h, new_w, 3)。"""
    h, w, c = rgb.shape
    ys = np.linspace(0, h - 1.001, new_h, dtype=np.float32)
    xs = np.linspace(0, w - 1.001, new_w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0 = yy.astype(np.int32)
    x0 = xx.astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (yy - y0)[:, :, None]
    fx = (xx - x0)[:, :, None]
    top = rgb[y0, x0] * (1.0 - fx) + rgb[y0, x1] * fx
    bot = rgb[y1, x0] * (1.0 - fx) + rgb[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def align_and_normalize_views(
    views: list[np.ndarray], canvas_w: int = 680, canvas_h: int = 1000
) -> list[np.ndarray]:
    """
    偵測每個視角中角色的實際上下左右邊界，將三個視角的角色高度統一縮放至畫布高度的 84%，
    並將頭頂與腳底對齊在同一水平線上、水平置中於純白畫布。
    """
    target_char_h = int(canvas_h * 0.84)
    target_y_top = int(canvas_h * 0.09)
    aligned = []

    for rgb in views:
        h, w, _ = rgb.shape
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        sat = rgb.max(axis=2) - rgb.min(axis=2)
        fg = (luma < 0.88) & (sat > 0.06)

        rows = np.where(fg.sum(axis=1) >= 2)[0]
        cols = np.where(fg.sum(axis=0) >= 2)[0]
        if len(rows) == 0 or len(cols) == 0:
            aligned.append(resize_bilinear(rgb, canvas_h, canvas_w))
            continue

        y0 = max(0, int(rows[0] - 4))
        y1 = min(h - 1, int(rows[-1] + 4))
        x0 = max(0, int(cols[0] - 4))
        x1 = min(w - 1, int(cols[-1] + 4))
        crop = rgb[y0 : y1 + 1, x0 : x1 + 1, :]
        ch, cw, _ = crop.shape

        scale = target_char_h / max(1, ch)
        new_h = target_char_h
        new_w = int(round(cw * scale))
        if new_w > canvas_w - 24:
            scale = (canvas_w - 24) / max(1, cw)
            new_w = canvas_w - 24
            new_h = int(round(ch * scale))

        scaled = resize_bilinear(crop, new_h, new_w)
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)
        px_x0 = (canvas_w - new_w) // 2
        px_y0 = target_y_top + (target_char_h - new_h)
        canvas[px_y0 : px_y0 + new_h, px_x0 : px_x0 + new_w, :] = scaled
        aligned.append(canvas)

    return aligned


def tune_workflow_json(workflow_path: Path) -> None:
    """
    優化 workflow_api.json 推論參數以解決背刺搶佔面數/UV 與貼圖模糊問題：
    - Node 31 (Hy3DVAEDecode): 將 octree_resolution 設為 256，抑制微碎刺體素噪聲，讓臉部獲得更多面數與 UV 空間。
    - Node 35 (Hy3DRenderMultiView): 保持 render_size 1024 (對齊 Node 36 view_size 1024)、texture_size 2048。
    - Node 36 (Hy3DSampleMultiView): 將 steps 提升至 35、guidance_scale 提升至 5.5，增強五官與腮紅細節。
    - Node 38 (CV2InpaintTexture): 將 inpaint_radius 提升至 5，減少 UV 接縫處的黑邊滲色。
    """
    if not workflow_path.is_file():
        print(f"⚠️ 找不到工作流檔案: {workflow_path}", file=sys.stderr)
        return

    wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    if "31" in wf and "inputs" in wf["31"]:
        wf["31"]["inputs"]["octree_resolution"] = 256
    if "35" in wf and "inputs" in wf["35"]:
        wf["35"]["inputs"]["render_size"] = 1024
        wf["35"]["inputs"]["texture_size"] = 2048
    if "36" in wf and "inputs" in wf["36"]:
        wf["36"]["inputs"]["steps"] = 35
        wf["36"]["inputs"]["guidance_scale"] = 5.5
    if "38" in wf and "inputs" in wf["38"]:
        wf["38"]["inputs"]["inpaint_radius"] = 5

    workflow_path.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ 已自動優化 ComfyUI 工作流參數: {workflow_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("✨ 啟動 2D 三視圖自動化前處理管線")
    print("============================================================")

    if args.sheet:
        sheet_path = Path(args.sheet)
        if not sheet_path.is_file():
            print(f"❌ 找不到三視圖總表: {sheet_path}", file=sys.stderr)
            sys.exit(1)
        print(f"🔪 [1/4] 正在從三視圖總表自動切分正/側/背視圖: {sheet_path}")
        sheet_rgb = load_image_topdown(sheet_path)
        front_rgb, side_rgb, back_rgb = split_three_view_sheet(sheet_rgb)
    elif args.front and args.side and args.back:
        print("📂 [1/4] 正在讀取正/側/背三視圖檔案...")
        front_rgb = load_image_topdown(Path(args.front))
        side_rgb = load_image_topdown(Path(args.side))
        back_rgb = load_image_topdown(Path(args.back))
    else:
        print("❌ 請提供 --sheet <總表路徑> 或同時提供 --front, --side, --back！", file=sys.stderr)
        sys.exit(1)

    print("🧹 [2/4] 正在清除背景水平對齊線與腳底接觸陰影...")
    front_rgb = clean_background_lines_and_ground_shadow(front_rgb)
    side_rgb = clean_background_lines_and_ground_shadow(side_rgb)
    back_rgb = clean_background_lines_and_ground_shadow(back_rgb)

    print(f"🔧 [3/4] 正在校正側面口鼻深度比例 (pull={args.side_snout_pull}) 並提亮固有暗部...")
    side_rgb = warp_side_snout(side_rgb, pull_strength=args.side_snout_pull)
    front_rgb = enhance_and_smooth_view(front_rgb, delight=args.delight, smooth_spikes=args.smooth_spikes)
    side_rgb = enhance_and_smooth_view(side_rgb, delight=args.delight, smooth_spikes=args.smooth_spikes)
    back_rgb = enhance_and_smooth_view(back_rgb, delight=args.delight, smooth_spikes=args.smooth_spikes)

    print(f"📐 [4/4] 正在執行跨視角頭頂/腳底對齊正規化 ({args.canvas_width}x{args.canvas_height})...")
    front_out, side_out, back_out = align_and_normalize_views(
        [front_rgb, side_rgb, back_rgb],
        canvas_w=args.canvas_width,
        canvas_h=args.canvas_height,
    )

    p_front = out_dir / f"{args.prefix}_front.png"
    p_side = out_dir / f"{args.prefix}_side.png"
    p_back = out_dir / f"{args.prefix}_back.png"

    save_image_topdown(front_out, p_front)
    save_image_topdown(side_out, p_side)
    save_image_topdown(back_out, p_back)

    print(f"✅ 前處理完成！已輸出乾淨對齊的三視圖：")
    print(f"   - 正面: {p_front}")
    print(f"   - 側面: {p_side}")
    print(f"   - 背面: {p_back}")

    if args.tune_workflow:
        tune_workflow_json(Path(args.tune_workflow))


if __name__ == "__main__":
    main()
