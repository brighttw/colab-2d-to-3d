#!/usr/bin/env python3
"""
2D Multi-View Pre-Processing & Workflow Optimizer (`scripts/preprocess_views.py`)

針對輸入至 Hunyuan3D-2 MV 的 2D 三視圖進行自動化前處理與工作流調優：
1. 2D 連通分量智慧視角分離 (Connected-Component View Extraction)：
   自動識別 Front / Side / Back 三大角色連通區塊，即使 A-Pose 雙手與鄰近視角的長尾在水平 X 軸方向交錯重疊，
   也能無損獨立提取各視角並自動過濾底部文字標籤 (FRONT/SIDE/BACK)。
2. 自動抹除背景水平對齊輔助線与底板陰影，並正規化為純白背景 (#FFFFFF)。
3. 跨視角共同垂直基準線對齊與正方形畫布正規化 (預設 1024x1024)，確保 A-Pose 張開雙臂不被壓縮，且三視角頭頂與腳底 100% 像素級對齊。
4. 可選同步升級 workflow_api.json 推論參數 (--tune-workflow)。
"""

from __future__ import annotations

import argparse
from collections import deque
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
    parser.add_argument("--sheet", default=None, help="單張包含正/側/背三視圖的總表圖片路徑")
    parser.add_argument("--front", default=None, help="正面圖路徑")
    parser.add_argument("--side", default=None, help="側面圖路徑")
    parser.add_argument("--back", default=None, help="背面圖路徑")
    parser.add_argument("--out-dir", default="assets/inputs", help="輸出目錄 (預設: assets/inputs)")
    parser.add_argument("--prefix", default="hedgehog", help="輸出檔名前綴 (預設: hedgehog -> hedgehog_front.png 等)")
    parser.add_argument("--canvas-width", type=int, default=1024, help="輸出畫布寬度 (預設: 1024)")
    parser.add_argument("--canvas-height", type=int, default=1024, help="輸出畫布高度 (預設: 1024)")
    parser.add_argument("--side-snout-pull", type=float, default=0.0, help="側面圖口鼻向內液化回縮比例 0.0~0.5 (預設: 0.0)")
    parser.add_argument("--clean-ground-shadow", action="store_true", default=False, help="是否強制清除腳底地板陰影 (預設: False)")
    parser.add_argument("--delight", type=float, default=0.0, help="2D 固有暗部陰影提亮強度 0.0~0.5 (預設: 0.0)")
    parser.add_argument("--smooth-spikes", type=float, default=0.0, help="高頻背刺邊緣柔化強度 0.0~1.0 (預設: 0.0)")
    parser.add_argument("--tune-workflow", default=None, help="選用：自動優化指定的 workflow_api.json 檔案路徑")
    return parser.parse_args(argv)


def load_image_topdown(path: Path) -> np.ndarray:
    """讀取圖片並轉為 top-down (H, W, 3) float32 RGB 陣列 (0.0~1.0)。"""
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[::-1, :, :3].copy()


def save_image_topdown(rgb: np.ndarray, path: Path) -> None:
    """將 top-down (H, W, 3) float32 RGB 陣列存為 PNG。"""
    h, w, _ = rgb.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)
    rgba_bu = rgba[::-1, :, :]
    out = bpy.data.images.new("out_img", width=w, height=h, alpha=True)
    out.pixels[:] = rgba_bu.flatten().tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.filepath_raw = str(path.resolve())
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


def _dilate_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """對二值遮罩執行快速矩形膨脹以保留邊緣抗鋸齒漸層像素。"""
    out = mask.copy()
    for _ in range(radius):
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        out = (
            out
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def _find_exterior_background(
    rgb: np.ndarray,
    luma_thresh: float = 0.965,
    sat_thresh: float = 0.035,
    clean_shadow: bool = False,
) -> np.ndarray:
    """
    從影像四條外邊界向內執行 Flood-Fill，僅標記與外部相連的純白背景像素。
    角色內部被輪廓包圍的雪白肚皮、眼白與高光點絕對不會被標記為背景。
    若 clean_shadow=True，亦將影像底部與背景相連的無彩色地面投影陰影納入背景清除。
    """
    h, w, _ = rgb.shape
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    is_white = (luma > luma_thresh) & (sat < sat_thresh)

    if clean_shadow:
        is_shadow = (np.arange(h)[:, None] > int(h * 0.72)) & (sat < 0.04) & (luma > 0.40)
        is_white = is_white | is_shadow

    # 在 1/2 降採樣網格上快速 Flood-Fill 外部背景
    sh, sw = h // 2, w // 2
    w_s = is_white[: sh * 2 : 2, : sw * 2 : 2]
    ext_s = np.zeros((sh, sw), dtype=bool)
    q = deque()

    for x in range(sw):
        if w_s[0, x]:
            ext_s[0, x] = True
            q.append((0, x))
        if w_s[sh - 1, x] and not ext_s[sh - 1, x]:
            ext_s[sh - 1, x] = True
            q.append((sh - 1, x))
    for y in range(sh):
        if w_s[y, 0] and not ext_s[y, 0]:
            ext_s[y, 0] = True
            q.append((y, 0))
        if w_s[y, sw - 1] and not ext_s[y, sw - 1]:
            ext_s[y, sw - 1] = True
            q.append((y, sw - 1))

    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < sh and 0 <= nx < sw and w_s[ny, nx] and not ext_s[ny, nx]:
                ext_s[ny, nx] = True
                q.append((ny, nx))

    ext_full = np.ones((h, w), dtype=bool)
    ext_full[: sh * 2, : sw * 2] = np.repeat(np.repeat(ext_s, 2, axis=0), 2, axis=1)
    return ext_full


def split_three_view_sheet(sheet_rgb: np.ndarray, clean_shadow: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    透過「外部背景 Flood-Fill 實心前景」+「2D 連通分量分析」分離正、側、背三個視角：
    - 角色內部的白肚皮與眼白 100% 保持實心連通，不產生任何破洞或過曝斑塊。
    - 即使 A-Pose 張開的手臂與鄰近視角的長尾在 X 軸範圍重疊，也能依連通塊完美分離而不截斷任何肢體。
    """
    h, w, _ = sheet_rgb.shape
    ext_bg = _find_exterior_background(sheet_rgb, luma_thresh=0.965, sat_thresh=0.035, clean_shadow=clean_shadow)
    fg = ~ext_bg

    # 在 1/2 降採樣網格上快速搜尋實心前景連通分量
    sh, sw = h // 2, w // 2
    fg_s = fg[: sh * 2 : 2, : sw * 2 : 2]
    labels = np.zeros((sh, sw), dtype=np.int32)
    comps = []
    cid = 0

    for y in range(sh):
        for x in range(sw):
            if fg_s[y, x] and labels[y, x] == 0:
                cid += 1
                q = deque([(y, x)])
                labels[y, x] = cid
                cnt = 0
                sum_x = 0.0
                min_x, max_x, min_y, max_y = x, x, y, y
                while q:
                    cy, cx = q.popleft()
                    cnt += 1
                    sum_x += cx
                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < sh and 0 <= nx < sw and fg_s[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cid
                            q.append((ny, nx))
                if cnt > 3000:
                    comps.append({
                        "cid": cid,
                        "count": cnt,
                        "cx": (sum_x / cnt) * 2.0,
                        "x0": max(0, min_x * 2 - 12),
                        "x1": min(w - 1, max_x * 2 + 12),
                        "y0": max(0, min_y * 2 - 12),
                        "y1": min(h - 1, max_y * 2 + 12),
                    })

    if len(comps) >= 3:
        comps.sort(key=lambda c: c["count"], reverse=True)
        top3 = sorted(comps[:3], key=lambda c: c["cx"])
        shared_y0 = min(c["y0"] for c in top3)
        shared_y1 = max(c["y1"] for c in top3)

        extracted = []
        for c in top3:
            target_cid = c["cid"]
            mask_s = labels == target_cid
            mask_s = _dilate_mask(mask_s, radius=4)
            mask_full = np.zeros((h, w), dtype=bool)
            mask_full[: sh * 2, : sw * 2] = np.repeat(np.repeat(mask_s, 2, axis=0), 2, axis=1)

            x0, x1 = c["x0"], c["x1"]
            crop = np.ones((shared_y1 - shared_y0 + 1, x1 - x0 + 1, 3), dtype=np.float32)
            sub_img = sheet_rgb[shared_y0 : shared_y1 + 1, x0 : x1 + 1, :]
            sub_mask = mask_full[shared_y0 : shared_y1 + 1, x0 : x1 + 1]
            crop[sub_mask] = sub_img[sub_mask]
            extracted.append(crop)

        return extracted[0], extracted[1], extracted[2]

    # Fallback: 傳統垂直波谷切分
    col_counts = fg.sum(axis=0)
    s1_min, s1_max = int(w * 0.28), int(w * 0.42)
    s2_min, s2_max = int(w * 0.58), int(w * 0.72)
    split1 = s1_min + int(np.argmin(col_counts[s1_min:s1_max]))
    split2 = s2_min + int(np.argmin(col_counts[s2_min:s2_max]))

    active_rows = np.where(fg.sum(axis=1) > 15)[0]
    y0 = max(0, int(active_rows[0] - 8)) if len(active_rows) > 0 else int(0.10 * h)
    y1 = min(h - 1, int(active_rows[-1] + 8)) if len(active_rows) > 0 else int(0.90 * h)

    band = sheet_rgb[y0 : y1 + 1, :, :]
    return band[:, :split1, :].copy(), band[:, split1:split2, :].copy(), band[:, split2:, :].copy()


def clean_background_lines_and_ground_shadow(rgb: np.ndarray, clean_shadow: bool = False) -> np.ndarray:
    """僅將與外邊界連通的外部背景像素正規化為純白 1.0，100% 保留角色內部的原始漸層與白肚皮。"""
    out = rgb.copy()
    ext_bg = _find_exterior_background(out, luma_thresh=0.965, sat_thresh=0.035, clean_shadow=clean_shadow)
    out[ext_bg] = 1.0
    return out


def warp_side_snout(rgb: np.ndarray, pull_strength: float = 0.26) -> np.ndarray:
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
    warped[xx - shift_x < 1.0] = 1.0
    return warped


def enhance_and_smooth_view(rgb: np.ndarray, delight: float = 0.0, smooth_spikes: float = 0.0) -> np.ndarray:
    out = rgb.copy()
    if delight <= 0.0 and smooth_spikes <= 0.0:
        return out

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
    h, w, _ = rgb.shape
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
    views: list[np.ndarray], canvas_w: int = 1024, canvas_h: int = 1024
) -> list[np.ndarray]:
    """
    偵測每個視角中角色的實際上下左右邊界，將三個視角的角色高度統一縮放至畫布高度的 86%，
    並將頭頂與腳底對齊在同一水平線上、水平置中於純白正方形畫布。
    """
    target_char_h = int(canvas_h * 0.86)
    crops = []

    for rgb in views:
        h, w, _ = rgb.shape
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        sat = rgb.max(axis=2) - rgb.min(axis=2)
        fg = (luma < 0.94) & ((sat > 0.04) | (luma < 0.89))

        rows = np.where(fg.sum(axis=1) >= 2)[0]
        cols = np.where(fg.sum(axis=0) >= 2)[0]
        if len(rows) == 0 or len(cols) == 0:
            crops.append(rgb.copy())
            continue

        y0 = max(0, int(rows[0] - 4))
        y1 = min(h - 1, int(rows[-1] + 4))
        x0 = max(0, int(cols[0] - 4))
        x1 = min(w - 1, int(cols[-1] + 4))
        crop = rgb[y0 : y1 + 1, x0 : x1 + 1, :].copy()
        crops.append(crop)

    max_ch = max(c.shape[0] for c in crops)
    max_cw = max(c.shape[1] for c in crops)
    scale = target_char_h / max(1, max_ch)
    if max_cw * scale > canvas_w - 40:
        scale = (canvas_w - 40) / max(1, max_cw)

    shared_h = int(round(max_ch * scale))
    target_y_top = (canvas_h - shared_h) // 2
    aligned = []

    for crop in crops:
        ch, cw, _ = crop.shape
        scaled_w = int(round(cw * scale))
        scaled_h = int(round(ch * scale))
        scaled = resize_bilinear(crop, scaled_h, scaled_w)

        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)
        px_x0 = (canvas_w - scaled_w) // 2
        px_y0 = target_y_top + (shared_h - scaled_h)
        canvas[px_y0 : px_y0 + scaled_h, px_x0 : px_x0 + scaled_w, :] = scaled
        aligned.append(canvas)

    return aligned


def tune_workflow_json(workflow_path: Path) -> None:
    """
    升級 workflow_api.json 推論參數至高解析度幾何與材質配置：
    - Node 30 (Hy3DGenerateMeshMultiView): steps=50, guidance_scale=6.0。
    - Node 31 (Hy3DVAEDecode): octree_resolution=384, num_chunks=16000，大幅提升刺、耳朵、五官與鰭的幾何解析度。
    - Node 32 (Hy3DPostprocessMesh): max_facenum=40000，對標 sample.mp4 遊戲模型面數。
    - Node 35 (Hy3DRenderMultiView): render_size=1024, texture_size=2048。
    - Node 36 (Hy3DSampleMultiView): steps=35, guidance_scale=5.5。
    """
    if not workflow_path.is_file():
        print(f"⚠️ 找不到工作流檔案: {workflow_path}", file=sys.stderr)
        return

    wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    if "30" in wf and "inputs" in wf["30"]:
        wf["30"]["inputs"]["steps"] = 50
        wf["30"]["inputs"]["guidance_scale"] = 6.0
    if "31" in wf and "inputs" in wf["31"]:
        wf["31"]["inputs"]["octree_resolution"] = 384
        wf["31"]["inputs"]["num_chunks"] = 16000
    if "32" in wf and "inputs" in wf["32"]:
        wf["32"]["inputs"]["max_facenum"] = 40000
    if "35" in wf and "inputs" in wf["35"]:
        wf["35"]["inputs"]["render_size"] = 1024
        wf["35"]["inputs"]["texture_size"] = 2048
    if "36" in wf and "inputs" in wf["36"]:
        wf["36"]["inputs"]["steps"] = 35
        wf["36"]["inputs"]["guidance_scale"] = 5.5
    if "38" in wf and "inputs" in wf["38"]:
        wf["38"]["inputs"]["inpaint_radius"] = 5

    workflow_path.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ 已升級 ComfyUI 高解析度工作流參數: {workflow_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("✨ 啟動 2D 三視圖自動化前處理管線 (Connected-Component Edition)")
    print("============================================================")

    if args.sheet:
        sheet_path = Path(args.sheet)
        if not sheet_path.is_file():
            print(f"❌ 找不到三視圖總表: {sheet_path}", file=sys.stderr)
            sys.exit(1)
        print(f"🔪 [1/4] 正在從三視圖總表透過連通分量分離正/側/背視圖: {sheet_path}")
        sheet_rgb = load_image_topdown(sheet_path)
        front_rgb, side_rgb, back_rgb = split_three_view_sheet(sheet_rgb, clean_shadow=args.clean_ground_shadow)
    elif args.front and args.side and args.back:
        print("📂 [1/4] 正在讀取正/側/背三視圖檔案...")
        front_rgb = load_image_topdown(Path(args.front))
        side_rgb = load_image_topdown(Path(args.side))
        back_rgb = load_image_topdown(Path(args.back))
    else:
        print("❌ 請提供 --sheet <總表路徑> 或同時提供 --front, --side, --back！", file=sys.stderr)
        sys.exit(1)

    print("🧹 [2/4] 正在清除背景雜訊並正規化純白背景...")
    front_rgb = clean_background_lines_and_ground_shadow(front_rgb, clean_shadow=args.clean_ground_shadow)
    side_rgb = clean_background_lines_and_ground_shadow(side_rgb, clean_shadow=args.clean_ground_shadow)
    back_rgb = clean_background_lines_and_ground_shadow(back_rgb, clean_shadow=args.clean_ground_shadow)

    if args.side_snout_pull > 0.0 or args.delight > 0.0 or args.smooth_spikes > 0.0:
        print(f"🔧 [3/4] 正在執行選用影像調整 (pull={args.side_snout_pull}, delight={args.delight})...")
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
