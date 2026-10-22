#!/usr/bin/env python3
"""
Occlusion-Aware Multi-View Projection & Mesh Harmonic PBR Baker (`scripts/bake_multiview_textures.py`)

針對 Hunyuan3D-2 高解析度幾何模型進行無條紋、遮擋感知的多視角 PBR 貼圖烘焙：
1. 自動檢測並生成高密度無重疊 UV 圖譜 (若模型尚未包含有效 UV)。
2. 3D BVHTree Ray-cast 自遮擋檢測 (Self-Occlusion Rejection)：
   自動識別在正、背、左、右視角中被手臂 (A-Pose)、前排背刺、腮冠或尾巴遮擋的區域，
   杜絕手臂顏色穿透印在身體側面，以及正面臉頰顏色投射到後方背刺造成的橫向條紋。
3. 2D 外部背景 Flood-Fill 與輪廓內縮距離場保護 (Silhouette Erosion Margin)：
   杜絕 2D 輪廓邊緣像素在斜角處被拉長成條紋，並防止純白背景滲色。
4. 3D 網格拓撲拉普拉斯諧波修補 (Mesh Geodesic Harmonic Inpainting)：
   讓被遮擋的腋下、刺間深溝與斜角過渡區沿 3D 幾何表面自然過渡。
5. 自動烘焙 Cycles AO 立體漸層與分區 Roughness PBR 貼圖 (眼睛/鼻頭亮面高光 vs 身體柔光盲盒質感)。
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 確保 Blender 執行環境能夠匯入同目錄下的 styles 模組
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from styles import get_style, StyleConfig


def _ensure_blender_runtime() -> None:
    try:
        import bpy  # noqa: F401
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
        print("❌ 錯誤: 找不到 Blender 執行檔。", file=sys.stderr)
        sys.exit(1)

    script_path = str(Path(__file__).resolve())
    cmd = [blender_bin, "-b", "-noaudio", "--python", script_path, "--", *sys.argv[1:]]
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


_ensure_blender_runtime()

import bpy  # noqa: E402
import bmesh  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="遮擋感知多視角 PBR 貼圖烘焙器")
    parser.add_argument("--mesh", required=True, help="輸入幾何 GLB 模型路徑")
    parser.add_argument("--front", required=True, help="正面 2D 參考圖")
    parser.add_argument("--back", required=True, help="背面 2D 參考圖")
    parser.add_argument("--side", required=True, help="側面 2D 參考圖")
    parser.add_argument("-o", "--output", required=True, help="輸出 GLB 路徑")
    parser.add_argument("--style", "-s", default="popmart", help="材質風格 (預設: popmart，可選: popmart, anime, ceramic, handpainted, mecha, clay, fur)")
    parser.add_argument("--texture-size", type=int, default=2048, help="烘焙貼圖解析度 (預設: 2048)")
    parser.add_argument("--ao-strength", type=float, default=None, help="AO 立體陰影融合強度 (預設依風格自適應)")
    parser.add_argument("--roughness", type=float, default=None, help="身體材質基礎粗糙度 (預設依風格自適應)")
    parser.add_argument("--bake-normal", action="store_true", default=True, help="自動計算高頻毛流並烘焙切線空間法線貼圖 (預設開啟)")
    parser.add_argument("--no-bake-normal", action="store_false", dest="bake_normal", help="停用法線貼圖烘焙")
    parser.add_argument("--normal-strength", type=float, default=None, help="法線起伏強度 (預設依風格自適應: fur 為 0.75，其他風格為 0.35)")
    parser.add_argument("--sharpen-blending", action="store_true", default=True, help="啟用非線性權重指數銳化，消除多視角毛髮重影 (預設開啟)")
    parser.add_argument("--no-sharpen-blending", action="store_false", dest="sharpen_blending", help="停用權重銳化")
    parser.add_argument("--eye-refine", action="store_true", default=True, help="自動偵測雙眼並啟用無損抗鋸齒增艷與高光防截斷保護 (預設開啟)")
    parser.add_argument("--no-eye-refine", action="store_false", dest="eye_refine", help="停用自動眼部精修")
    parser.add_argument("--clean-webbing", action="store_true", default=True, help="自動檢測並消除懸空薄片與蹼狀鬍鬚黏連 (預設開啟)")
    parser.add_argument("--no-clean-webbing", action="store_false", dest="clean_webbing", help="停用自動去蹼平滑")
    parser.add_argument("--cornea-dome", action="store_true", default=True, help="自動沿眼球光學軸微推球面微弧角膜 (預設開啟)")
    parser.add_argument("--no-cornea-dome", action="store_false", dest="cornea_dome", help="停用微弧角膜幾何微調")
    parser.add_argument("--fur-gamma", type=float, default=None, help="毛皮深灰質感伽瑪微調 (預設: fur 風格為 1.25 加深濃郁灰階，其他風格為 1.0)")
    parser.add_argument("--update-godot", action="store_true", default=True, help="自動觸發 Godot 4 headless 重新匯入")
    parser.add_argument("--verify", action="store_true", default=True, help="烘焙後自動執行閉環差分質檢 (預設開啟)")
    parser.add_argument("--no-verify", action="store_false", dest="verify", help="停用閉環差分質檢")
    parser.add_argument("--diff-report", type=str, default=None, help="自訂差分質檢報告輸出 PNG 路徑")
    return parser.parse_args(argv)


def load_view_and_distance_field(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    讀取 2D 視角圖片 (bottom-up 對齊 Blender UV V 軸: 0 為底, 1 為頂)，
    透過外部純白 Flood-Fill 計算實心前景遮罩與向內侵蝕距離權重場 (0.0 ~ 1.0)，
    同時將外部背景像素以邊緣前景顏色向外擴展填充 (Edge Dilation)，徹底消除白邊滲色。
    """
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[:, :, :3].copy()
    bpy.data.images.remove(img)

    luma = 0.299 * px[:, :, 0] + 0.587 * px[:, :, 1] + 0.114 * px[:, :, 2]
    sat = px.max(axis=2) - px.min(axis=2)
    is_white = (luma > 0.965) & (sat < 0.035)

    ext = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        if is_white[0, x]:
            ext[0, x] = True
            q.append((0, x))
        if is_white[h - 1, x] and not ext[h - 1, x]:
            ext[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if is_white[y, 0] and not ext[y, 0]:
            ext[y, 0] = True
            q.append((y, 0))
        if is_white[y, w - 1] and not ext[y, w - 1]:
            ext[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and is_white[ny, nx] and not ext[ny, nx]:
                ext[ny, nx] = True
                q.append((ny, nx))

    fg = ~ext
    y_idx, x_idx = np.where(fg)
    if len(y_idx) == 0:
        return px, np.ones((h, w), dtype=np.float32), (0.5, 1.0, 0.5, 1.0)

    x_min, x_max = float(x_idx.min()), float(x_idx.max())
    y_min, y_max = float(y_idx.min()), float(y_idx.max())
    bbox = (
        (x_min + x_max) / (2.0 * w),
        (x_max - x_min) / float(w),
        (y_min + y_max) / (2.0 * h),
        (y_max - y_min) / float(h),
    )

    # 計算前景向內距離權重 (距離背景 <= 3px 為 0，3..14px 線性爬升至 1.0)
    dist = np.zeros((h, w), dtype=np.float32)
    curr = fg.copy()
    max_erode = 14
    for step in range(1, max_erode + 1):
        p = np.pad(curr, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        curr = curr & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        if step >= 3:
            dist[curr] = (step - 2) / float(max_erode - 2)

    # 將前景內縮 2px 後向外擴張填滿背景 (以 48 次迭代徹底填滿外圍純白背景，防止邊緣抗鋸齒白邊被採樣)
    valid_color_mask = dist > 0.05
    dilated_px = px.copy()
    fill_mask = valid_color_mask.copy()
    for _ in range(48):
        unfilled = ~fill_mask
        if not unfilled.any():
            break
        p_m = np.pad(fill_mask, ((1, 1), (1, 1)), mode="edge")
        p_c = np.pad(dilated_px, ((1, 1), (1, 1), (0, 0)), mode="edge")
        acc_c = np.zeros_like(dilated_px)
        acc_w = np.zeros((h, w, 1), dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nm = p_m[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            nc = p_c[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx, :]
            w_nb = nm[:, :, None].astype(np.float32)
            acc_c += nc * w_nb
            acc_w += w_nb
        can_fill = unfilled & (acc_w[:, :, 0] > 0)
        dilated_px[can_fill] = acc_c[can_fill] / np.maximum(acc_w[can_fill], 1e-5)
        fill_mask[can_fill] = True

    return dilated_px, dist, bbox


def sample_bilinear_array(arr: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """在 bottom-up (H, W, ...) 陣列上執行向量化雙線性插值採樣。"""
    h, w = arr.shape[:2]
    x = np.clip(u, 0.0, 1.0) * (w - 1.001)
    y = np.clip(v, 0.0, 1.0) * (h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    if arr.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    top = arr[y0, x0] * (1.0 - fx) + arr[y0, x1] * fx
    bot = arr[y1, x0] * (1.0 - fx) + arr[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def detect_eyes_and_ears_universal(
    px_f: np.ndarray,
    dist_f: np.ndarray,
    bbox_f: tuple[float, float, float, float],
    coords: np.ndarray,
    normals: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    cx: float,
    cy: float,
    cz: float,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    z_rel: np.ndarray,
) -> dict:
    """
    自適應偵測角色 2D/3D 雙眼五官中心、半徑、光學軸，以及幾何耳根高度與鞍部分界。
    100% 動態計算，無任何寫死座標；若未檢測到雙眼或無外耳，優雅退避安全預設。
    """
    h, w, _ = px_f.shape
    uf_c, uf_w, vf_c, vf_h = bbox_f

    # 1. 2D 雙眼候選搜尋
    luma = 0.299 * px_f[:, :, 0] + 0.587 * px_f[:, :, 1] + 0.114 * px_f[:, :, 2]
    u_grid, v_grid = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    u_rel = (u_grid - uf_c) / uf_w
    v_rel = (v_grid - (vf_c - 0.5 * vf_h)) / vf_h

    head_mask = (dist_f > 0.40) & (v_rel >= 0.35) & (v_rel <= 0.90) & (np.abs(u_rel) < 0.36)
    head_luma = luma[head_mask]
    dark_thresh = min(0.18, float(np.percentile(head_luma, 8.0)) + 0.05) if len(head_luma) > 0 else 0.12
    pupil_candidates = head_mask & (luma < dark_thresh)

    visited = np.zeros_like(pupil_candidates, dtype=bool)
    clusters = []
    min_pixels = int(0.0002 * w * h)
    max_pixels = int(0.040 * w * h)

    for y in range(h):
        for x in range(w):
            if pupil_candidates[y, x] and not visited[y, x]:
                q = deque([(y, x)])
                visited[y, x] = True
                pts = []
                while q:
                    cy_pt, cx_pt = q.popleft()
                    pts.append((cy_pt, cx_pt))
                    for ny, nx in ((cy_pt - 1, cx_pt), (cy_pt + 1, cx_pt), (cy_pt, cx_pt - 1), (cy_pt, cx_pt + 1)):
                        if 0 <= ny < h and 0 <= nx < w and pupil_candidates[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                if min_pixels <= len(pts) <= max_pixels:
                    pts_arr = np.array(pts)
                    cy_m = pts_arr[:, 0].mean()
                    cx_m = pts_arr[:, 1].mean()
                    u_m = cx_m / w
                    v_m = cy_m / h
                    rad_px = np.sqrt(len(pts) / math.pi)
                    rad_u = rad_px / w
                    clusters.append({
                        "u": u_m, "v": v_m,
                        "u_rel": (u_m - uf_c) / uf_w,
                        "v_rel": (v_m - (vf_c - 0.5 * vf_h)) / vf_h,
                        "rad_u": rad_u, "count": len(pts),
                        "cy": int(cy_m), "cx": int(cx_m),
                        "rad_px": int(rad_px)
                    })

    best_pair = None
    min_diff = 1e9
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i >= j:
                continue
            if c1["u_rel"] * c2["u_rel"] < 0:
                left = c1 if c1["u_rel"] < 0 else c2
                right = c2 if c1["u_rel"] < 0 else c1
                v_diff = abs(left["v_rel"] - right["v_rel"])
                sym_diff = abs(abs(left["u_rel"]) - abs(right["u_rel"]))
                size_ratio = min(left["count"], right["count"]) / max(left["count"], right["count"])
                if v_diff < 0.08 and sym_diff < 0.14 and size_ratio > 0.35:
                    score = v_diff * 3.0 + sym_diff * 2.0 + (1.0 - size_ratio)
                    if score < min_diff:
                        min_diff = score
                        best_pair = (left, right)

    crown_mask = (np.abs(x_rel) < 0.05) & (z_rel > 0.70)
    lateral_mask = (np.abs(x_rel) > 0.08) & (z_rel > 0.70)
    z_crown = z_rel[crown_mask].max() if crown_mask.any() else 0.85
    z_lateral = z_rel[lateral_mask].max() if lateral_mask.any() else 0.85
    has_protruding_ears = (z_lateral > z_crown + 0.035)

    eye_info = None
    if best_pair is not None:
        left, right = best_pair
        u_f_all = uf_c + x_rel * uf_w
        v_f_all = vf_c + ((coords[:, 2] - cz) / dz) * vf_h
        front_verts = (normals[:, 1] < -0.15) & (y_rel < 0.0)

        d_l = (u_f_all - left["u"]) ** 2 + (v_f_all - left["v"]) ** 2
        d_l[~front_verts] = 1e9
        idx_l = int(np.argmin(d_l))

        d_r = (u_f_all - right["u"]) ** 2 + (v_f_all - right["v"]) ** 2
        d_r[~front_verts] = 1e9
        idx_r = int(np.argmin(d_r))

        center_l = coords[idx_l]
        center_r = coords[idx_r]
        R_dome = float(max(left["rad_u"], right["rad_u"]) * 1.65 * dx / uf_w)

        mask_l_verts = np.linalg.norm(coords - center_l, axis=1) < R_dome
        mask_r_verts = np.linalg.norm(coords - center_r, axis=1) < R_dome
        axis_l = normals[mask_l_verts].mean(axis=0) if mask_l_verts.any() else normals[idx_l]
        axis_r = normals[mask_r_verts].mean(axis=0) if mask_r_verts.any() else normals[idx_r]
        axis_l = axis_l / max(1e-5, float(np.linalg.norm(axis_l)))
        axis_r = axis_r / max(1e-5, float(np.linalg.norm(axis_r)))

        max_eye_z = float(max(z_rel[idx_l], z_rel[idx_r]))

        eye_info = {
            "center_l": center_l,
            "center_r": center_r,
            "R_dome": R_dome,
            "axis_l": axis_l,
            "axis_r": axis_r,
            "max_eye_z": max_eye_z,
            "left_2d": left,
            "right_2d": right,
        }

    ear_z_min = None
    if has_protruding_ears:
        floor = (eye_info["max_eye_z"] + (eye_info["R_dome"] / dz) * 0.75) if eye_info else 0.82
        ear_z_min = max(float(floor), 0.84)

    return {
        "has_eyes": eye_info is not None,
        "eye_info": eye_info,
        "has_protruding_ears": has_protruding_ears,
        "ear_z_min": ear_z_min,
    }


def clean_thin_webbing_fins(mesh_obj: bpy.types.Object, thickness_ratio: float = 0.015) -> int:
    """
    通用 3D 局部薄片厚度檢測與蹼狀物切除 (Ray-cast Thickness De-Webbing)：
    1. 沿頂點內向法線 -v.normal 執行 BVHTree 射線投射，測量局部前後厚度。
    2. 識別 inward_thick < thickness_ratio * scale 的極薄頂點，進行 2 圈拓撲擴展後做連通分量分析。
    3. 僅鎖定中位厚度極薄 (< 0.014 * scale)、頂點數 >= 25 且非耳尖的薄片/蹼狀物 (如懸空鬍鬚蹼)。
    4. 乾淨切除懸空薄片，使用邊界環縫合與局部細分修補，並在臉頰根部執行區域受控拉普拉斯平滑。
    回傳被清除的蹼狀頂點總數。
    """
    mesh = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm, epsilon=1e-5)

    n_verts = len(bm.verts)
    coords = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float32)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    dx = max(1e-5, float(x_max - x_min))
    dy = max(1e-5, float(y_max - y_min))
    dz = max(1e-5, float(z_max - z_min))
    scale = max(dx, dy, dz)
    eps = 0.002 * scale

    inward_thick = np.full(n_verts, 999.0, dtype=np.float32)
    for i, v in enumerate(bm.verts):
        hit, _, _, dist = bvh.ray_cast(v.co - v.normal * eps, -v.normal, 2.0)
        if hit is not None:
            inward_thick[i] = dist

    is_thin = inward_thick < (thickness_ratio * scale)
    thin_dilated = is_thin.copy()
    for _ in range(2):
        nxt = thin_dilated.copy()
        for v in bm.verts:
            if thin_dilated[v.index]:
                for e in v.link_edges:
                    nxt[e.other_vert(v).index] = True
        thin_dilated = nxt

    visited = np.zeros(n_verts, dtype=bool)
    webbing_mask = np.zeros(n_verts, dtype=bool)
    for i in range(n_verts):
        if thin_dilated[i] and not visited[i]:
            q = deque([i])
            visited[i] = True
            comp = []
            while q:
                curr = q.popleft()
                comp.append(curr)
                for e in bm.verts[curr].link_edges:
                    nb = e.other_vert(bm.verts[curr]).index
                    if thin_dilated[nb] and not visited[nb]:
                        visited[nb] = True
                        q.append(nb)
            # 必須為薄片且非高處立耳尖端 (z_rel < 0.80)
            comp_pts = coords[comp]
            z_rel_comp = (comp_pts[:, 2] - z_min) / dz
            if len(comp) >= 25 and np.median(inward_thick[comp]) < (0.014 * scale) and z_rel_comp.mean() < 0.80:
                webbing_mask[comp] = True

    total_web_verts = int(np.count_nonzero(webbing_mask))
    if total_web_verts == 0:
        bm.free()
        return 0

    web_pts = coords[webbing_mask]
    web_z_max = float(web_pts[:, 2].max()) + 0.015 * scale

    # 向外擴展 1 圈包含根部折角
    for _ in range(1):
        nxt = webbing_mask.copy()
        for v in bm.verts:
            if webbing_mask[v.index]:
                for e in v.link_edges:
                    nxt[e.other_vert(v).index] = True
        webbing_mask = nxt

    orig_surviving_coords = {
        (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5))
        for v in bm.verts if not webbing_mask[v.index]
    }

    verts_to_del = [v for v in bm.verts if webbing_mask[v.index]]
    bmesh.ops.delete(bm, geom=verts_to_del, context="VERTS")
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # 清理孤立微碎片 (< 100 頂點)
    visited_v = set()
    for v in list(bm.verts):
        if v.is_valid and v not in visited_v:
            q = deque([v])
            visited_v.add(v)
            comp_v = []
            while q:
                curr = q.popleft()
                comp_v.append(curr)
                for e in curr.link_edges:
                    nb = e.other_vert(curr)
                    if nb not in visited_v:
                        visited_v.add(nb)
                        q.append(nb)
            if len(comp_v) < 100:
                bmesh.ops.delete(bm, geom=comp_v, context="VERTS")

    bm.to_mesh(mesh)
    bm.free()

    # 使用 Edit Mode 高效三角化縫合與局部細分
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.fill()
    bpy.ops.mesh.beautify_fill()
    bpy.ops.mesh.subdivide(number_cuts=3, smoothness=0.0)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    # 在 BMesh 中鎖定修補區域 + 周圍 6 圈頂點，嚴格受控於雙眼下方 (z <= web_z_max) 與遠離鼻中線進行平滑
    bm2 = bmesh.new()
    bm2.from_mesh(mesh)
    bm2.verts.ensure_lookup_table()

    patch_verts = {
        v for v in bm2.verts
        if (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5)) not in orig_surviving_coords
    }
    smooth_verts = set(patch_verts)
    for _ in range(6):
        for v in list(smooth_verts):
            for e in v.link_edges:
                nb = e.other_vert(v)
                if nb.co.z <= web_z_max and abs(nb.co.x) >= 0.085 * dx:
                    smooth_verts.add(nb)

    smooth_list = list(smooth_verts)
    for _ in range(45):
        old_co = {v: v.co.copy() for v in smooth_list}
        for v in smooth_list:
            nbs = [e.other_vert(v) for e in v.link_edges]
            if nbs:
                avg = sum((old_co.get(n, n.co) for n in nbs), Vector((0.0, 0.0, 0.0))) / len(nbs)
                v.co = old_co[v] * 0.20 + avg * 0.80

    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces[:])
    bm2.to_mesh(mesh)
    bm2.free()

    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.update()

    # 清除舊 UV，確保縫合處與平滑後的臉頰重新展開均勻 UV
    while len(mesh.uv_layers) > 0:
        mesh.uv_layers.remove(mesh.uv_layers[0])

    return total_web_verts


def main() -> None:
    args = parse_args()
    style_cfg = get_style(args.style)
    is_fur = (style_cfg.id == "fur")
    effective_roughness = float(args.roughness) if args.roughness is not None else float(style_cfg.roughness)
    effective_ao_strength = float(args.ao_strength) if args.ao_strength is not None else float(style_cfg.ao_strength)
    effective_normal_strength = float(args.normal_strength) if args.normal_strength is not None else (0.75 if is_fur else 0.35)

    mesh_path = Path(args.mesh).resolve()
    front_path = Path(args.front).resolve()
    back_path = Path(args.back).resolve()
    side_path = Path(args.side).resolve()
    out_path = Path(args.output).resolve()

    print("============================================================")
    print("🎨 啟動遮擋感知多視角 PBR 貼圖烘焙管線 (Occlusion-Aware PBR Baker)")
    print(f"   幾何網格: {mesh_path.name}")
    print(f"   視角圖檔: 正面={front_path.name}, 背面={back_path.name}, 側面={side_path.name}")
    print(f"   目標輸出: {out_path}")
    print(f"   🎨 套用風格: {style_cfg.name} (ID: {style_cfg.id})")
    print(f"      - 粗糙度 (Roughness): {effective_roughness:.2f}{' (手動覆寫)' if args.roughness is not None else ''}")
    print(f"      - 金屬度 (Metallic): {style_cfg.metallic:.2f}")
    print(f"      - 高光強度 (Specular IOR Level): {style_cfg.specular_ior_level:.2f}")
    print(f"      - AO 融合強度: {effective_ao_strength:.2f}{' (手動覆寫)' if args.ao_strength is not None else ''} (底限 Clamp: {style_cfg.ao_min_clamp:.2f})")
    print(f"      - 法線烘焙 (Normal Map): {'開啟 (強度=' + str(effective_normal_strength) + ')' if args.bake_normal else '關閉'}")
    print(f"      - 多視角防重影銳化 (Sharpen Blending): {'開啟' if args.sharpen_blending else '關閉'}")
    print("============================================================")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"在 {mesh_path} 中找不到任何 MESH 物件！")

    # 若有多個網格物件則合併為一體
    bpy.context.view_layer.objects.active = mesh_objs[0]
    for o in mesh_objs:
        o.select_set(True)
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    mesh_obj = bpy.context.view_layer.objects.active
    mesh = mesh_obj.data

    if mesh.has_custom_normals:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()

    # 自動檢測並切除懸空薄片與蹼狀鬍鬚黏連 (De-webbing)
    if args.clean_webbing:
        n_cleaned = clean_thin_webbing_fins(mesh_obj, thickness_ratio=0.015)
        if n_cleaned > 0:
            print(f"✨ [通用去蹼] 成功切除 {n_cleaned} 個懸空薄片/殘蹼頂點，並完成平滑臉頰曲面修補！")

    # 焊接重疊頂點並平滑法線，確保網格拓撲連通性以支援拉普拉斯擴散
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    for poly in mesh.polygons:
        poly.use_smooth = True

    # 檢查是否已有展開的 UV，若無則使用 Blender 高速 Smart UV Project
    has_valid_uv = False
    if len(mesh.uv_layers) > 0:
        uv_data = np.array([d.uv for d in mesh.uv_layers.active.data], dtype=np.float32)
        if len(uv_data) > 0 and (uv_data.max() - uv_data.min()) > 0.1:
            has_valid_uv = True

    if not has_valid_uv:
        print("🗺️  檢測到模型未包含 UV，正在執行高速 Smart UV Unwrap...")
        if len(mesh.uv_layers) == 0:
            mesh.uv_layers.new(name="UVMap")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(52.0), island_margin=0.008)
        bpy.ops.uv.pack_islands(margin=0.008)
        bpy.ops.object.mode_set(mode="OBJECT")
        print("✅ Smart UV 展開完成！(角度閾值: 52.0° + Pack Islands)")

    orig_uv_name = mesh.uv_layers.active.name

    # 建立 BMesh 與 BVHTree 用於 3D 光線遮擋檢測與鄰接圖建立
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm, epsilon=1e-5)

    n_verts = len(bm.verts)
    coords = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float32)
    normals = np.array([[v.normal.x, v.normal.y, v.normal.z] for v in bm.verts], dtype=np.float32)

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    dx = max(1e-5, float(x_max - x_min))
    dy = max(1e-5, float(y_max - y_min))
    dz = max(1e-5, float(z_max - z_min))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)

    print(f"📐 網格統計: 頂點={n_verts}, 三角面={len(bm.faces)} | 尺寸: 寬={dx:.3f}, 深={dy:.3f}, 高={dz:.3f}")

    # 讀取三視圖並計算無白邊擴展色彩與輪廓內縮距離場
    px_f, dist_f, (uf_c, uf_w, vf_c, vf_h) = load_view_and_distance_field(front_path)
    px_b, dist_b, (ub_c, ub_w, vb_c, vb_h) = load_view_and_distance_field(back_path)
    px_s, dist_s, (us_c, us_w, vs_c, vs_h) = load_view_and_distance_field(side_path)

    z_rel = (coords[:, 2] - z_min) / dz
    y_rel = (coords[:, 1] - cy) / dy
    x_rel = (coords[:, 0] - cx) / dx

    # 自動偵測雙眼五官與幾何耳根/鞍部分界
    det_res = detect_eyes_and_ears_universal(
        px_f, dist_f, (uf_c, uf_w, vf_c, vf_h),
        coords, normals, dx, dy, dz, cx, cy, cz, x_rel, y_rel, z_rel
    )
    eye_info = det_res["eye_info"] if args.eye_refine else None
    has_protruding_ears = det_res["has_protruding_ears"]
    ear_z_min = det_res["ear_z_min"]

    if eye_info is not None:
        print(f"👁️  [自適應雙眼感知] 偵測到雙眼五官: L={eye_info['center_l'].round(3)}, R={eye_info['center_r'].round(3)}, R_dome={eye_info['R_dome']:.4f}")
    if has_protruding_ears:
        print(f"👂 [自適應耳朵感知] 檢測到角色立耳，自動幾何耳根分界線: z_rel > {ear_z_min:.3f}")
    else:
        print(f"👤 [自適應耳朵感知] 檢測為圓頭/無立耳造型，自動停用外耳背強制遮罩")

    # 對 2D 正面圖雙眼 ROI 施加 100% 連續平滑的深邃黑瞳與全色相虹膜增艷 (絕無二值化硬邊鋸齒)
    if eye_info is not None and args.eye_refine:
        ih, iw, _ = px_f.shape
        for eye_data in [eye_info["left_2d"], eye_info["right_2d"]]:
            cy_e, cx_e = eye_data["cy"], eye_data["cx"]
            r_box = int(eye_data["rad_px"] * 1.85)
            y0_e = max(0, cy_e - r_box)
            y1_e = min(ih, cy_e + r_box)
            x0_e = max(0, cx_e - r_box)
            x1_e = min(iw, cx_e + r_box)
            roi = px_f[y0_e:y1_e, x0_e:x1_e]

            roi_luma = 0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]
            dark_factor = 0.30 + 0.70 * np.clip((roi_luma - 0.015) / 0.055, 0.0, 1.0)
            roi[:] = roi * dark_factor[..., None]

            c_max = np.maximum(roi[..., 0], np.maximum(roi[..., 1], roi[..., 2]))
            c_min = np.minimum(roi[..., 0], np.minimum(roi[..., 1], roi[..., 2]))
            chroma = c_max - c_min
            chroma_factor = np.clip(chroma / 0.18, 0.0, 1.0)
            mean_c = roi.mean(axis=-1, keepdims=True)
            sat_boost = 1.0 + 0.40 * chroma_factor[..., None]
            roi[:] = np.clip(mean_c + (roi - mean_c) * sat_boost, 0.0, 1.0)

    # 毛皮風格深灰濃郁度加深 (僅作用於非眼球的毛皮區域，完整保護雙眼與翡翠綠光環)
    fur_gamma = float(args.fur_gamma) if args.fur_gamma is not None else (1.50 if is_fur else 1.0)
    if fur_gamma != 1.0:
        print(f"🧶 [毛皮深灰質感調節] 套用毛皮 Gamma={fur_gamma:.2f} (深邃厚實灰階，雙眼高光全保護)")
        ih, iw, _ = px_f.shape
        eye_protect = np.zeros((ih, iw), dtype=np.float32)
        if eye_info is not None:
            y_coords, x_coords = np.ogrid[:ih, :iw]
            for ed in [eye_info["left_2d"], eye_info["right_2d"]]:
                d2 = (y_coords - ed["cy"]) ** 2 + (x_coords - ed["cx"]) ** 2
                r_in = ed["rad_px"] * 1.15
                r_out = ed["rad_px"] * 1.55
                w_ed = np.clip((r_out - np.sqrt(d2)) / max(1.0, r_out - r_in), 0.0, 1.0)
                eye_protect = np.maximum(eye_protect, w_ed)
        g_map = 1.0 + (fur_gamma - 1.0) * (1.0 - eye_protect[..., None])
        px_f = np.clip(px_f ** g_map, 0.0, 1.0)
        px_b = np.clip(px_b ** fur_gamma, 0.0, 1.0)
        px_s = np.clip(px_s ** fur_gamma, 0.0, 1.0)

    # 計算每個頂點在正、背、側視角的投影 UV
    u_f = uf_c + ((coords[:, 0] - cx) / dx) * uf_w
    v_f = vf_c + ((coords[:, 2] - cz) / dz) * vf_h

    # 背面投影：對臀部區域消除 2D 尾巴殘影 (若 x_rel < 0，採用右側無尾臀部投影色彩)
    is_butt_tail_leak = (z_rel > 0.15) & (z_rel < 0.42) & (y_rel > 0.02) & (x_rel < -0.05) & (normals[:, 1] > 0.1)
    coords_b_x = coords[:, 0].copy()
    coords_b_x[is_butt_tail_leak] = cx + np.abs(coords[is_butt_tail_leak, 0] - cx)

    u_b = ub_c - ((coords_b_x - cx) / dx) * ub_w
    v_b = vb_c + ((coords[:, 2] - cz) / dz) * vb_h

    u_s = us_c + ((coords[:, 1] - cy) / dy) * us_w
    v_s = vs_c + ((coords[:, 2] - cz) / dz) * vs_h

    # 採樣各視角色彩與輪廓內縮權重
    col_f = sample_bilinear_array(px_f, u_f, v_f)
    sil_f = sample_bilinear_array(dist_f, u_f, v_f)

    col_b = sample_bilinear_array(px_b, u_b, v_b)
    sil_b = sample_bilinear_array(dist_b, u_b, v_b)

    col_s = sample_bilinear_array(px_s, u_s, v_s)
    sil_s = sample_bilinear_array(dist_s, u_s, v_s)

    # ============================================================
    # 步驟 1: 3D BVHTree 光線投射自遮擋檢測 (Ray-cast Visibility)
    # ============================================================
    print("🔍 [1/4] 正在執行 3D BVHTree 光線投射遮擋檢測 (剔除手臂/背刺遮擋重影)...")
    vis_f = np.zeros(n_verts, dtype=np.float32)
    vis_b = np.zeros(n_verts, dtype=np.float32)
    vis_l = np.zeros(n_verts, dtype=np.float32)
    vis_r = np.zeros(n_verts, dtype=np.float32)

    dir_front = Vector((0.0, -1.0, 0.0))
    dir_back = Vector((0.0, 1.0, 0.0))
    dir_left = Vector((-1.0, 0.0, 0.0))
    dir_right = Vector((1.0, 0.0, 0.0))
    ray_eps = 0.004 * max(dx, dy, dz)

    for i, v in enumerate(bm.verts):
        origin = v.co + v.normal * ray_eps
        if v.normal.y < -0.05:
            # 頭部/面部在正面絕對不會被身體自遮擋 (杜絕被 3D 胡鬚阻擋導致的臉頰黑洞破綻)
            if z_rel[i] > 0.45 and y_rel[i] < 0.0:
                vis_f[i] = 1.0
            else:
                hit, _, _, _ = bvh.ray_cast(origin, dir_front, 10.0)
                if hit is None:
                    vis_f[i] = 1.0
        if v.normal.y > 0.05:
            hit, _, _, _ = bvh.ray_cast(origin, dir_back, 10.0)
            if hit is None:
                vis_b[i] = 1.0
        if v.normal.x < -0.05:
            hit, _, _, _ = bvh.ray_cast(origin, dir_left, 10.0)
            if hit is None:
                vis_l[i] = 1.0
        if v.normal.x > 0.05:
            hit, _, _, _ = bvh.ray_cast(origin, dir_right, 10.0)
            if hit is None:
                vis_r[i] = 1.0

    # 計算各視角法線入射角權重 (Grazing-Angle Suppression: cos(theta) < 0.25 平滑歸零)
    ang_f = np.clip((-normals[:, 1] - 0.12) / 0.55, 0.0, 1.0) ** 1.3
    ang_b = np.clip((normals[:, 1] - 0.12) / 0.55, 0.0, 1.0) ** 1.3
    ang_l = np.clip((-normals[:, 0] - 0.32) / 0.50, 0.0, 1.0) ** 1.6
    ang_r = np.clip((normals[:, 0] - 0.32) / 0.50, 0.0, 1.0) ** 1.6

    # 正面權重予以 2.4 倍優先權以完美保護五官、腮紅與肚皮邊界
    w_f = 2.4 * vis_f * ang_f * (sil_f ** 1.0)
    w_b = 1.8 * vis_b * ang_b * (sil_b ** 1.0)
    w_s = 0.95 * (vis_l * ang_l + vis_r * ang_r) * (sil_s ** 1.3)

    # 在側視圖中，角色下垂的手臂/翅膀通常畫在軀幹中段偏前側 (z_rel 介於 0.18~0.56、y_rel < 0.15)
    # 抑制該區域軀幹側面的側視圖權重，改由正背面與拉普拉斯諧波平滑過渡，徹底消除側腹「假手臂/假黑翅」投影印記
    side_arm_shadow = (z_rel > 0.18) & (z_rel < 0.56) & (y_rel < 0.15)
    w_s[side_arm_shadow] = 0.0

    # 在正面臉部與肚皮中央區域 (|x| < 0.32*dx, y < cy, normal.y < -0.35)，強制鎖定純正面視角
    front_lock = (np.abs(coords[:, 0] - cx) < 0.32 * dx) & (coords[:, 1] < cy) & (normals[:, 1] < -0.35) & (sil_f > 0.3)
    if eye_info is not None:
        c_l = eye_info["center_l"]
        c_r = eye_info["center_r"]
        r_d = eye_info["R_dome"]
        dist_l_pts = np.linalg.norm(coords - c_l, axis=1)
        dist_r_pts = np.linalg.norm(coords - c_r, axis=1)
        eye_cheek_lock = ((dist_l_pts < 1.65 * r_d) | (dist_r_pts < 1.65 * r_d)) & (normals[:, 1] < -0.18) & (sil_f > 0.3)
        front_lock = front_lock | eye_cheek_lock

    w_s[front_lock] = 0.0
    w_b[front_lock] = 0.0
    w_f[front_lock] = np.maximum(w_f[front_lock], 2.0)

    # 提取可靠的主體底色 (用於修補未覆蓋區域與附屬肢體，杜絕採樣到背景純白)
    back_upper_mask = (y_rel > 0.04) & (z_rel > 0.45) & (z_rel < 0.75) & (normals[:, 1] > 0.2) & (sil_b > 0.5)
    if back_upper_mask.any():
        back_base_col = np.median(col_b[back_upper_mask], axis=0)
    else:
        back_base_col = np.array([0.4, 0.4, 0.4], dtype=np.float32)

    front_mask = (np.abs(x_rel) < 0.25) & (coords[:, 1] < cy) & (normals[:, 1] < -0.35) & (sil_f > 0.5)
    if front_mask.any():
        front_base_col = np.median(col_f[front_mask], axis=0)
    else:
        front_base_col = back_base_col.copy()

    # 抑制後方附屬肢體 (尾巴: z_rel < 0.40 且 y_rel > 0.05 的後方突出部位，完整涵蓋尾根至彎曲尾尖)
    is_tail = (z_rel < 0.40) & (y_rel > 0.05)
    if is_tail.any():
        # 尾巴不接受 2D 視角的直接投影 (因為 2D 背面視角常畫在單側，或被胯下純白背景/正面白肚皮穿透)
        w_f[is_tail] = 0.0
        w_b[is_tail] = 0.0
        w_s[is_tail] = 0.0

    # 3D 鬍鬚/薄面特徵辨識：若未啟用去蹼時保留基礎辨識
    is_whisker = (np.abs(x_rel) > 0.16) & (y_rel < -0.22) & (z_rel > 0.52) & (z_rel < 0.68) if not args.clean_webbing else np.zeros(n_verts, dtype=bool)

    # 眼睛辨識 (用於高光與低粗糙度)
    if eye_info is not None:
        is_eye = ((dist_l_pts < 1.25 * r_d) | (dist_r_pts < 1.25 * r_d)) & (normals[:, 1] < -0.15)
    else:
        is_eye = (np.abs(x_rel) > 0.08) & (np.abs(x_rel) < 0.24) & (z_rel > 0.60) & (z_rel < 0.74) & (y_rel < -0.12) & (normals[:, 1] < -0.2)

    # 鼻頭辨識
    is_nose = (np.abs(x_rel) < 0.06) & (z_rel > 0.58) & (z_rel < 0.64) & (y_rel < -0.22)

    # 耳朵區域幾何自適應分區 (依幾何鞍部高度動態設定，徹底杜絕側面圖滲漏或切掉眼球外緣)
    if has_protruding_ears and ear_z_min is not None:
        is_ear = (z_rel > ear_z_min) & (np.abs(x_rel) > 0.08)
    else:
        is_ear = np.zeros(n_verts, dtype=bool)
    if is_ear.any():
        # 1. 側面圖在 2D 壓平時內耳粉紅容易重疊外緣，全耳禁止側面圖投影
        w_s[is_ear] = 0.0

        # 2. 幾何判別真正的內耳耳窩 (凹面朝前偏內: x_rel<0 時 norm_x>-0.32 且 norm_y<-0.15; x_rel>0 時 norm_x<0.32 且 norm_y<-0.15)
        inner_l = (x_rel < -0.08) & (normals[:, 0] > -0.32) & (normals[:, 1] < -0.15) & is_ear
        inner_r = (x_rel > 0.08) & (normals[:, 0] < 0.32) & (normals[:, 1] < -0.15) & is_ear
        is_inner_ear = inner_l | inner_r
        is_outer_ear = is_ear & ~is_inner_ear

        # 內耳耳窩由正面圖投影粉紅耳窩
        w_b[is_inner_ear] = 0.0
        w_f[is_inner_ear] = np.maximum(w_f[is_inner_ear], 2.5)

        # 耳背、外緣輪廓與頂端 (is_outer_ear)：絕對禁止正面投影 (根絕正面粉紅外溢)，鎖定由背面灰毛覆蓋
        w_f[is_outer_ear] = 0.0
        w_b[is_outer_ear] = np.maximum(w_b[is_outer_ear], 2.4)

    # 抑制小腿後側/雙腿內側鄰近胯下純白背景處的背面投影溢色
    # 若背面法向朝後 (normals[:, 1] > 0.1) 但投影極靠近輪廓外緣 (sil_b < 0.25)，將背面權重歸零，交由周圍腿部諧波自然過渡
    inner_leg_leak = (z_rel < 0.35) & (y_rel < 0.06) & (normals[:, 1] > 0.1) & (sil_b < 0.25)
    w_b[inner_leg_leak] = 0.0
    w_f[inner_leg_leak] = 0.0

    w_sum = w_f + w_b + w_s
    confidence = np.clip(w_sum / 0.60, 0.0, 1.0)

    # 非線性指數銳化權重分配 (Winner-Takes-Most Sharpening):
    # p=2.6 消除毛髮重影且避免三角面切角鋸齒 (Faceting)
    if args.sharpen_blending:
        p_exp = 2.6 if is_fur else 3.0
        w_f_p = w_f ** p_exp
        w_b_p = w_b ** p_exp
        w_s_p = w_s ** p_exp
        w_p_sum = w_f_p + w_b_p + w_s_p
        mask_nz = w_p_sum > 1e-6
        norm_factor = np.zeros_like(w_sum)
        norm_factor[mask_nz] = w_sum[mask_nz] / w_p_sum[mask_nz]
        w_f = w_f_p * norm_factor
        w_b = w_b_p * norm_factor
        w_s = w_s_p * norm_factor

    # 合成直接可見頂點色彩
    safe_sum = np.maximum(w_sum, 1e-6)[:, None]
    direct_color = (col_f * w_f[:, None] + col_b * w_b[:, None] + col_s * w_s[:, None]) / safe_sum

    # 對於無直接視角覆蓋或被抑制的頂點，使用主體區域底色作為初始種子，徹底根除純白背景滲色
    fallback_fb = np.clip((normals[:, 1] + 0.1) / 0.2, 0.0, 1.0)[:, None]
    fallback_col = front_base_col * (1.0 - fallback_fb) + back_base_col * fallback_fb
    v_color = direct_color * confidence[:, None] + fallback_col * (1.0 - confidence[:, None])

    # 尾巴精修紋理：採樣背部與側身的高解析深灰毛皮材質，並賦予沿尾巴生長方向的柔和漸層與微紋理
    if is_tail.any():
        tail_z = z_rel[is_tail]
        tail_y = y_rel[is_tail]
        tail_curve = np.sin(tail_z * 18.0 + tail_y * 12.0) * 0.04
        tail_color_gradient = back_base_col * (0.92 + tail_curve[:, None])
        v_color[is_tail] = np.clip(tail_color_gradient, 0.0, 1.0)

    # 3D 鬍鬚精修：純白乾淨高光
    if is_whisker.any():
        v_color[is_whisker] = np.array([0.96, 0.96, 0.96], dtype=np.float32)

    # ============================================================
    # 步驟 2: 3D 網格拉普拉斯諧波修補 (Mesh Harmonic Inpainting)
    # ============================================================
    print("🌊 [2/4] 正在執行 3D 網格拓撲拉普拉斯諧波色彩擴散 (填補刺間深溝與腋下遮擋區)...")
    edges_u = []
    edges_v = []
    for e in bm.edges:
        u_i = e.verts[0].index
        v_i = e.verts[1].index
        edges_u.extend([u_i, v_i])
        edges_v.extend([v_i, u_i])
    edges_u_arr = np.array(edges_u, dtype=np.int32)
    edges_v_arr = np.array(edges_v, dtype=np.int32)

    deg = np.bincount(edges_u_arr, minlength=n_verts).astype(np.float32)
    deg = np.maximum(deg, 1.0)[:, None]

    anchor_color = direct_color.copy()
    anchor_weight = (confidence ** 1.3)[:, None]
    if is_tail.any():
        anchor_color[is_tail] = v_color[is_tail]
        anchor_weight[is_tail] = 1.0
    if is_whisker.any():
        anchor_color[is_whisker] = np.array([0.96, 0.96, 0.96], dtype=np.float32)
        anchor_weight[is_whisker] = 1.0

    # 嚴格校驗耳朵外緣與耳背色彩：若有任何粉紅色相 (R > 0.36 且 R > 1.15 * G)，強制替換為背部純淨灰毛
    if is_ear.any():
        is_leaked_pink = is_outer_ear & (v_color[:, 0] > 0.36) & (v_color[:, 0] > v_color[:, 1] * 1.15)
        if is_leaked_pink.any():
            v_color[is_leaked_pink] = back_base_col
            anchor_color[is_leaked_pink] = back_base_col
            anchor_weight[is_leaked_pink] = 1.0

    for _ in range(36):
        nb_sum = np.zeros_like(v_color)
        np.add.at(nb_sum, edges_u_arr, v_color[edges_v_arr])
        nb_avg = nb_sum / deg
        v_color = anchor_color * anchor_weight + nb_avg * (1.0 - anchor_weight)

    # 將修復後的頂點顏色與視角混合權重寫入 BMesh Color Attributes，供 Shader 高清混合使用
    col_layer = bm.loops.layers.color.new("HarmonicColor")
    weight_layer = bm.loops.layers.color.new("ViewWeights")
    uv_f_lay = bm.loops.layers.uv.new("UV_Front")
    uv_b_lay = bm.loops.layers.uv.new("UV_Back")
    uv_s_lay = bm.loops.layers.uv.new("UV_Side")

    norm_wf = (w_f / np.maximum(w_sum, 1e-5)) * confidence
    norm_wb = (w_b / np.maximum(w_sum, 1e-5)) * confidence
    norm_ws = (w_s / np.maximum(w_sum, 1e-5)) * confidence

    # 針對耳朵正背面進行 Shader 權重嚴格隔離，杜絕任何視角重疊滲色
    if is_ear.any():
        norm_wf[is_outer_ear] = 0.0
        norm_ws[is_outer_ear] = 0.0
        norm_wb[is_outer_ear] = 1.0

        norm_wb[is_inner_ear] = 0.0
        norm_ws[is_inner_ear] = 0.0
        norm_wf[is_inner_ear] = 1.0

    for face in bm.faces:
        for loop in face.loops:
            vi = loop.vert.index
            c = v_color[vi]
            loop[col_layer] = (float(c[0]), float(c[1]), float(c[2]), 1.0)
            loop[weight_layer] = (float(norm_wf[vi]), float(norm_wb[vi]), float(norm_ws[vi]), float(confidence[vi]))
            loop[uv_f_lay].uv = (float(u_f[vi]), float(v_f[vi]))
            loop[uv_b_lay].uv = (float(u_b[vi]), float(v_b[vi]))
            loop[uv_s_lay].uv = (float(u_s[vi]), float(v_s[vi]))

    bm.to_mesh(mesh)
    bm.free()

    # 將無白邊擴展後的視角圖載入 Blender Image (確保色彩空間為 Linear Rec.709，防止二次 Gamma 壓暗)
    def array_to_bpy_image(name: str, arr: np.ndarray) -> bpy.types.Image:
        ih, iw, _ = arr.shape
        im = bpy.data.images.new(name, width=iw, height=ih, alpha=True)
        im.colorspace_settings.name = "Linear Rec.709"
        rgba = np.ones((ih, iw, 4), dtype=np.float32)
        rgba[:, :, :3] = np.clip(arr, 0.0, 1.0)
        im.pixels[:] = rgba.flatten().tolist()
        return im

    img_front = array_to_bpy_image("DilatedFront", px_f)
    img_back = array_to_bpy_image("DilatedBack", px_b)
    img_side = array_to_bpy_image("DilatedSide", px_s)

    # ============================================================
    # 步驟 3: Cycles AO 環境光遮蔽烘焙
    # ============================================================
    print(f"🔄 [3/4] 正在烘焙 {args.texture_size}x{args.texture_size} 幾何環境光遮蔽圖譜 (Cycles AO)...")
    mesh.uv_layers[orig_uv_name].active = True
    mesh.uv_layers[orig_uv_name].active_render = True

    ao_mat = bpy.data.materials.new("BakeAOMat")
    ao_mat.use_nodes = True
    ao_nodes = ao_mat.node_tree.nodes
    ao_nodes.clear()

    ao_img = bpy.data.images.new("BakedAOTex", width=args.texture_size, height=args.texture_size)
    ao_tex_target = ao_nodes.new("ShaderNodeTexImage")
    ao_tex_target.image = ao_img
    ao_nodes.active = ao_tex_target

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(ao_mat)
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.bake_type = "AO"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.render.bake.margin = 6
    bpy.context.scene.render.bake.max_ray_distance = 0.15 * max(dx, dy, dz)
    bpy.ops.object.bake(type="AO")

    # ============================================================
    # 步驟 4: 結合高解析度相機投影 + 諧波頂點色修補 + AO 烘焙最終貼圖
    # ============================================================
    print(f"🔄 [4/4] 正在合成高解析度無條紋 Albedo 與分區 Roughness PBR 貼圖...")
    mat = bpy.data.materials.new("ProjectionBakeMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    def make_tex_branch(uv_name: str, img: bpy.types.Image) -> bpy.types.Node:
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.uv_map = uv_name
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.extension = "EXTEND"
        tex_node.interpolation = "Cubic"
        links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
        return tex_node

    tex_f = make_tex_branch("UV_Front", img_front)
    tex_b = make_tex_branch("UV_Back", img_back)
    tex_s = make_tex_branch("UV_Side", img_side)

    vcol_harm = nodes.new("ShaderNodeVertexColor")
    vcol_harm.layer_name = "HarmonicColor"

    vcol_w = nodes.new("ShaderNodeVertexColor")
    vcol_w.layer_name = "ViewWeights"

    sep_w = nodes.new("ShaderNodeSeparateColor")
    links.new(vcol_w.outputs["Color"], sep_w.inputs["Color"])

    # 以 HarmonicColor 為基底，按遮擋感知權重疊加 Front / Back / Side 的 2K 銳利像素細節
    # Color = Harmonic * (1 - w_f - w_b - w_s) + Front * w_f + Back * w_b + Side * w_s
    mix1 = nodes.new("ShaderNodeMix")
    mix1.data_type = "RGBA"
    links.new(sep_w.outputs["Red"], mix1.inputs[0])  # w_f
    links.new(vcol_harm.outputs["Color"], mix1.inputs[6])
    links.new(tex_f.outputs["Color"], mix1.inputs[7])

    mix2 = nodes.new("ShaderNodeMix")
    mix2.data_type = "RGBA"
    links.new(sep_w.outputs["Green"], mix2.inputs[0])  # w_b
    links.new(mix1.outputs[2], mix2.inputs[6])
    links.new(tex_b.outputs["Color"], mix2.inputs[7])

    mix3 = nodes.new("ShaderNodeMix")
    mix3.data_type = "RGBA"
    links.new(sep_w.outputs["Blue"], mix3.inputs[0])  # w_s
    links.new(mix2.outputs[2], mix3.inputs[6])
    links.new(tex_s.outputs["Color"], mix3.inputs[7])

    # 融入柔和 AO
    ao_sample_node = nodes.new("ShaderNodeTexImage")
    ao_sample_node.image = ao_img
    ao_uv_node = nodes.new("ShaderNodeUVMap")
    ao_uv_node.uv_map = orig_uv_name
    links.new(ao_uv_node.outputs["UV"], ao_sample_node.inputs["Vector"])

    map_ao = nodes.new("ShaderNodeMapRange")
    map_ao.inputs["From Min"].default_value = 0.0
    map_ao.inputs["From Max"].default_value = 1.0
    map_ao.inputs["To Min"].default_value = 0.55 if is_fur else max(float(style_cfg.ao_min_clamp), 1.0 - float(effective_ao_strength))
    map_ao.inputs["To Max"].default_value = 1.0
    map_ao.clamp = True
    links.new(ao_sample_node.outputs["Color"], map_ao.inputs["Value"])

    mix_ao = nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mix_ao.inputs[0].default_value = 0.82 if is_fur else 1.0
    links.new(mix3.outputs[2], mix_ao.inputs[6])
    links.new(map_ao.outputs["Result"], mix_ao.inputs[7])

    emit_node = nodes.new("ShaderNodeEmission")
    links.new(mix_ao.outputs[2], emit_node.inputs["Color"])

    out_node = nodes.new("ShaderNodeOutputMaterial")
    links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    baked_img = bpy.data.images.new("BakedPBRTexture", width=args.texture_size, height=args.texture_size)
    baked_tex_node = nodes.new("ShaderNodeTexImage")
    baked_tex_node.image = baked_img
    nodes.active = baked_tex_node

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)

    bpy.context.scene.cycles.bake_type = "EMIT"
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.render.bake.margin = 6
    bpy.ops.object.bake(type="EMIT")

    baked_tex_path = out_path.parent / f"{out_path.stem}_diffuse.png"
    baked_img.filepath_raw = str(baked_tex_path)
    baked_img.file_format = "PNG"
    baked_img.save()
    print(f"💾 已儲存無條紋烘焙貼圖至: {baked_tex_path}")

    # ============================================================
    # 步驟 5: 高頻毛流微細節與切線空間法線貼圖烘焙 (Fur Normal Map)
    # ============================================================
    baked_norm_img = None
    if args.bake_normal:
        print(f"🧶 [5/5] 正在計算高頻毛流微細節並烘焙 {args.texture_size}x{args.texture_size} 切線空間法線貼圖 (Tangent Normal)...")
        w_tex, h_tex = baked_img.size
        px_albedo = np.array(baked_img.pixels[:], dtype=np.float32).reshape(h_tex, w_tex, 4)[:, :, :3]
        luma_tex = 0.299 * px_albedo[:, :, 0] + 0.587 * px_albedo[:, :, 1] + 0.114 * px_albedo[:, :, 2]

        def fast_box_blur_2d(arr: np.ndarray, r: int = 6) -> np.ndarray:
            p1 = np.pad(arr, ((0, 0), (r, r)), mode="edge")
            c1 = np.cumsum(p1, axis=1)
            b1 = (c1[:, 2 * r :] - c1[:, :-2 * r]) / (2 * r)
            p0 = np.pad(b1, ((r, r), (0, 0)), mode="edge")
            c0 = np.cumsum(p0, axis=0)
            return (c0[2 * r :, :] - c0[:-2 * r, :]) / (2 * r)

        blur_radius = 8 if is_fur else 6
        low_freq = fast_box_blur_2d(fast_box_blur_2d(luma_tex, r=blur_radius), r=blur_radius)
        high_pass = np.clip(luma_tex - low_freq + 0.5, 0.0, 1.0)

        height_img = bpy.data.images.new("FurHeightMap", width=w_tex, height=h_tex)
        height_rgba = np.ones((h_tex, w_tex, 4), dtype=np.float32)
        height_rgba[:, :, 0] = high_pass
        height_rgba[:, :, 1] = high_pass
        height_rgba[:, :, 2] = high_pass
        height_img.pixels[:] = height_rgba.flatten().tolist()

        norm_bake_mat = bpy.data.materials.new("BakeNormalMat")
        norm_bake_mat.use_nodes = True
        n_nodes = norm_bake_mat.node_tree.nodes
        n_links = norm_bake_mat.node_tree.links
        n_nodes.clear()

        bsdf_nb = n_nodes.new("ShaderNodeBsdfPrincipled")
        h_tex_node = n_nodes.new("ShaderNodeTexImage")
        h_tex_node.image = height_img
        h_uv = n_nodes.new("ShaderNodeUVMap")
        h_uv.uv_map = orig_uv_name
        n_links.new(h_uv.outputs["UV"], h_tex_node.inputs["Vector"])

        bump_nb = n_nodes.new("ShaderNodeBump")
        bump_nb.inputs["Strength"].default_value = float(effective_normal_strength)
        bump_nb.inputs["Distance"].default_value = 0.035 if is_fur else 0.02
        n_links.new(h_tex_node.outputs["Color"], bump_nb.inputs["Height"])
        n_links.new(bump_nb.outputs["Normal"], bsdf_nb.inputs["Normal"])

        baked_norm_img = bpy.data.images.new("BakedNormalTexture", width=args.texture_size, height=args.texture_size)
        baked_norm_node = n_nodes.new("ShaderNodeTexImage")
        baked_norm_node.image = baked_norm_img
        n_nodes.active = baked_norm_node

        out_nb = n_nodes.new("ShaderNodeOutputMaterial")
        n_links.new(bsdf_nb.outputs["BSDF"], out_nb.inputs["Surface"])

        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(norm_bake_mat)
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)

        bpy.context.scene.cycles.bake_type = "NORMAL"
        bpy.context.scene.render.bake.normal_space = "TANGENT"
        bpy.context.scene.render.bake.margin = 6
        bpy.ops.object.bake(type="NORMAL")

        baked_norm_path = out_path.parent / f"{out_path.stem}_normal.png"
        baked_norm_img.filepath_raw = str(baked_norm_path)
        baked_norm_img.file_format = "PNG"
        baked_norm_img.save()
        print(f"💾 已儲存切線空間法線貼圖至: {baked_norm_path}")

        bpy.data.images.remove(height_img)
        bpy.data.materials.remove(norm_bake_mat)

    # 烘焙分區粗糙度貼圖 (Roughness Map: 身體毛皮 0.95 vs 眼睛 0.12 / 鼻頭 0.22)
    baked_rough_img = None
    if is_fur and is_eye.any():
        print(f"✨ 正在烘焙分區粗糙度貼圖 (眼睛高光/毛皮啞光)...")
        bm_r = bmesh.new()
        bm_r.from_mesh(mesh)
        bm_r.verts.ensure_lookup_table()
        rough_attr = bm_r.loops.layers.color.new("LoopRoughness")
        for face in bm_r.faces:
            for loop in face.loops:
                vi = loop.vert.index
                if is_eye[vi]:
                    r_val = 0.04
                elif is_nose[vi]:
                    r_val = 0.22
                else:
                    r_val = float(effective_roughness)
                loop[rough_attr] = (r_val, r_val, r_val, 1.0)
        bm_r.to_mesh(mesh)
        bm_r.free()

        rough_mat = bpy.data.materials.new("BakeRoughMat")
        rough_mat.use_nodes = True
        r_nodes = rough_mat.node_tree.nodes
        r_links = rough_mat.node_tree.links
        r_nodes.clear()

        vcol_r = r_nodes.new("ShaderNodeVertexColor")
        vcol_r.layer_name = "LoopRoughness"
        emit_r = r_nodes.new("ShaderNodeEmission")
        r_links.new(vcol_r.outputs["Color"], emit_r.inputs["Color"])
        out_r = r_nodes.new("ShaderNodeOutputMaterial")
        r_links.new(emit_r.outputs["Emission"], out_r.inputs["Surface"])

        baked_rough_img = bpy.data.images.new("BakedRoughTexture", width=args.texture_size, height=args.texture_size)
        baked_rough_node = r_nodes.new("ShaderNodeTexImage")
        baked_rough_node.image = baked_rough_img
        r_nodes.active = baked_rough_node

        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(rough_mat)
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)

        bpy.context.scene.cycles.bake_type = "EMIT"
        bpy.ops.object.bake(type="EMIT")

        baked_rough_path = out_path.parent / f"{out_path.stem}_roughness.png"
        baked_rough_img.filepath_raw = str(baked_rough_path)
        baked_rough_img.file_format = "PNG"
        baked_rough_img.save()
        bpy.data.materials.remove(rough_mat)

    # 清理輔助層，僅保留主 UVMap
    for lay_name in ["UV_Front", "UV_Back", "UV_Side"]:
        if lay_name in mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers[lay_name])
    for attr_name in ["HarmonicColor", "ViewWeights", "LoopRoughness"]:
        if attr_name in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes[attr_name])

    # 建立符合 Godot 4 標準的 PBR 材質
    pbr_mat = bpy.data.materials.new(f"M_{out_path.stem}_{style_cfg.id}_PBR")
    pbr_mat.use_nodes = True
    pbr_nodes = pbr_mat.node_tree.nodes
    pbr_links = pbr_mat.node_tree.links
    pbr_nodes.clear()

    bsdf = pbr_nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = float(style_cfg.metallic)
    if baked_rough_img is not None:
        tex_rough = pbr_nodes.new("ShaderNodeTexImage")
        tex_rough.image = baked_rough_img
        tex_rough.image.colorspace_settings.name = "Non-Color"
        uv_r = pbr_nodes.new("ShaderNodeUVMap")
        uv_r.uv_map = orig_uv_name
        pbr_links.new(uv_r.outputs["UV"], tex_rough.inputs["Vector"])
        pbr_links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = float(effective_roughness)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(style_cfg.specular_ior_level)
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = float(style_cfg.specular_ior_level)

    # 針對毛皮風格啟用 Sheen (絨毛邊緣背光散射)
    if is_fur:
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = 0.15
        if "Sheen Roughness" in bsdf.inputs:
            bsdf.inputs["Sheen Roughness"].default_value = 0.50

    # 針對陶瓷/琺瑯樹脂風格套用清透面漆層 (Clearcoat / Coat)
    if style_cfg.coat_weight > 0.0:
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = float(style_cfg.coat_weight)
            if "Coat Roughness" in bsdf.inputs:
                bsdf.inputs["Coat Roughness"].default_value = float(style_cfg.coat_roughness)
        elif "Clearcoat" in bsdf.inputs:
            bsdf.inputs["Clearcoat"].default_value = float(style_cfg.coat_weight)
            if "Clearcoat Roughness" in bsdf.inputs:
                bsdf.inputs["Clearcoat Roughness"].default_value = float(style_cfg.coat_roughness)

    final_tex_node = pbr_nodes.new("ShaderNodeTexImage")
    final_tex_node.image = baked_img
    pbr_links.new(final_tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    # 連接切線空間法線貼圖
    if baked_norm_img is not None:
        norm_map_node = pbr_nodes.new("ShaderNodeNormalMap")
        norm_map_node.space = "TANGENT"
        norm_map_node.inputs["Strength"].default_value = 1.0

        norm_tex_node = pbr_nodes.new("ShaderNodeTexImage")
        norm_tex_node.image = baked_norm_img
        norm_tex_node.image.colorspace_settings.name = "Non-Color"

        norm_uv = pbr_nodes.new("ShaderNodeUVMap")
        norm_uv.uv_map = orig_uv_name
        pbr_links.new(norm_uv.outputs["UV"], norm_tex_node.inputs["Vector"])
        pbr_links.new(norm_tex_node.outputs["Color"], norm_map_node.inputs["Color"])
        pbr_links.new(norm_map_node.outputs["Normal"], bsdf.inputs["Normal"])

    pbr_out = pbr_nodes.new("ShaderNodeOutputMaterial")
    pbr_links.new(bsdf.outputs["BSDF"], pbr_out.inputs["Surface"])

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(pbr_mat)

    # 自動微弧角膜 (Cornea Dome: 沿自適應眼球光學軸微推球面微弧)
    if args.cornea_dome and eye_info is not None:
        c_l = eye_info["center_l"]
        c_r = eye_info["center_r"]
        r_d = eye_info["R_dome"]
        ax_l = eye_info["axis_l"]
        ax_r = eye_info["axis_r"]
        h_d = 0.075 * r_d

        d_l = np.linalg.norm(coords - c_l, axis=1)
        d_r = np.linalg.norm(coords - c_r, axis=1)
        disp_l = np.where(d_l < r_d, h_d * np.cos(math.pi * d_l / (2.0 * r_d)) ** 2, 0.0)[:, None] * ax_l[None, :]
        disp_r = np.where(d_r < r_d, h_d * np.cos(math.pi * d_r / (2.0 * r_d)) ** 2, 0.0)[:, None] * ax_r[None, :]

        new_coords = coords + disp_l + disp_r
        for i, v in enumerate(mesh.vertices):
            v.co = new_coords[i]
        mesh.update()
        print(f"👁️  已成功套用自適應微弧角膜 (Cornea Dome, R={r_d:.4f}m, h={h_d*1000:.1f}mm)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_yup=True,
    )
    print(f"🎉 成功導出 Godot 4 PBR 模型至: {out_path} (大小: {out_path.stat().st_size / (1024*1024):.2f} MB)")

    if args.update_godot:
        godot_candidates = [
            shutil.which("godot"),
            "/Applications/Godot_mono.app/Contents/MacOS/Godot",
            "/Applications/Godot.app/Contents/MacOS/Godot",
        ]
        godot_bin = next((p for p in godot_candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)
        godot_dir = None
        for p in [out_path.parent, *out_path.parents]:
            if (p / "project.godot").is_file():
                godot_dir = p
                break

        if godot_bin and godot_dir:
            subprocess.run(
                [godot_bin, "--headless", "--path", str(godot_dir), "--import", "--quit"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅ Godot 4 重新匯入完成！")

    if args.verify:
        verify_script = Path(__file__).resolve().parent / "verify_against_reference.py"
        report_path = Path(args.diff_report) if args.diff_report else out_path.parent / f"{out_path.stem}_diff_report.png"
        print(f"\n🔍 自動啟動閉環差分質檢 (Closed-Loop Reference Diff QA)...", flush=True)
        try:
            # 調用 verify_against_reference 模組進行驗證
            blender_candidates = [
                shutil.which("blender"),
                "/opt/homebrew/bin/blender",
                "/Applications/Blender.app/Contents/MacOS/Blender",
                "/usr/local/bin/blender",
            ]
            blender_bin = next((p for p in blender_candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)
            verify_cmd = [
                blender_bin if blender_bin else sys.executable,
                "-b", "-noaudio", "--python", str(verify_script),
                "--",
                "--glb", str(out_path),
                "--front", str(front_path),
                "--back", str(back_path),
                "--side", str(side_path),
                "-o", str(report_path),
                "--res", "1024",
            ] if blender_bin else [
                sys.executable,
                str(verify_script),
                "--glb", str(out_path),
                "--front", str(front_path),
                "--back", str(back_path),
                "--side", str(side_path),
                "-o", str(report_path),
                "--res", "1024",
            ]
            subprocess.run(verify_cmd, check=False)
        except Exception as exc:
            print(f"⚠️ 質檢執行發生非致命異常: {exc}")


if __name__ == "__main__":
    main()
