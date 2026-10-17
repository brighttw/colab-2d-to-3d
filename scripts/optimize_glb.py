#!/usr/bin/env python3
"""
3D GLB Post-Processing & Godot 4 Scene Optimizer (`scripts/optimize_glb.py`)

針對 Hunyuan3D-2 MV + xatlas 產出的 .glb 模型進行自動化幾何、UV 與貼圖修復：
1. 孤島 UV 縫合與重心插值修補 (Orphan UV Island Stitching & Barycentric Inpainting)：
   修復 xatlas 展開時將臉部/眼下/嘴角零星三角面拋至背刺區導致的深色破塊。
2. 跨 UV 接縫預分割細分與空間頂點焊接 (Seam-Safe Subdivision & Spatial Vertex Welding)：
   解決臉部與肚子分配面數過少（僅佔全模型 <3%）造成的塊狀低模感，並消除 xatlas 造成的 6,000+ 個裂開法線接縫。
3. 體積保留 Taubin 平滑與嘴部凹陷修復 (Taubin Surface Fairing & Mouth Trench Relaxation)：
   撫平正側面視角衝突造成的嘴部深坑與下巴多邊形稜角，同時柔化背部尖刺碎片感。
4. 正側面口鼻深度比例調和 (Proportional Snout Compaction)：
   針對側面圖口鼻過度突出（長吻）進行平滑高斯回縮，貼合正面圖圓潤可愛的比例。
5. 內嵌貼圖去暗部、奶油絨毛提亮、3D 投影腮紅與眼神光還原 (Texture De-Lighting, Blush & Catchlights)。
6. 自動配置 Godot 4 柔光攝影棚三點燈光與絨毛次表面材質 (`--update-godot-scene`)。

用法:
  python3 scripts/optimize_glb.py godot_project/assets/models/hedgehog.glb --update-godot-scene
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_blender_runtime() -> None:
    """若當前執行的不是 Blender 內建 Python (無 bpy 模組)，自動轉發至 headless Blender 執行。"""
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
        print("❌ 錯誤: 找不到 Blender 執行檔 (需要 bpy / bmesh 進行 3D 網格與 UV 拓撲修復)。", file=sys.stderr)
        sys.exit(1)

    script_path = str(Path(__file__).resolve())
    cmd = [blender_bin, "-b", "-noaudio", "--python", script_path, "--", *sys.argv[1:]]
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


_ensure_blender_runtime()

# 以下在 Blender Python (bpy + bmesh + numpy) 環境內執行
import math  # noqa: E402
import bpy  # noqa: E402
import bmesh  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="優化 Hunyuan3D-2 生成之 GLB 模型與 Godot 4 場景")
    parser.add_argument("input_glb", help="輸入 .glb 模型路徑 (例如 godot_project/assets/models/hedgehog.glb)")
    parser.add_argument("-o", "--output", default=None, help="輸出 .glb 路徑 (預設覆寫輸入檔，並自動建立 .orig.glb 備份)")
    parser.add_argument("--subdivide-cuts", type=int, default=1, help="正面/臉部平滑區域細分階數 (預設: 1)")
    parser.add_argument("--smooth-steps", type=int, default=12, help="Taubin 體積保留平滑迭代次數 (預設: 12)")
    parser.add_argument("--snout-compact", type=float, default=0.28, help="口鼻突出回縮比例 0.0~0.6 (預設: 0.28)")
    parser.add_argument("--spike-relax", type=float, default=0.16, help="背部尖刺碎片柔化強度 0.0~0.5 (預設: 0.16)")
    parser.add_argument("--delight-strength", type=float, default=0.34, help="貼圖暗部陰影提亮強度 0.0~0.8 (預設: 0.34)")
    parser.add_argument("--blush-strength", type=float, default=0.30, help="雙頰粉紅腮紅還原強度 0.0~0.6 (預設: 0.30)")
    parser.add_argument("--eye-highlight", action="store_true", default=True, help="自動增強眼睛深邃度與白色眼神高光 (預設開啟)")
    parser.add_argument("--no-eye-highlight", dest="eye_highlight", action="store_false", help="停用眼神高光增強")
    parser.add_argument("--update-godot-scene", action="store_true", help="同步更新 godot_project/main.tscn 柔光攝影棚與絨毛材質")
    parser.add_argument("--render-preview", default=None, help="選用：渲染一張優化後的預覽 PNG 圖片至指定路徑")
    return parser.parse_args(argv)


def stitch_orphan_uv_islands_and_inpaint(obj: bpy.types.Object, rgb: np.ndarray, w: int, h: int) -> int:
    """
    偵測正面/臉部被 xatlas 切碎並拋至遠處的孤島三角面 (<= 6 faces)，
    將其 Loop UV 縫合回相鄰的主臉部/身體 UV 島，並以重心座標插值修補貼圖孔洞。
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    uv_lay = bm.loops.layers.uv.active

    # 計算模型邊界以正規化座標 (Blender 空間: X 左右, -Y 正面, Z 上下)
    coords = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float32)
    b_min = coords.min(axis=0)
    b_max = coords.max(axis=0)
    b_center = 0.5 * (b_min + b_max)
    b_half = np.maximum(0.5 * (b_max - b_min), 1e-5)

    # 建立 UV 連通分量 (UV Islands)
    visited = set()
    islands = []
    for f in bm.faces:
        if f in visited:
            continue
        comp = []
        stack = [f]
        visited.add(f)
        while stack:
            curr = stack.pop()
            comp.append(curr)
            for e in curr.edges:
                for nf in e.link_faces:
                    if nf not in visited:
                        same_uv = True
                        for v in e.verts:
                            uv0 = [l[uv_lay].uv for l in curr.loops if l.vert == v][0]
                            uv1 = [l[uv_lay].uv for l in nf.loops if l.vert == v][0]
                            if (uv0 - uv1).length > 1e-3:
                                same_uv = False
                                break
                        if same_uv:
                            visited.add(nf)
                            stack.append(nf)
        islands.append(comp)

    face_to_island_size = {}
    face_to_island_id = {}
    for idx, comp in enumerate(islands):
        for f in comp:
            face_to_island_size[f] = len(comp)
            face_to_island_id[f] = idx

    stitched_tris_uv = []
    stitched_count = 0
    for comp in islands:
        if len(comp) <= 6:
            fc = sum((f.calc_center_median() for f in comp), Vector((0, 0, 0))) / len(comp)
            norm_y = (fc.y - b_center[1]) / b_half[1]
            if norm_y < -0.10:
                cand_islands = {}
                all_nbr_islands = set()
                for f in comp:
                    for v in f.verts:
                        for l in v.link_loops:
                            if l.face not in comp:
                                iid = face_to_island_id[l.face]
                                all_nbr_islands.add(iid)
                                if face_to_island_size[l.face] > 15:
                                    cand_islands[iid] = cand_islands.get(iid, 0) + 1
                if len(all_nbr_islands) == 1 and len(cand_islands) == 1:
                    best_iid = next(iter(cand_islands))
                    can_stitch = True
                    vert_target_uv = {}
                    for f in comp:
                        for v in f.verts:
                            matching_uvs = [
                                l[uv_lay].uv.copy()
                                for l in v.link_loops
                                if face_to_island_id[l.face] == best_iid
                            ]
                            if matching_uvs:
                                vert_target_uv[v] = matching_uvs[0]
                            else:
                                can_stitch = False
                                break
                    if can_stitch:
                        for f in comp:
                            tri_uv = []
                            for l in f.loops:
                                l[uv_lay].uv = vert_target_uv[l.vert]
                                tri_uv.append((l[uv_lay].uv.x, l[uv_lay].uv.y))
                            if len(tri_uv) == 3:
                                stitched_tris_uv.append(tri_uv)
                        stitched_count += len(comp)

    # 重心座標三角孔洞修補 (Barycentric Triangle Inpainting)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    for (u0, v0), (u1, v1), (u2, v2) in stitched_tris_uv:
        pad_out = 0.006
        ox0 = max(0, int((min(u0, u1, u2) - pad_out) * w))
        ox1 = min(w, int((max(u0, u1, u2) + pad_out) * w) + 1)
        oy0 = max(0, int((min(v0, v1, v2) - pad_out) * h))
        oy1 = min(h, int((max(v0, v1, v2) + pad_out) * h) + 1)
        outer = rgb[oy0:oy1, ox0:ox1]
        o_luma = 0.299 * outer[:, :, 0] + 0.587 * outer[:, :, 1] + 0.114 * outer[:, :, 2]
        bright_pixels = outer[o_luma > np.percentile(o_luma, 60)]
        if len(bright_pixels) == 0:
            continue
        fill_col = np.median(bright_pixels, axis=0)

        margin = 1.5 / w
        bx0 = max(0, int((min(u0, u1, u2) - margin) * w))
        bx1 = min(w, int((max(u0, u1, u2) + margin) * w) + 1)
        by0 = max(0, int((min(v0, v1, v2) - margin) * h))
        by1 = min(h, int((max(v0, v1, v2) + margin) * h) + 1)
        px_u = xx[by0:by1, bx0:bx1]
        px_v = yy[by0:by1, bx0:bx1]
        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(denom) < 1e-8:
            continue
        b0 = ((v1 - v2) * (px_u - u2) + (u2 - u1) * (px_v - v2)) / denom
        b1 = ((v2 - v0) * (px_u - u2) + (u0 - u2) * (px_v - v2)) / denom
        b2 = 1.0 - b0 - b1
        min_b = np.minimum(np.minimum(b0, b1), b2)
        tri_mask = np.clip((min_b + 0.08) / 0.13, 0.0, 1.0)

        inner = rgb[by0:by1, bx0:bx1]
        i_luma = 0.299 * inner[:, :, 0] + 0.587 * inner[:, :, 1] + 0.114 * inner[:, :, 2]
        f_luma_val = 0.299 * fill_col[0] + 0.587 * fill_col[1] + 0.114 * fill_col[2]
        w_fill = np.clip((f_luma_val - i_luma) / 0.12, 0.0, 1.0) * tri_mask
        for c_i in range(3):
            inner[:, :, c_i] = inner[:, :, c_i] * (1.0 - w_fill) + fill_col[c_i] * w_fill
        rgb[by0:by1, bx0:bx1] = inner

    bm.to_mesh(obj.data)
    bm.free()
    return stitched_count


def optimize_geometry_and_normals(
    obj: bpy.types.Object,
    subdivide_cuts: int = 1,
    smooth_steps: int = 12,
    snout_compact: float = 0.28,
    spike_relax: float = 0.16,
) -> tuple[int, int]:
    """執行 UV 接縫安全細分、空間頂點焊接、Taubin 體積保留平滑、口鼻比例回縮與平滑法線重建。"""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    uv_lay = bm.loops.layers.uv.active

    coords = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float32)
    b_min = coords.min(axis=0)
    b_max = coords.max(axis=0)
    b_center = 0.5 * (b_min + b_max)
    b_half = np.maximum(0.5 * (b_max - b_min), 1e-5)

    # 1. 先沿 UV 接縫切開邊，確保細分 (subdivide_edges) 絕對不會跨 UV 島插值座標
    seam_edges = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            f0, f1 = e.link_faces[0], e.link_faces[1]
            for v in e.verts:
                uv0 = [l[uv_lay].uv for l in f0.loops if l.vert == v][0]
                uv1 = [l[uv_lay].uv for l in f1.loops if l.vert == v][0]
                if (uv0 - uv1).length > 1e-3:
                    seam_edges.append(e)
                    break
    if seam_edges:
        bmesh.ops.split_edges(bm, edges=seam_edges)

    # 2. 對模型正面區域 (臉部、五官、手臂、肚子) 進行細分以提升曲面圓潤度
    if subdivide_cuts > 0:
        front_edges = [
            e
            for e in bm.edges
            if (
                (e.verts[0].co.y - b_center[1]) / b_half[1] < -0.05
                or (e.verts[1].co.y - b_center[1]) / b_half[1] < -0.05
            )
        ]
        if front_edges:
            bmesh.ops.subdivide_edges(
                bm,
                edges=front_edges,
                cuts=subdivide_cuts,
                smooth=0.0,
                use_grid_fill=True,
                use_single_edge=True,
            )
            bmesh.ops.triangulate(bm, faces=bm.faces)

    # 3. 細分完成後，將所有 3D 空間重疊頂點焊接起來，徹底消除 UV 接縫處的法線斷裂
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # 4. Taubin 體積保留平滑 + 嘴部深溝修復
    for step in range(smooth_steps):
        deltas = {}
        taubin_factor = 0.45 if (step % 2 == 0) else -0.22
        for v in bm.verts:
            nx = (v.co.x - b_center[0]) / b_half[0]
            ny = (v.co.y - b_center[1]) / b_half[1]
            nz = (v.co.z - b_center[2]) / b_half[2]

            in_mouth = abs(nx) < 0.36 and ny < -0.45 and -0.12 < nz < 0.22
            in_front = ny < -0.10 and abs(nx) < 0.78 and -0.92 < nz < 0.52
            if in_mouth or in_front:
                nbrs = [e.other_vert(v).co for e in v.link_edges]
                if nbrs:
                    avg = sum(nbrs, Vector((0, 0, 0))) / len(nbrs)
                    w_v = (0.52 if step < (smooth_steps // 2 + 1) else taubin_factor) if in_mouth else taubin_factor
                    deltas[v] = (avg - v.co) * w_v
        for v, d in deltas.items():
            v.co += d

    # 5. 口鼻部比例回縮 (修正正側面視角長吻差異)
    if snout_compact > 0.0:
        y_thresh = b_center[1] - 0.50 * b_half[1]
        for v in bm.verts:
            nx = (v.co.x - b_center[0]) / b_half[0]
            ny = (v.co.y - b_center[1]) / b_half[1]
            nz = (v.co.z - b_center[2]) / b_half[2]
            if ny < -0.50 and abs(nx) < 0.52 and -0.16 < nz < 0.46:
                excess_y = y_thresh - v.co.y
                wx = max(0.0, 1.0 - (abs(nx) / 0.52) ** 2)
                wz = max(0.0, 1.0 - ((nz - 0.16) / 0.32) ** 2)
                w_snout = wx * wz
                v.co.y += excess_y * snout_compact * w_snout
                v.co.z += excess_y * (snout_compact * 0.22) * w_snout

    # 6. 背部尖刺適度柔化 (減少碎石感)
    if spike_relax > 0.0:
        for _ in range(2):
            spike_deltas = {}
            for v in bm.verts:
                ny = (v.co.y - b_center[1]) / b_half[1]
                if ny >= -0.10:
                    nbrs = [e.other_vert(v).co for e in v.link_edges]
                    if nbrs:
                        avg = sum(nbrs, Vector((0, 0, 0))) / len(nbrs)
                        spike_deltas[v] = (avg - v.co) * spike_relax
            for v, d in spike_deltas.items():
                v.co += d

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    n_verts = len(bm.verts)
    n_faces = len(bm.faces)
    bm.to_mesh(obj.data)
    bm.free()

    bpy.ops.object.shade_smooth()
    return n_verts, n_faces


def enhance_embedded_texture(
    obj: bpy.types.Object,
    rgb: np.ndarray,
    w: int,
    h: int,
    delight_strength: float = 0.34,
    blush_strength: float = 0.30,
    eye_highlight: bool = True,
) -> np.ndarray:
    """對烘焙貼圖進行暗部去陰影、奶油色溫提亮、3D 臉頰腮紅投影與眼神高光增強。"""
    me = obj.data
    uv_data = me.uv_layers.active.data

    coords = np.array([[v.co.x, v.co.y, v.co.z] for v in me.vertices], dtype=np.float32)
    b_min = coords.min(axis=0)
    b_max = coords.max(axis=0)
    b_center = 0.5 * (b_min + b_max)
    b_half = np.maximum(0.5 * (b_max - b_min), 1e-5)

    cheek_uvs = []
    eye_cand_uvs = []
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            nx = (co.x - b_center[0]) / b_half[0]
            ny = (co.y - b_center[1]) / b_half[1]
            nz = (co.z - b_center[2]) / b_half[2]
            uv = uv_data[li].uv
            if 0.20 <= abs(nx) <= 0.38 and ny < -0.45 and 0.04 <= nz <= 0.20:
                cheek_uvs.append((uv.x, uv.y))
            if 0.12 <= abs(nx) <= 0.32 and ny < -0.55 and 0.24 <= nz <= 0.44:
                eye_cand_uvs.append((nx, uv.x, uv.y))

    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    cream_target = np.array([0.96, 0.89, 0.81], dtype=np.float32)

    # 1. 柔和提亮皮膚/絨毛區域的暗部陰影並校正為溫暖奶油米色（全域連續色調映射，確保跨 UV 接縫零色差）
    if delight_strength > 0.0:
        w_luma = np.clip((luma - 0.34) / 0.20, 0.0, 1.0) * np.clip((0.88 - luma) / 0.20, 0.0, 1.0)
        w_blue = np.clip((rgb[:, :, 2] - 0.24) / 0.15, 0.0, 1.0)
        w_skin_soft = w_luma * w_blue
        for c_i in range(3):
            rgb[:, :, c_i] = (
                rgb[:, :, c_i] * (1.0 - delight_strength * w_skin_soft)
                + cream_target[c_i] * (delight_strength * w_skin_soft)
            )

    # 2. 3D 雙頰座標投影粉紅腮紅
    if blush_strength > 0.0 and cheek_uvs:
        cheek_arr = np.unique(np.round(np.array(cheek_uvs), 3), axis=0)
        blush_field = np.zeros((h, w), dtype=np.float32)
        step = max(1, len(cheek_arr) // 32)
        for pu, pv in cheek_arr[::step]:
            d2 = (xx - pu) ** 2 + (yy - pv) ** 2
            blush_field = np.maximum(blush_field, np.exp(-d2 / (2 * 0.011**2)))
        blush_color = np.array([0.96, 0.62, 0.62], dtype=np.float32)
        bw = blush_field * (blush_strength * 0.55) * np.clip((luma - 0.42) / 0.18, 0.0, 1.0)
        for c_i in range(3):
            rgb[:, :, c_i] = rgb[:, :, c_i] * (1.0 - bw) + blush_color[c_i] * bw

    # 3. 自動定位雙眼瞳孔中心，加深瞳孔對比並點上高光
    if eye_highlight and eye_cand_uvs:
        eye_arr = np.array(eye_cand_uvs)
        for side_sign in (-1.0, 1.0):
            side_pts = eye_arr[eye_arr[:, 0] * side_sign > 0]
            if len(side_pts) == 0:
                continue
            eu, ev = np.median(side_pts[:, 1:], axis=0)
            eye_box = (np.abs(xx - eu) < 0.028) & (np.abs(yy - ev) < 0.028)
            if eye_box.any():
                sub_luma = np.where(eye_box, luma, 1.0)
                cy, cx = np.unravel_index(np.argmin(sub_luma), luma.shape)
                cu, cv = xx[cy, cx], yy[cy, cx]
                d_eye = np.sqrt((xx - cu) ** 2 + (yy - cv) ** 2)
                iris_w = np.clip((0.012 - d_eye) / 0.004, 0.0, 1.0) * np.clip((0.36 - luma) / 0.12, 0.0, 1.0)
                for c_i in range(3):
                    rgb[:, :, c_i] *= 1.0 - 0.45 * iris_w
                hl_d = np.sqrt((xx - (cu - 0.0022)) ** 2 + (yy - (cv + 0.0025)) ** 2)
                hl_w = np.exp(-(hl_d**2) / (2 * 0.0014**2)) * 0.88
                for c_i in range(3):
                    rgb[:, :, c_i] = rgb[:, :, c_i] * (1.0 - hl_w) + hl_w

    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, 0.88)
    return rgb


def update_godot_project_scene(glb_path: Path) -> None:
    """更新 Godot 4 專案的 main.tscn，配置柔光攝影棚三點光源與絨毛感環境光，並觸發 Headless 重新匯入。"""
    # 尋找 godot_project 根目錄
    godot_dir = None
    for parent in [glb_path.parent, *glb_path.parents]:
        if (parent / "project.godot").is_file():
            godot_dir = parent
            break

    if not godot_dir:
        print("⚠️ 找不到 project.godot，跳過 Godot 場景更新。")
        return

    rel_glb = glb_path.resolve().relative_to(godot_dir.resolve()).as_posix()
    main_tscn = godot_dir / "main.tscn"

    tscn_content = f"""[gd_scene format=3 uid="uid://b8xq7y3m6rv1p"]

[ext_resource type="PackedScene" path="res://{rel_glb}" id="1_hedgehog"]

[sub_resource type="ProceduralSkyMaterial" id="ProceduralSkyMaterial_preview"]
sky_top_color = Color(0.78, 0.82, 0.88, 1)
sky_horizon_color = Color(0.88, 0.85, 0.82, 1)
ground_bottom_color = Color(0.58, 0.54, 0.50, 1)
ground_horizon_color = Color(0.84, 0.80, 0.77, 1)

[sub_resource type="Sky" id="Sky_preview"]
sky_material = SubResource("ProceduralSkyMaterial_preview")

[sub_resource type="Environment" id="Environment_preview"]
background_mode = 2
sky = SubResource("Sky_preview")
ambient_light_source = 3
ambient_light_color = Color(0.98, 0.93, 0.88, 1)
ambient_light_sky_contribution = 0.65
ambient_light_energy = 0.85
tonemap_mode = 2
tonemap_exposure = 1.05
ssao_enabled = true
ssao_radius = 0.45
ssao_intensity = 0.6
glow_enabled = true
glow_intensity = 0.25
glow_bloom = 0.05

[node name="Main" type="Node3D"]

[node name="WorldEnvironment" type="WorldEnvironment" parent="."]
environment = SubResource("Environment_preview")

[node name="KeyLight3D" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.819152, -0.32899, 0.469846, 0, 0.819152, 0.573576, -0.573576, -0.469846, 0.67101, 0, 3, 2)
light_color = Color(1, 0.98, 0.95, 1)
light_energy = 0.85
shadow_enabled = true
shadow_bias = 0.03
shadow_normal_bias = 1.5
shadow_opacity = 0.55
shadow_blur = 3.0

[node name="FillLight3D" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.766044, 0.219846, -0.604023, 0, 0.939693, 0.34202, 0.642788, -0.262003, 0.719846, 0, 1, 2)
light_color = Color(1, 0.92, 0.86, 1)
light_energy = 0.45
shadow_enabled = false

[node name="RimLight3D" type="DirectionalLight3D" parent="."]
transform = Transform3D(-0.939693, -0.116978, -0.321394, 0, 0.939693, -0.34202, 0.34202, -0.321394, -0.883022, 0, 2.5, -2)
light_color = Color(1, 0.96, 0.9, 1)
light_energy = 0.35
shadow_enabled = false

[node name="Camera3D" type="Camera3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 0.965926, 0.258819, 0, -0.258819, 0.965926, 0, 0.25, 2.6)
fov = 45.0

[node name="ModelRoot" type="Node3D" parent="."]

[node name="Hedgehog" parent="ModelRoot" instance=ExtResource("1_hedgehog")]
"""
    main_tscn.write_text(tscn_content, encoding="utf-8")
    print(f"✅ 已更新 Godot 柔光攝影棚場景: {main_tscn}")

    # 自動觸發 Godot Headless 重新匯入
    godot_candidates = [
        shutil.which("godot"),
        "/Applications/Godot_mono.app/Contents/MacOS/Godot",
        "/Applications/Godot.app/Contents/MacOS/Godot",
    ]
    godot_bin = next((p for p in godot_candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)
    if godot_bin:
        print("🔄 正在執行 Godot 4 Headless 重新匯入資產...")
        subprocess.run(
            [godot_bin, "--headless", "--path", str(godot_dir), "--import", "--quit"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ Godot 4 資產匯入快取更新完成！")


def render_preview_image(obj: bpy.types.Object, preview_path: str) -> None:
    """使用 Cycles CPU 渲染優化後的 3/4 視角預覽圖。"""
    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_data.lens = 55
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = Vector((-1.45, -2.55, 0.35))
    direction = Vector((0.0, 0.0, -0.05)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    world = bpy.data.worlds.new("PreviewWorld")
    bpy.context.scene.world = world
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.85, 0.82, 0.80, 1.0)
    bg.inputs[1].default_value = 0.75

    l1_data = bpy.data.lights.new("Key", type="SUN")
    l1_data.energy = 2.2
    l1_data.angle = math.radians(25.0)
    l1 = bpy.data.objects.new("Key", l1_data)
    bpy.context.collection.objects.link(l1)
    l1.rotation_euler = (math.radians(50), math.radians(15), math.radians(-35))

    l2_data = bpy.data.lights.new("Fill", type="SUN")
    l2_data.energy = 1.3
    l2_data.color = (1.0, 0.95, 0.90)
    l2_data.angle = math.radians(45.0)
    l2 = bpy.data.objects.new("Fill", l2_data)
    bpy.context.collection.objects.link(l2)
    l2.rotation_euler = (math.radians(105), math.radians(-10), math.radians(25))

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.render.resolution_x = 720
    scene.render.resolution_y = 800
    scene.render.filepath = str(Path(preview_path).resolve())
    bpy.ops.render.render(write_still=True)
    print(f"🖼️  已輸出優化後 3D 預覽渲染圖: {preview_path}")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_glb).resolve()
    if not in_path.is_file():
        print(f"❌ 找不到輸入 GLB 檔案: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).resolve() if args.output else in_path
    backup_path = in_path.with_suffix(".orig.glb")

    # 若直接覆寫原檔且尚未有 .orig.glb 備份，先自動建立備份，並總是從乾淨原始檔讀取以避免重複疊加變形
    if out_path == in_path:
        if not backup_path.is_file():
            shutil.copy2(in_path, backup_path)
            print(f"📦 已自動備份原始未優化模型至: {backup_path.name}")
        source_glb = backup_path
    else:
        source_glb = backup_path if backup_path.is_file() else in_path

    print("============================================================")
    print("✨ 啟動 3D GLB 模型幾何、UV 孤島與貼圖自動化優化")
    print(f"   來源模型: {source_glb}")
    print(f"   目標輸出: {out_path}")
    print("============================================================")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source_glb))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB 檔案中未包含任何 MESH 物件！")

    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # 清除破脆的原始 split normals
    bpy.ops.mesh.customdata_custom_splitnormals_clear()

    mat = obj.data.materials[0] if obj.data.materials else None
    img = None
    if mat and mat.node_tree:
        tex_nodes = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
        if tex_nodes:
            img = tex_nodes[0].image
        bsdf_nodes = [n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]
        if bsdf_nodes:
            # 調高 Roughness 消除塑膠油亮感，呈現絨毛漫反射質感
            bsdf_nodes[0].inputs["Roughness"].default_value = 0.92

    if img:
        w, h = img.size
        px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        rgb = px[:, :, :3].copy()

        print("🔧 [1/3] 正在縫合臉部/身體孤島 UV 並以重心座標修補貼圖破孔...")
        stitched = stitch_orphan_uv_islands_and_inpaint(obj, rgb, w, h)
        print(f"   已成功縫合 {stitched} 個孤立破面回主 UV 島。")
    else:
        rgb = None
        w = h = 0

    print("🔧 [2/3] 正在執行跨 UV 接縫安全細分、頂點焊接、Taubin 曲面平滑與口鼻比例校正...")
    n_verts, n_faces = optimize_geometry_and_normals(
        obj,
        subdivide_cuts=args.subdivide_cuts,
        smooth_steps=args.smooth_steps,
        snout_compact=args.snout_compact,
        spike_relax=args.spike_relax,
    )
    print(f"   優化後拓撲: 焊接頂點數 = {n_verts}, 三角面數 = {n_faces}")

    if img is not None and rgb is not None:
        print("🔧 [3/3] 正在進行貼圖去陰影提亮、雙頰腮紅 3D 投影與眼神高光修復...")
        rgb = enhance_embedded_texture(
            obj,
            rgb,
            w,
            h,
            delight_strength=args.delight_strength,
            blush_strength=args.blush_strength,
            eye_highlight=args.eye_highlight,
        )
        px[:, :, :3] = rgb
        img.pixels[:] = px.flatten().tolist()
        img.update()

        # 若同目錄有展開的 <stem>_0.png 貼圖檔，一併同步寫出
        ext_png = out_path.parent / f"{out_path.stem}_0.png"
        img.filepath_raw = str(ext_png)
        img.file_format = "PNG"
        img.save()
        img.pack()
        print(f"   已同步更新外部貼圖: {ext_png.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        export_normals=True,
        export_tangents=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✅ 已導出優化後的 GLB 模型: {out_path} ({size_mb:.2f} MB)")

    if args.render_preview:
        render_preview_image(obj, args.render_preview)

    if args.update_godot_scene:
        update_godot_project_scene(out_path)


if __name__ == "__main__":
    main()
