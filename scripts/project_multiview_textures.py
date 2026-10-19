#!/usr/bin/env python3
"""
Multi-View Texture Projection & Godot Scene Optimizer (`scripts/project_multiview_textures.py`)

直接將乾淨的正、背、側三視圖投影烘焙至 Hunyuan3D 產出的 3D 網格 UV 貼圖上：
1. 解決 HunyuanPaint 單視角腦補導致背面粉紅/紫色刺全部漂白遺失的根本缺陷。
2. 根據 3D 頂點法向權重進行 Front (-Y), Back (+Y), Side (-X, +X) 多角度平滑混合。
3. 採用 UV 接縫膨脹保護 (Dilation)，消除接縫黑邊。
4. 校正 Godot 4 展示場景主光與環境光過曝。
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ensure_runtime() -> None:
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
        print("❌ 錯誤: 找不到 Blender 執行檔。", file=sys.stderr)
        sys.exit(1)

    script_path = str(Path(__file__).resolve())
    cmd = [blender_bin, "-b", "-noaudio", "--python", script_path, "--", *sys.argv[1:]]
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


_ensure_runtime()

import bpy  # noqa: E402
import numpy as np  # noqa: E402


def load_img(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[::-1, :, :3].copy()  # top-down RGB


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0 = xs.astype(np.int32)
    y0 = ys.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[:, None]
    fy = (ys - y0)[:, None]
    top = img[y0, x0] * (1.0 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1.0 - fx) + img[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


def project_textures(
    glb_path: Path,
    front_path: Path,
    side_path: Path,
    back_path: Path,
    out_glb_path: Path,
    tex_res: int = 2048,
) -> None:
    print(f"📖 讀取 3D 模型: {glb_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("GLB 檔案未包含任何網格！")

    obj = mesh_objs[0]
    me = obj.data
    uv_layer = me.uv_layers.active.data

    print("🖼️  載入三視圖素材...")
    front_img = load_img(front_path)
    side_img = load_img(side_path)
    back_img = load_img(back_path)

    # 取得網格邊界
    coords = np.array([v.co for v in me.vertices], dtype=np.float32)
    normals = np.array([v.normal for v in me.vertices], dtype=np.float32)

    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_span = max(1e-4, z_max - z_min)
    y_center = 0.5 * (y_min + y_max)

    # 影像映射常數 (對齊 preprocess_views 的畫布規格: 680x1000, 840h, 90 y_top)
    c_w, c_h = front_img.shape[1], front_img.shape[0]
    target_char_h = int(c_h * 0.84)
    target_y_top = int(c_h * 0.09)
    scale_factor = target_char_h / z_span
    x_center = c_w / 2.0

    print(f"📐 幾何映射參數: Z:[{z_min:.2f}, {z_max:.2f}], scale={scale_factor:.1f}, center_x={x_center:.1f}")

    # 建立 2048x2048 緩衝區
    out_rgb = np.zeros((tex_res, tex_res, 3), dtype=np.float32)
    hit_mask = np.zeros((tex_res, tex_res), dtype=bool)

    # 三角形點投影光柵化到 UV 空間
    print("🎨 正在執行多視角 3D 法向權重投影烘焙...")
    poly_count = len(me.polygons)

    for p_idx, p in enumerate(me.polygons):
        if p_idx % 5000 == 0:
            print(f"   進度: {p_idx}/{poly_count} ({p_idx*100//poly_count}%)")

        v_indices = p.vertices
        l_indices = p.loop_indices

        if len(v_indices) < 3:
            continue

        p0 = coords[v_indices[0]]
        p1 = coords[v_indices[1]]
        p2 = coords[v_indices[2]]

        n0 = normals[v_indices[0]]
        n1 = normals[v_indices[1]]
        n2 = normals[v_indices[2]]

        uv0 = np.array(uv_layer[l_indices[0]].uv)
        uv1 = np.array(uv_layer[l_indices[1]].uv)
        uv2 = np.array(uv_layer[l_indices[2]].uv)

        # UV 三角形包圍盒 (tex_res 空間)
        u_pts = np.array([uv0[0], uv1[0], uv2[0]]) * (tex_res - 1)
        v_pts = np.array([uv0[1], uv1[1], uv2[1]]) * (tex_res - 1)

        min_u, max_u = max(0, int(np.floor(u_pts.min()))), min(tex_res - 1, int(np.ceil(u_pts.max())))
        min_v, max_v = max(0, int(np.floor(v_pts.min()))), min(tex_res - 1, int(np.ceil(v_pts.max())))

        if min_u > max_u or min_v > max_v:
            continue

        grid_u, grid_v = np.meshgrid(
            np.arange(min_u, max_u + 1, dtype=np.float32),
            np.arange(min_v, max_v + 1, dtype=np.float32),
        )
        flat_u = grid_u.flatten()
        flat_v = grid_v.flatten()

        # 重心座標計算
        det = (uv1[1] - uv2[1]) * (uv0[0] - uv2[0]) + (uv2[0] - uv1[0]) * (uv0[1] - uv2[1])
        if abs(det) < 1e-9:
            continue

        u_norm = flat_u / (tex_res - 1)
        v_norm = flat_v / (tex_res - 1)

        w0 = ((uv1[1] - uv2[1]) * (u_norm - uv2[0]) + (uv2[0] - uv1[0]) * (v_norm - uv2[1])) / det
        w1 = ((uv2[1] - uv0[1]) * (u_norm - uv2[0]) + (uv0[0] - uv2[0]) * (v_norm - uv2[1])) / det
        w2 = 1.0 - w0 - w1

        # 稍微外擴 (pad 0.05) 避免三角面交界黑線
        inside = (w0 >= -0.05) & (w1 >= -0.05) & (w2 >= -0.05)
        if not np.any(inside):
            continue

        w0_in = w0[inside, None]
        w1_in = w1[inside, None]
        w2_in = w2[inside, None]
        iu_in = flat_u[inside].astype(np.int32)
        iv_in = flat_v[inside].astype(np.int32)

        # 插值 3D 空間座標與表面法線
        p_interp = w0_in * p0 + w1_in * p1 + w2_in * p2
        n_interp = w0_in * n0 + w1_in * n1 + w2_in * n2
        n_len = np.linalg.norm(n_interp, axis=1, keepdims=True)
        n_interp = np.where(n_len > 1e-5, n_interp / n_len, np.array([0, -1, 0], dtype=np.float32))

        # 投影至各視圖影像座標
        z_vals = p_interp[:, 2]
        img_y = target_y_top + (z_max - z_vals) * scale_factor

        # 1. Front View (-Y)
        front_x = x_center + p_interp[:, 0] * scale_factor
        col_front = sample_bilinear(front_img, front_x, img_y)
        w_front = np.clip(-n_interp[:, 1:2], 0.0, 1.0) ** 1.8

        # 2. Back View (+Y)
        back_x = x_center - p_interp[:, 0] * scale_factor
        col_back = sample_bilinear(back_img, back_x, img_y)
        w_back = np.clip(n_interp[:, 1:2], 0.0, 1.0) ** 1.8

        # 3. Side View (-X: left)
        side_l_x = x_center + (p_interp[:, 1] - y_center) * scale_factor
        col_side_l = sample_bilinear(side_img, side_l_x, img_y)
        w_side_l = np.clip(-n_interp[:, 0:1], 0.0, 1.0) ** 1.8

        # 4. Side View (+X: right, mirrored)
        side_r_x = x_center + (p_interp[:, 1] - y_center) * scale_factor
        col_side_r = sample_bilinear(side_img, side_r_x, img_y)
        w_side_r = np.clip(n_interp[:, 0:1], 0.0, 1.0) ** 1.8

        # 若取樣到純白背景 (>0.97)，降低該視圖權重
        def bg_penalty(col: np.ndarray) -> np.ndarray:
            is_bg = (col[:, 0] > 0.96) & (col[:, 1] > 0.96) & (col[:, 2] > 0.96)
            return np.where(is_bg[:, None], 0.05, 1.0)

        w_front *= bg_penalty(col_front)
        w_back *= bg_penalty(col_back)
        w_side_l *= bg_penalty(col_side_l)
        w_side_r *= bg_penalty(col_side_r)

        w_total = w_front + w_back + w_side_l + w_side_r + 1e-6
        blended = (
            col_front * w_front
            + col_back * w_back
            + col_side_l * w_side_l
            + col_side_r * w_side_r
        ) / w_total

        out_rgb[iv_in, iu_in] = blended
        hit_mask[iv_in, iu_in] = True

    print("✨ 多視角投影計算完成，正在執行 UV 接縫邊緣修補與色彩強化...")

    # 針對少數未覆蓋的 UV 接縫執行擴散修補 (Inpainting / Dilation)
    for _ in range(12):
        unhit = ~hit_mask
        if not np.any(unhit):
            break
        padded_rgb = np.pad(out_rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_mask = np.pad(hit_mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)

        neighbor_sum = np.zeros_like(out_rgb)
        neighbor_cnt = np.zeros((tex_res, tex_res, 1), dtype=np.float32)

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shift_rgb = padded_rgb[1 + dy : 1 + dy + tex_res, 1 + dx : 1 + dx + tex_res]
            shift_mask = padded_mask[1 + dy : 1 + dy + tex_res, 1 + dx : 1 + dx + tex_res, None]
            neighbor_sum += shift_rgb * shift_mask
            neighbor_cnt += shift_mask

        can_fill = unhit[:, :, None] & (neighbor_cnt > 0)
        out_rgb = np.where(can_fill, neighbor_sum / np.maximum(neighbor_cnt, 1e-5), out_rgb)
        hit_mask = hit_mask | (neighbor_cnt[:, :, 0] > 0)

    # 將所有完全未命中的背景區域填入柔和粉白底色，徹底避免 mipmap 接縫處滲出黑邊
    unhit = ~hit_mask
    out_rgb[unhit] = np.array([0.95, 0.82, 0.86], dtype=np.float32)

    # 適度調整對比與飽和度
    out_rgb = np.clip(out_rgb, 0.0, 1.0)
    # 微調提升粉紅與紫色背刺色彩飽和度
    luma = 0.299 * out_rgb[:, :, 0] + 0.587 * out_rgb[:, :, 1] + 0.114 * out_rgb[:, :, 2]
    is_spike = (out_rgb[:, :, 0] > 0.70) & (out_rgb[:, :, 2] > 0.70) & (out_rgb[:, :, 1] < 0.85)
    for c_i in (0, 2):  # boost Red & Blue (Pink / Purple)
        out_rgb[:, :, c_i] = np.where(
            is_spike,
            np.clip(out_rgb[:, :, c_i] * 1.12, 0.0, 1.0),
            out_rgb[:, :, c_i],
        )

    # 寫入 Blender 影像並更新材質
    mat = obj.data.materials[0] if obj.data.materials else None
    tex_node = next((n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"), None) if mat else None
    if tex_node and tex_node.image:
        target_img = tex_node.image
    else:
        target_img = bpy.data.images.new("hedgehog_pbr_tex", width=tex_res, height=tex_res, alpha=False)
        if tex_node:
            tex_node.image = target_img

    rgba = np.ones((tex_res, tex_res, 4), dtype=np.float32)
    rgba[:, :, :3] = out_rgb
    target_img.pixels[:] = rgba.flatten().tolist()
    target_img.update()

    # 匯出外部 PNG 貼圖
    ext_png = out_glb_path.parent / f"{out_glb_path.stem}_0.png"
    target_img.filepath_raw = str(ext_png.resolve())
    target_img.file_format = "PNG"
    target_img.save()
    target_img.pack()
    print(f"💾 已儲存高色彩飽和 PBR 貼圖: {ext_png}")

    # 導出 GLB
    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb_path),
        export_format="GLB",
        export_normals=True,
        export_tangents=True,
        export_materials="EXPORT",
    )
    print(f"🎉 已導出多視角色彩烘焙 3D 模型: {out_glb_path}")


def update_godot_lighting(godot_dir: Path) -> None:
    main_tscn = godot_dir / "main.tscn"
    if not main_tscn.is_file():
        return

    content = main_tscn.read_text(encoding="utf-8")

    # 調整光照強度，避免白模過曝
    content = content.replace("ambient_light_energy = 0.85", "ambient_light_energy = 0.35")
    content = content.replace("ambient_light_sky_contribution = 0.65", "ambient_light_sky_contribution = 0.40")
    content = content.replace("tonemap_exposure = 1.05", "tonemap_exposure = 0.95")
    content = content.replace("light_energy = 0.85", "light_energy = 0.50")
    content = content.replace("light_energy = 0.45", "light_energy = 0.28")
    content = content.replace("light_energy = 0.35", "light_energy = 0.22")

    main_tscn.write_text(content, encoding="utf-8")
    print("💡 已校正 Godot 4 柔光攝影棚燈光強度 (防止高亮過曝)！")

    # 觸發 Godot 重新導入
    godot_candidates = [
        shutil.which("godot"),
        "/Applications/Godot_mono.app/Contents/MacOS/Godot",
        "/Applications/Godot.app/Contents/MacOS/Godot",
    ]
    godot_bin = next((p for p in godot_candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)
    if godot_bin:
        print("🔄 正在觸發 Godot 4 Headless 重新載入資產快取...")
        subprocess.run(
            [godot_bin, "--headless", "--path", str(godot_dir), "--import", "--quit"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ Godot 4 快取更新完畢！")


def main() -> None:
    parser = argparse.ArgumentParser(description="多視角真實色彩 3D 貼圖投影烘焙工具")
    parser.add_argument("--glb", default="godot_project/assets/models/hedgehog_v2.glb")
    parser.add_argument("--front", default="assets/inputs/hedgehog_v2_front.png")
    parser.add_argument("--side", default="assets/inputs/hedgehog_v2_side.png")
    parser.add_argument("--back", default="assets/inputs/hedgehog_v2_back.png")
    parser.add_argument("--output", default="godot_project/assets/models/hedgehog_v2.glb")
    parser.add_argument("--res", type=int, default=2048)

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])

    glb_p = Path(args.glb).resolve()
    front_p = Path(args.front).resolve()
    side_p = Path(args.side).resolve()
    back_p = Path(args.back).resolve()
    out_p = Path(args.output).resolve()

    project_textures(glb_p, front_p, side_p, back_p, out_p, tex_res=args.res)
    update_godot_lighting(Path("godot_project").resolve())


if __name__ == "__main__":
    main()
