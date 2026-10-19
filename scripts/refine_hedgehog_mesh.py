#!/usr/bin/env python3
"""
Hedgehog Mesh Proportions & Geometry Refiner (`scripts/refine_hedgehog_mesh.py`)

針對 Hunyuan3D-2 幾何模型進行體態比例調和與細節雕刻：
1. 水滴/湯圓體態重構 (Pear-Shape Deform)：
   將 Z < 0.6 的下半身頂點依照平滑餘弦曲線漸進加寬 14%，賦予原畫坐姿扎實圓滾的飽滿下盤。
2. 兩側粉紅果凍腳掌 (Jellybean Feet) 前伸與微抬：
   定位腳部頂點 (Z < -0.6, Y < -0.2, |X| in [0.2, 0.5])，沿前方 (-Y) 前推 0.11 單位並微抬 +Z 0.045 單位，
   確保正面平視時清晰顯露可愛前伸的小腳掌。
3. 兩側小圓耳 (Ears Detachment) 微調：
   將雙耳向外推移與微傾，加強頭部剪影的辨識度與杯狀立體感。
4. 全面法線重整與平滑化 (Recalculate Smooth Normals)。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


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
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="刺蝟 3D 幾何體態比例雕刻器")
    parser.add_argument("--input", default="godot_project/assets/models/hedgehog_v2.orig.glb", help="輸入原始 GLB 模型")
    parser.add_argument("--output", default="godot_project/assets/models/hedgehog_v2.refined.glb", help="輸出精修 GLB 模型")
    parser.add_argument("--pear-scale", type=float, default=0.14, help="下盤水滴形加寬強度 (預設: 0.14)")
    parser.add_argument("--feet-forward", type=float, default=0.11, help="腳掌前伸距離 (預設: 0.11)")
    parser.add_argument("--ear-boost", type=float, default=0.035, help="耳朵外推加強幅度 (預設: 0.035)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

    print("============================================================")
    print("✨ 啟動刺蝟 3D 幾何體態比例雕刻 (Geometry Refinement)")
    print(f"   來源網格: {in_path.name}")
    print(f"   目標輸出: {out_path.name}")
    print(f"   參數配置: 水滴加寬={args.pear_scale}, 腳掌前伸={args.feet_forward}, 耳朵外推={args.ear_boost}")
    print("============================================================")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(in_path))
    mesh_obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
    mesh = mesh_obj.data

    coords = np.array([v.co for v in mesh.vertices])
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    z_min, z_max = z.min(), z.max()
    t = (z - z_min) / max(1e-5, (z_max - z_min))

    # 1. 水滴/湯圓體態漸進加寬 (Pear-shape Deform)
    # 在 t < 0.65 範圍內使用平滑餘弦曲線對 X 與 Y 進行水平放大
    scale = np.where(t < 0.65, 1.0 + args.pear_scale * np.cos(np.pi * t / 1.3), 1.0)
    new_x = x * scale
    new_y = y * (1.0 + (scale - 1.0) * 0.75)  # Y 軸略微膨脹以保持腹背自然圓滾
    new_z = z.copy()

    # 2. 雙腳掌 (Jellybean Feet) 前伸與微抬
    c_left = np.array([-0.33, -0.42, -0.76])
    c_right = np.array([0.33, -0.42, -0.76])
    d_l = np.linalg.norm(coords - c_left, axis=1)
    d_r = np.linalg.norm(coords - c_right, axis=1)
    d_feet = np.minimum(d_l, d_r)
    w_feet = np.exp(-(d_feet**2) / (2.0 * 0.17**2))

    new_y -= float(args.feet_forward) * w_feet  # 前方為 -Y
    new_z += 0.045 * w_feet  # 腳尖微向上翹起，露出圓潤肉墊

    # 3. 雙耳輪廓獨立性加強
    c_ear_l = np.array([-0.52, -0.22, 0.30])
    c_ear_r = np.array([0.52, -0.22, 0.30])
    d_ear_l = np.linalg.norm(coords - c_ear_l, axis=1)
    d_ear_r = np.linalg.norm(coords - c_ear_r, axis=1)

    w_ear_l = np.exp(-(d_ear_l**2) / (2.0 * 0.14**2))
    w_ear_r = np.exp(-(d_ear_r**2) / (2.0 * 0.14**2))

    new_x -= float(args.ear_boost) * w_ear_l  # 左耳向外推
    new_x += float(args.ear_boost) * w_ear_r  # 右耳向外推
    new_y -= 0.02 * (w_ear_l + w_ear_r)      # 略微前傾朝向鏡頭

    # 4. 套用變形回網格頂點
    for i, v in enumerate(mesh.vertices):
        v.co = (new_x[i], new_y[i], new_z[i])

    # 5. 強制頂點法線全平滑
    for poly in mesh.polygons:
        poly.use_smooth = True

    # 導出精修模型
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_yup=True,
    )
    print(f"🎉 成功導出精修幾何模型: {out_path} (大小: {out_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
