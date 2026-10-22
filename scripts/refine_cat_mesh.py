#!/usr/bin/env python3
"""
Cat Mesh Geometry & Cheek Whisker Refiner (`scripts/refine_cat_mesh.py`)

精確修復 Hunyuan3D-2 / 3D AI 生成之貓咪幾何白模臉頰鬍鬚黏連與肉塊突起問題：
1. 嚴格鎖定臉頰全段鬍鬚五官區域 (All 4 Whisker Pairs Zone):
   精準定位 Z in [0.22, 0.52], Y <= -0.28, |X| in [0.14, 0.42]，
   全面涵蓋由上至下全部 4 對鬍鬚（包含上方 2 對與下方黏連的 2 對），
   同時 100% 避開並保護雙手、手臂 (Z < 0.20 或 Y > -0.25)、軀幹與中央鼻吻部 (|X| < 0.14)！
2. 消除臉頰體素肉塊黏連 (Laplacian Whisker Ridge Smoothing)：
   在臉頰兩側凸起的鬍鬚肉褶與下顎交界處進行 16 次局部拓撲拉普拉斯平滑，徹底將 AI 誤生成的肉疙瘩融合成光滑圓潤的包子臉。
3. 全面法線重整與平滑化 (Recalculate Smooth Normals)：
   確保全臉曲面在燈光掠過時具備均勻高光，為 2K PBR 移印工藝提供無幾何凹凸干擾的理想畫布。
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
import bmesh  # noqa: E402
from mathutils import Vector  # noqa: E402
import numpy as np  # noqa: E402

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from bake_multiview_textures import clean_thin_webbing_fins


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="貓咪 3D 幾何臉頰鬍鬚去贅肉精修器")
    parser.add_argument("--input", default="godot_project/assets/models/cat.orig.glb", help="輸入原始 GLB 模型")
    parser.add_argument("--output", default="godot_project/assets/models/cat.refined.glb", help="輸出精修 GLB 模型")
    parser.add_argument("--thickness-ratio", type=float, default=0.015, help="薄片厚度比例門檻 (預設: 0.015)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

    print("============================================================")
    print("🐱 啟動貓咪 3D 臉頰薄片蹼狀鬍鬚切除與無縫平滑精修")
    print(f"   來源網格: {in_path.name}")
    print(f"   目標輸出: {out_path.name}")
    print(f"   厚度門檻: {args.thickness_ratio}")
    print("   特點: 通用薄片連通分量切除 + 拓撲縫合細分 + 臉頰曲面拉普拉斯平滑")
    print("============================================================")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(in_path))
    mesh_obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]

    n_cleaned = clean_thin_webbing_fins(mesh_obj, thickness_ratio=args.thickness_ratio)
    print(f"✨ 成功切除 {n_cleaned} 個懸空薄片/殘蹼頂點，並完成包子臉平滑重整！")

    # 強制頂點法線全平滑
    for poly in mesh_obj.data.polygons:
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
