#!/usr/bin/env python3
"""
專門生成 3D 模型 360° 旋轉展示 MP4 影片工具 (scripts/generate_turntable_mp4.py)

功能特色:
1. 支援任意 .glb / .gltf 3D 模型自動置中與視角邊界自適應。
2. 預設提供媲美 Tripo Studio 的「暗色攝影棚無縫底板 + 地面反射 (studio_dark)」預設風格。
3. 支援多種攝影棚風格: studio_dark, studio_white, dark, white, clay 等。
4. 自由自訂輸出影片解析度 (預設 720x720，可自訂 1080x1080, 1920x1080 等)。
5. 自由調整影片時長 (預設 4.0 秒)、影格率 (預設 24 fps) 與相機俯角 (預設 12°)。
6. 自動調用 Blender (Cycles CPU) 渲染與 ffmpeg 編碼為全平台相容的 H.264 MP4 (yuv420p)。

範例:
  # 預設 720x720, 24fps, 4s 攝影棚暗色反射展示影片
  python3 scripts/generate_turntable_mp4.py --glb path/to/model.glb

  # 指定輸出路徑與 1080x1080 高解析度
  python3 scripts/generate_turntable_mp4.py --glb path/to/model.glb -o showcase.mp4 --res 1080x1080

  # 白模雕刻 (Clay) 模式
  python3 scripts/generate_turntable_mp4.py --glb path/to/model.glb --mode clay -o clay_turntable.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


BLENDER_RENDER_SCRIPT = r"""
import bpy
import bmesh
import math
import os
import sys
import json
import time
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
config = json.loads(argv[0])

glb_path = config["glb_path"]
frames_dir = config["frames_dir"]
total_frames = config["total_frames"]
res_x = config["res_x"]
res_y = config["res_y"]
samples = config["samples"]
elevation_deg = config["elevation_deg"]
camera_lens = config["camera_lens"]
camera_zoom = config["camera_zoom"]
clockwise = config["clockwise"]
preset = config["preset"]
mode = config["mode"]

os.makedirs(frames_dir, exist_ok=True)

# 1. 初始化乾淨場景
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = samples
scene.cycles.use_denoising = False
scene.render.resolution_x = res_x
scene.render.resolution_y = res_y
scene.render.film_transparent = (preset == "transparent")

# 2. 設定世界背景與顏色
world = bpy.data.worlds.new("TurntableWorld")
scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")

if preset in ("studio_dark", "dark"):
    bg_color = (0.07, 0.07, 0.08, 1.0)
elif preset in ("studio_white", "white"):
    bg_color = (0.94, 0.94, 0.95, 1.0)
else:
    bg_color = (0.12, 0.12, 0.13, 1.0)

if bg_node:
    bg_node.inputs[0].default_value = bg_color
    bg_node.inputs[1].default_value = 1.0

# 3. 建立無縫攝影棚 Cyclorama 弧形底板 (若是 studio 風格)
cyc_obj = None
if preset in ("studio_dark", "studio_white"):
    bm = bmesh.new()
    # 建立地面與後方平滑延展的立體無縫底幕
    v1 = bm.verts.new((-8.0, -8.0, 0.0))
    v2 = bm.verts.new(( 8.0, -8.0, 0.0))
    v3 = bm.verts.new(( 8.0,  3.5, 0.0))
    v4 = bm.verts.new((-8.0,  3.5, 0.0))
    v5 = bm.verts.new(( 8.0,  7.0, 6.0))
    v6 = bm.verts.new((-8.0,  7.0, 6.0))
    bm.faces.new((v1, v2, v3, v4))
    bm.faces.new((v4, v3, v5, v6))
    mesh = bpy.data.meshes.new("CycMesh")
    bm.to_mesh(mesh)
    bm.free()

    cyc_obj = bpy.data.objects.new("Cyclorama", mesh)
    scene.collection.objects.link(cyc_obj)
    sub = cyc_obj.modifiers.new("Subsurf", 'SUBSURF')
    sub.levels = 3
    sub.render_levels = 3

    cmat = bpy.data.materials.new("CycMat")
    cmat.use_nodes = True
    bsdf = cmat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if preset == "studio_dark":
            bsdf.inputs["Base Color"].default_value = (0.07, 0.07, 0.08, 1.0)
            # 柔和地面倒影 (Roughness 0.26 營造微霧鏡面)
            bsdf.inputs["Roughness"].default_value = 0.26
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.5
        else: # studio_white
            bsdf.inputs["Base Color"].default_value = (0.94, 0.94, 0.95, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.45
    cyc_obj.data.materials.append(cmat)
    for p in mesh.polygons:
        p.use_smooth = True

elif preset in ("dark", "white"):
    # 平面底板 (純投影)
    bpy.ops.mesh.primitive_cylinder_add(radius=15.0, depth=0.01, location=(0, 0, -0.005))
    ground = bpy.context.active_object
    cyc_obj = ground
    gmat = bpy.data.materials.new("FlatGroundMat")
    gmat.use_nodes = True
    bsdf = gmat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = bg_color
        bsdf.inputs["Roughness"].default_value = 0.5
    ground.data.materials.append(gmat)

# 4. 匯入 GLB 模型
bpy.ops.import_scene.gltf(filepath=glb_path)
imported_meshes = [o for o in scene.objects if o.type == 'MESH' and o != cyc_obj]
if not imported_meshes:
    raise RuntimeError(f"在 {glb_path} 中找不到任何 MESH 網格！")

# 5. 計算模型包圍盒與自適應標準化
min_v = Vector((1e9, 1e9, 1e9))
max_v = Vector((-1e9, -1e9, -1e9))
for obj in imported_meshes:
    for v in obj.bound_box:
        wv = obj.matrix_world @ Vector(v)
        for i in range(3):
            min_v[i] = min(min_v[i], wv[i])
            max_v[i] = max(max_v[i], wv[i])

center = (min_v + max_v) * 0.5
dim = max_v - min_v
max_dim = max(dim)
target_height = 1.6
scale = (target_height / max_dim) if max_dim > 0 else 1.0

root = bpy.data.objects.new("TurntableRoot", None)
scene.collection.objects.link(root)
for obj in list(scene.objects):
    if obj != root and obj != cyc_obj and obj.parent is None and obj.type in ('MESH', 'EMPTY', 'ARMATURE'):
        obj.parent = root

root.scale = (scale, scale, scale)
# 確保腳底緊貼 Z=0，水平置中於 (0, 0)
root.location = Vector((-center.x * scale, -center.y * scale, -min_v.z * scale))

# 6. 若為白模 (Clay) 模式，統一上啞光黏土材質
if mode == "clay":
    clay_mat = bpy.data.materials.new("ClayStudioMat")
    clay_mat.use_nodes = True
    bsdf = clay_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.76, 0.77, 0.80, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.40
    for m in imported_meshes:
        m.data.materials.clear()
        m.data.materials.append(clay_mat)

# 7. 專業攝影棚 4 點柔光照明系統 (Key + Fill + Rim + BackFill)
light_multiplier = 1.0 if preset in ("studio_dark", "dark") else 0.85
studio_lights = [
    ("KeyLight", (3.2, -3.2, 3.8), 4.2 * light_multiplier, 12.0),
    ("FillLight", (-3.2, -1.8, 2.2), 2.2 * light_multiplier, 15.0),
    ("RimLight", (0.0, 3.2, 3.2), 3.2 * light_multiplier, 10.0),
    ("BackFillLight", (2.2, 3.2, 1.8), 1.8 * light_multiplier, 15.0),
]
for name, loc, energy, angle_deg in studio_lights:
    ld = bpy.data.lights.new(name=name, type='SUN')
    ld.energy = energy
    ld.angle = math.radians(angle_deg)
    lo = bpy.data.objects.new(name=name, object_data=ld)
    scene.collection.objects.link(lo)
    lo.location = loc
    aim_target = Vector((0.0, 0.0, 0.8))
    direction = aim_target - Vector(loc)
    lo.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# 8. 設定相機與俯視角 (Elevation)
cam_data = bpy.data.cameras.new("TurntableCam")
cam_data.lens = camera_lens
cam_obj = bpy.data.objects.new("TurntableCam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# 計算相機距離與高度
target_z = 0.80
cam_dist = 3.25 * camera_zoom
elev_rad = math.radians(elevation_deg)
cam_y = -cam_dist * math.cos(elev_rad)
cam_z = target_z + cam_dist * math.sin(elev_rad)
cam_obj.location = (0.0, cam_y, cam_z)

cam_dir = Vector((0.0, 0.0, target_z)) - Vector(cam_obj.location)
cam_obj.rotation_euler = cam_dir.to_track_quat('-Z', 'Y').to_euler()

# 9. 循序渲染 360° 旋轉影格
print(f"READY_TO_RENDER: total_frames={total_frames} res={res_x}x{res_y} samples={samples}", flush=True)

direction_sign = -1.0 if clockwise else 1.0

for i in range(total_frames):
    angle = direction_sign * (2.0 * math.pi * i) / total_frames
    root.rotation_euler = (0.0, 0.0, angle)
    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
    scene.render.filepath = frame_path
    
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    
    pct = ((i + 1) / total_frames) * 100.0
    print(f"FRAME_DONE: index={i+1}/{total_frames} pct={pct:.1f}% time={dt:.2f}s file={frame_path}", flush=True)

print("ALL_FRAMES_RENDERED", flush=True)
"""


def find_executable(name: str, fallback_paths: list[str]) -> str:
    path = shutil.which(name)
    if path:
        return path
    for p in fallback_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return str(p)
    raise FileNotFoundError(f"找不到必要工具: {name}。請確認環境已正確安裝。")


def parse_resolution(res_str: str) -> tuple[int, int]:
    res_str = res_str.lower().strip()
    if "x" in res_str:
        parts = res_str.split("x")
        return int(parts[0]), int(parts[1])
    val = int(res_str)
    return val, val


def main():
    parser = argparse.ArgumentParser(
        description="專門生成 3D 模型 360 度旋轉展示 MP4 影片工具 (Blender Cycles + ffmpeg)"
    )
    parser.add_argument("--glb", required=True, help="輸入的 .glb 或 .gltf 3D 模型檔案路徑")
    parser.add_argument("-o", "--output", help="輸出的 .mp4 影片檔案路徑 (若未指定則自動根據模型名稱產生)")
    parser.add_argument("--out-dir", default=None, help="輸出目錄 (預設與模型同目錄或 ref/new)")
    parser.add_argument("--duration", type=float, default=4.0, help="旋轉一圈的總秒數 (預設: 4.0 秒)")
    parser.add_argument("--fps", type=int, default=24, help="影片幀率 FPS (預設: 24，極致流暢)")
    parser.add_argument("--res", "--resolution", default="720x720", help="影片解析度，例如 720x720, 1080x1080, 1920x1080 (預設: 720x720)")
    parser.add_argument("--elevation", type=float, default=12.0, help="相機俯角角度 (度)，0 為水平平視，預設 12.0° 俯視展示")
    parser.add_argument("--preset", choices=["studio_dark", "studio_white", "dark", "white", "transparent"], default="studio_dark", help="攝影棚預設風格 (預設: studio_dark 暗色地面反射棚)")
    parser.add_argument("--mode", choices=["textured", "clay"], default="textured", help="材質模式: textured (原始貼圖) 或 clay (白模雕刻) (預設: textured)")
    parser.add_argument("--clockwise", action="store_true", default=False, help="順時針旋轉 (預設為逆時針)")
    parser.add_argument("--samples", type=int, default=20, help="Cycles 渲染採樣數 (預設: 20，兼顧極速與柔和畫質)")
    parser.add_argument("--zoom", type=float, default=1.0, help="相機距離縮放係數 (預設: 1.0)")
    parser.add_argument("--keep-frames", action="store_true", default=False, help="保留渲染過程的逐格 PNG 影格目錄")
    parser.add_argument("--blender-bin", default=None, help="手動指定 Blender 執行檔路徑")

    args = parser.parse_args()

    # 1. 驗證輸入模型檔案
    glb_path = Path(args.glb).resolve()
    if not glb_path.is_file():
        print(f"❌ 錯誤: 找不到輸入模型檔案: {glb_path}", file=sys.stderr)
        sys.exit(1)

    # 2. 決定輸出 MP4 路徑
    if args.output:
        out_mp4 = Path(args.output).resolve()
    elif args.out_dir:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix_mode = f"_{args.mode}" if args.mode != "textured" else ""
        out_mp4 = out_dir / f"turntable_{glb_path.stem}{suffix_mode}.mp4"
    else:
        out_dir = glb_path.parent
        suffix_mode = f"_{args.mode}" if args.mode != "textured" else ""
        out_mp4 = out_dir / f"turntable_{glb_path.stem}{suffix_mode}.mp4"

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # 3. 尋找 Blender 與 ffmpeg
    blender_candidates = [
        args.blender_bin,
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/opt/homebrew/bin/blender",
        "/usr/local/bin/blender",
    ]
    blender_bin = find_executable(
        "blender", [p for p in blender_candidates if p]
    )

    ffmpeg_candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    ffmpeg_bin = find_executable(
        "ffmpeg", [p for p in ffmpeg_candidates if p]
    )

    res_x, res_y = parse_resolution(args.res)
    total_frames = max(1, int(round(args.duration * args.fps)))

    print("============================================================")
    print("🎬 啟動 3D 模型 360° 旋轉展示影片渲染管線")
    print(f"   模型檔案: {glb_path.name}")
    print(f"   目標影片: {out_mp4}")
    print(f"   規格設定: {res_x}x{res_y} | {args.fps} FPS | {args.duration:.1f} 秒 (共 {total_frames} 幀)")
    print(f"   風格預設: {args.preset} | 模式: {args.mode} | 採樣: {args.samples}")
    print("============================================================")

    # 4. 準備暫存影格目錄
    if args.keep_frames:
        frames_dir = out_mp4.parent / f"{out_mp4.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_obj = None
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="turntable_frames_")
        frames_dir = Path(temp_dir_obj.name)

    # 5. 構建並傳遞參數給 Blender
    blender_config = {
        "glb_path": str(glb_path),
        "frames_dir": str(frames_dir),
        "total_frames": total_frames,
        "res_x": res_x,
        "res_y": res_y,
        "samples": args.samples,
        "elevation_deg": args.elevation,
        "camera_lens": 52.0,
        "camera_zoom": args.zoom,
        "clockwise": args.clockwise,
        "preset": args.preset,
        "mode": args.mode,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as script_file:
        script_file.write(BLENDER_RENDER_SCRIPT)
        runner_script_path = script_file.name

    t_start = time.time()
    try:
        cmd = [
            blender_bin,
            "-b",
            "-noaudio",
            "-P",
            runner_script_path,
            "--",
            json.dumps(blender_config),
        ]

        print(f"🚀 [1/2] 正在調用 Blender Cycles 循序渲染 {total_frames} 幀影格...")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        output_lines = []
        for line in proc.stdout:
            line_str = line.strip()
            output_lines.append(line_str)
            if line_str.startswith("FRAME_DONE:"):
                # 解析進度資訊
                parts = line_str.split()
                idx_part = next((p for p in parts if p.startswith("index=")), "")
                pct_part = next((p for p in parts if p.startswith("pct=")), "")
                time_part = next((p for p in parts if p.startswith("time=")), "")
                print(f"   🎥 渲染進度: {idx_part.replace('index=', '')} [{pct_part.replace('pct=', '')}] ({time_part.replace('time=', '')})", flush=True)
            elif "READY_TO_RENDER" in line_str:
                print("   ⚡ 場景建立與模型置中完成，開始輸出影格...", flush=True)

        proc.wait()
        if proc.returncode != 0:
            tail = "\n".join(output_lines[-20:])
            raise RuntimeError(f"Blender 渲染失敗，結束代碼: {proc.returncode}\n{tail}")
        if not any(l.startswith("FRAME_DONE:") for l in output_lines):
            tail = "\n".join(output_lines[-25:])
            raise RuntimeError(f"Blender 未產出任何影格，執行紀錄:\n{tail}")

        render_sec = time.time() - t_start
        print(f"✅ [1/2] 影格渲染全數完成！耗時: {render_sec:.1f} 秒 (平均 {(render_sec/total_frames):.2f}s/幀)")

        # 6. 調用 ffmpeg 合成高品質 H.264 MP4
        print(f"\n📦 [2/2] 正在透過 ffmpeg 合成 H.264 MP4 展示影片...")
        frame_pattern = str(frames_dir / "frame_%04d.png")
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate", str(args.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_mp4),
        ]

        ff_proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if ff_proc.returncode != 0:
            print(f"❌ ffmpeg 轉碼失敗:\n{ff_proc.stderr}", file=sys.stderr)
            sys.exit(1)

        file_size_mb = out_mp4.stat().st_size / (1024 * 1024)
        total_time = time.time() - t_start
        print(f"🎉 360° 旋轉展示影片產生成功！")
        print(f"   輸出檔案: {out_mp4}")
        print(f"   檔案大小: {file_size_mb:.2f} MB")
        print(f"   總計耗時: {total_time:.1f} 秒")

    finally:
        if os.path.exists(runner_script_path):
            os.remove(runner_script_path)
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
