#!/usr/bin/env python3
"""
3D 模型多視角檢視圖與 360° 旋轉展示影片渲染腳本 (Blender Headless CYCLES CPU)
用法:
  python3 scripts/render_turntable.py --glb godot_project/assets/models/hedgehog_new.glb --out-dir /path/to/out --prefix hedgehog_new [--turntable]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile


BLENDER_SCRIPT = r"""
import bpy
import sys
import os
import math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb_path = argv[0]
out_dir = argv[1]
prefix = argv[2]
do_turntable = argv[3] == "1"

os.makedirs(out_dir, exist_ok=True)

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = False

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.14, 0.14, 0.16, 1.0)
        bg.inputs[1].default_value = 1.0

    # Three-point studio lighting
    lights = [
        ("Key", (2.5, -2.5, 3.0), 4.0),
        ("Fill", (-2.5, -1.5, 1.5), 2.0),
        ("Rim", (0.0, 2.5, 2.5), 3.0),
        ("BackFill", (1.5, 2.5, 1.0), 2.0),
    ]
    for name, loc, energy in lights:
        ld = bpy.data.lights.new(name=name, type='SUN')
        ld.energy = energy
        ld.angle = math.radians(15.0)
        lo = bpy.data.objects.new(name=name, object_data=ld)
        scene.collection.objects.link(lo)
        lo.location = loc
        direction = Vector((0, 0, 0)) - Vector(loc)
        lo.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return scene

def load_and_normalize(glb_file):
    bpy.ops.import_scene.gltf(filepath=glb_file)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        raise RuntimeError(f"No mesh in {glb_file}")

    min_v = Vector((1e9, 1e9, 1e9))
    max_v = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        for v in obj.bound_box:
            wv = obj.matrix_world @ Vector(v)
            for i in range(3):
                min_v[i] = min(min_v[i], wv[i])
                max_v[i] = max(max_v[i], wv[i])
    center = (min_v + max_v) * 0.5
    size = max(max_v - min_v)
    scale = 1.65 / size if size > 0 else 1.0

    root = bpy.data.objects.new("ModelRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in bpy.context.scene.objects:
        if obj != root and obj.parent is None and obj.type in ('MESH', 'EMPTY', 'ARMATURE'):
            obj.parent = root
    root.scale = (scale, scale, scale)
    root.location = -center * scale

    total_v = sum(len(m.data.vertices) for m in meshes)
    total_f = sum(len(m.data.polygons) for m in meshes)
    print(f"MODEL_STATS: prefix={prefix} vertices={total_v} faces={total_f}")
    return meshes, root

def apply_clay(meshes):
    mat = bpy.data.materials.new(name="ClayMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.74, 0.78, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.42
    for m in meshes:
        m.data.materials.clear()
        m.data.materials.append(mat)

def render_4views(scene, out_prefix):
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 2.0
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    views = [
        ("front", (0.0, -3.0, 0.25)),
        ("quarter", (2.1, -2.1, 0.4)),
        ("side", (3.0, 0.0, 0.25)),
        ("back", (0.0, 3.0, 0.25)),
    ]
    for vname, loc in views:
        cam_obj.location = loc
        direction = Vector((0, 0, 0)) - Vector(loc)
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = os.path.join(out_dir, f"{out_prefix}_{vname}.png")
        bpy.ops.render.render(write_still=True)

scene = setup_scene()
meshes, root = load_and_normalize(glb_path)
render_4views(scene, f"{prefix}_tex")

if do_turntable:
    t_dir = os.path.join(out_dir, f"{prefix}_turntable_frames")
    os.makedirs(t_dir, exist_ok=True)
    cam_obj = scene.camera
    cam_obj.location = (0.0, -3.0, 0.35)
    direction = Vector((0, 0, 0)) - Vector(cam_obj.location)
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    num_frames = 36
    scene.cycles.samples = 24
    for i in range(num_frames):
        angle = (2.0 * math.pi * i) / num_frames
        root.rotation_euler = (0.0, 0.0, angle)
        scene.render.filepath = os.path.join(t_dir, f"frame_{i:03d}.png")
        bpy.ops.render.render(write_still=True)
    root.rotation_euler = (0.0, 0.0, 0.0)
    scene.cycles.samples = 32

apply_clay(meshes)
render_4views(scene, f"{prefix}_clay")
"""


def stitch_strip(out_dir: str, out_prefix: str, strip_path: str):
    views = ["front", "quarter", "side", "back"]
    inputs = [os.path.join(out_dir, f"{out_prefix}_{v}.png") for v in views]
    try:
        from PIL import Image, ImageDraw
        labels = ["Front", "3/4 View", "Right Side", "Back"]
        imgs = [Image.open(p) for p in inputs]
        w, h = imgs[0].size
        strip = Image.new("RGB", (w * 4, h), (35, 35, 40))
        draw = ImageDraw.Draw(strip)
        for i, (im, lbl) in enumerate(zip(imgs, labels)):
            strip.paste(im, (i * w, 0))
            draw.text((i * w + 16, 16), f"{out_prefix} - {lbl}", fill=(240, 240, 240))
        strip.save(strip_path)
    except ImportError:
        cmd = [
            "ffmpeg", "-y",
            "-i", inputs[0], "-i", inputs[1], "-i", inputs[2], "-i", inputs[3],
            "-filter_complex", "hstack=inputs=4",
            strip_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ 已合成四視角檢視圖: {strip_path}")


def main():
    parser = argparse.ArgumentParser(description="渲染 3D 模型四視角與 360 度旋轉展示")
    parser.add_argument("--glb", required=True, help="輸入 .glb 檔案路徑")
    parser.add_argument("--out-dir", required=True, help="輸出目錄")
    parser.add_argument("--prefix", required=True, help="輸出檔名前綴")
    parser.add_argument("--turntable", action="store_true", help="是否同時渲染 360 度旋轉 MP4")
    args = parser.parse_args()

    blender_bin = "/Applications/Blender.app/Contents/MacOS/Blender"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(BLENDER_SCRIPT)
        script_path = tf.name

    try:
        cmd = [
            blender_bin, "-b", "-P", script_path, "--",
            os.path.abspath(args.glb),
            os.path.abspath(args.out_dir),
            args.prefix,
            "1" if args.turntable else "0",
        ]
        subprocess.run(cmd, check=True)
        stitch_strip(
            args.out_dir,
            f"{args.prefix}_tex",
            os.path.join(args.out_dir, f"render_{args.prefix}_textured.png"),
        )
        stitch_strip(
            args.out_dir,
            f"{args.prefix}_clay",
            os.path.join(args.out_dir, f"render_{args.prefix}_clay.png"),
        )
        if args.turntable:
            t_dir = os.path.join(args.out_dir, f"{args.prefix}_turntable_frames")
            mp4_path = os.path.join(args.out_dir, f"turntable_{args.prefix}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "12",
                "-i", os.path.join(t_dir, "frame_%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                mp4_path,
            ], check=True)
            print(f"✅ 已合成 360° 旋轉影片: {mp4_path}")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


if __name__ == "__main__":
    main()
