---
name: colab-2d-to-3d
description: >-
  Generate commercial blind-box quality Godot 4 3D assets (.glb, 40,000 faces, 2K PBR) from 2D multi-view reference sheets
  or orthographic images (front, side, back) using Hunyuan3D-2 MV high-res geometry inference on Google Colab (L4 GPU)
  combined with local Blender occlusion-aware harmonic PBR texture baking. Supports single-session warm-start batch generation.
  Use this skill whenever the user wants to generate multi-view 2D prompts, convert 2D concept art into 3D meshes,
  run the automated Colab inference pipeline, bake multi-view PBR textures, or import/verify 3D models for Godot 4.
---

# Colab 2D-to-3D Skill (Hybrid Cloud Geometry + Local Occlusion-Aware PBR Pipeline)

This skill provides an end-to-end automated workflow to turn 2D multi-view character sheets (front, side, back) into commercial blind-box quality 3D assets (`.glb`) with 40,000-face sculpted geometry, seamless 2K occlusion-aware PBR textures, and automated 4-view / 360° turntable verification for Godot 4.

--------------------------------------------------------------------------------

## When to Use This Skill

Activate this skill when:
- The user asks for 2D multi-view character sheet generation prompts optimized for 3D reconstruction.
- The user provides a 3-view character turnaround sheet (or separate front, side, back images) and requests a 3D model.
- The user asks to run single-model or warm-start batch 3D generation on Google Colab L4 GPU.
- The user wants to bake or fix multi-view textures (removing side striping, white border seams, or phantom arm projections) and export to Godot 4.

--------------------------------------------------------------------------------

## Pre-Flight Verification

Before running inference, verify the execution environment:

1. **Check Colab CLI Authentication**:
   ```bash
   colab --auth=oauth2 auth status
   ```
   *If unauthenticated, run `colab --auth=oauth2 auth login` or notify the user.*

2. **Check Local Blender Installation**:
   Ensure `/Applications/Blender.app/Contents/MacOS/Blender` is available for headless UV unwrapping, occlusion-aware PBR baking, and Cycles CPU verification rendering.
   > **Sandbox Note**: Inside macOS sandboxes, always run headless Blender with `CYCLES` + `CPU` (`scene.render.engine = 'CYCLES'`, `scene.cycles.device = 'CPU'`), never `BLENDER_EEVEE` (which fails compiling Metal shaders in `/var/folders`).

--------------------------------------------------------------------------------

## Standard 4-Stage Execution SOP

### Stage 1: 2D Three-View Sheet Preprocessing (`scripts/preprocess_views.py`)

When the user provides a 3-view turnaround sheet (Left = Front, Middle = Left Side, Right = Back on a white background):
- Uses **Exterior White Flood-Fill** (`_find_exterior_background`) from image borders so internal white regions (e.g., penguin belly, eye highlights) are 100% preserved.
- Uses **2D Connected-Component Extraction** so horizontally overlapping parts (e.g., a side view's long tail overlapping X-coordinates with the back view's outstretched arm) are separated cleanly without clipping.
- Aligns and pads all three views into `1024x1024` square RGBA PNGs (`{prefix}_{front,side,back}.png`).

```bash
python3 scripts/preprocess_views.py \
  --sheet <path/to/three_view_sheet.jpg> \
  --out-dir assets/inputs \
  --prefix <character_prefix>
```

---

### Stage 2: Cloud High-Res Geometry Generation (`scripts/runner.py` / `run_colab_inference.sh`)

#### Option A: Single-Session Warm-Start Batch Mode (Recommended for Multiple Models)
Runs 1 cold start (~299s including setup & 4.9GB weight load) followed by $N-1$ warm starts (~78–87s GPU inference per model) inside a single Colab L4 session, then automatically stops the session to conserve CCU (~0.21 CCU per 40k-face model):

```bash
python3 scripts/runner.py \
  --batch-prefixes <prefix1> <prefix2> <prefix3> \
  --engine hunyuan \
  --faces 40000 \
  --gpu L4
```
*Outputs raw 40,000-face geometry to `godot_project/assets/models/<prefix>.orig.glb`.*

#### Option B: Single Model End-to-End Inference
```bash
./run_colab_inference.sh \
  --front assets/inputs/<prefix>_front.png \
  --side assets/inputs/<prefix>_side.png \
  --back assets/inputs/<prefix>_back.png \
  --output godot_project/assets/models/<prefix>.glb \
  --engine hunyuan \
  --faces 40000 \
  --gpu L4
```

---

### Stage 3: Local Occlusion-Aware Harmonic PBR Baking (`scripts/bake_multiview_textures.py`)

Takes the raw cloud geometry (`<prefix>.orig.glb`) and the three `1024x1024` views to bake a seamless `2048x2048` PBR texture in headless Blender (~12 seconds per model):
1. Automatically runs Blender `smart_project` UV unwrapping if the mesh has no UVs.
2. Performs **3D `BVHTree` ray-cast self-occlusion rejection** (`vis_f, vis_b, vis_l, vis_r`) and **side-arm/flipper torso shadow suppression** (`side_arm_shadow`) so A-pose arms or side-view flippers never project phantom marks onto the ribcage/belly.
3. Solves **36-step Mesh Graph Laplacian Harmonic Inpainting** (`HarmonicColor`) across occluded grooves (e.g., between hedgehog quills or under arms), eliminating lateral zebra striping.
4. Bakes `2048x2048` Cycles geometric Ambient Occlusion (AO) and exports a Godot 4 `.glb` with velvety blind-box vinyl PBR settings (`Roughness = 0.68`, `Specular IOR Level = 0.22`).

```bash
python3 scripts/bake_multiview_textures.py \
  --mesh godot_project/assets/models/<prefix>.orig.glb \
  --front assets/inputs/<prefix>_front.png \
  --side assets/inputs/<prefix>_side.png \
  --back assets/inputs/<prefix>_back.png \
  -o godot_project/assets/models/<prefix>.glb \
  --texture-size 2048
```

---

### Stage 4: Visual Verification & 360° Turntable Rendering (`scripts/render_turntable.py`)

Always render and visually inspect the 4-view comparison strips (`render_<prefix>_textured.png` and `render_<prefix>_clay.png`) and optionally a 360° turntable video (`turntable_<prefix>.mp4`):

```bash
python3 scripts/render_turntable.py \
  --glb godot_project/assets/models/<prefix>.glb \
  --out-dir <artifact_or_render_dir> \
  --prefix <prefix> \
  --turntable
```

#### Dedicated 360° Showcase MP4 Generator (`scripts/generate_turntable_mp4.py`)
To generate commercial studio-quality 360° turntable showcase videos (720x720 / 1080x1080, 24fps, H.264, with glossy ground reflections and seamless cyclorama backdrops matching Tripo Studio style):

```bash
# Standard 720x720 24fps 4s studio dark turntable with floor reflection
python3 scripts/generate_turntable_mp4.py \
  --glb godot_project/assets/models/<prefix>.glb \
  -o <path/to/showcase.mp4> \
  --res 720x720 \
  --fps 24 \
  --duration 4.0 \
  --preset studio_dark

# Clay sculpt turntable
python3 scripts/generate_turntable_mp4.py \
  --glb godot_project/assets/models/<prefix>.glb \
  --mode clay \
  -o <path/to/showcase_clay.mp4>
```

--------------------------------------------------------------------------------

## 2D Multi-View Reference Sheet Prompt Template

When the user asks for prompts to generate 2D multi-view character sheets for this pipeline, enforce these constraints:
- **Layout**: `horizontal 21:9 aspect ratio`, three full-body orthographic views in a single row: **Left = Front view (A-pose)**, **Middle = Left Side profile view (facing left)**, **Right = Back view (A-pose)**.
- **Background**: `Pure solid flat white background (#FFFFFF), no ground shadow, no floor line, no text, no labels, generous white margin between the three views so they never touch or overlap`.
- **Style**: `3D blind-box vinyl toy render style, Pop Mart / Nintendo amiibo aesthetic, soft global illumination, matte clay/vinyl material, strictly aligned head-to-toe heights across all three views`.

--------------------------------------------------------------------------------

## Reference Documentation

- [Pipeline Architecture & Hybrid Cloud/Local Flow](./references/pipeline_architecture.md)
- [Godot 4 Integration & Collision Guidelines](./references/godot_guidelines.md)
