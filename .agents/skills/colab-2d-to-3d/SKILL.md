---
name: colab-2d-to-3d
description: >-
  Generate Godot 4-compatible 3D assets (.glb) from 2D multi-view images (front, back, side)
  using Hunyuan3D-2 MV and ComfyUI on Google Colab (L4 GPU). Use this skill whenever the user
  wants to convert 2D concept art or orthographic character views into 3D meshes, run the
  automated Colab inference pipeline, or import and configure generated 3D models into Godot 4.
---

# Colab 2D-to-3D Skill

This skill provides an automated workflow to turn 2D multi-view concept artwork (front, back, side views) into game-ready 3D assets (`.glb`) with PBR textures, decimated polycounts, and collision support for Godot 4.

--------------------------------------------------------------------------------

## When to Use This Skill

Activate this skill when:
- The user provides 2D reference images (orthographic front, back, and side views) and requests a 3D model.
- The user asks to run or test the `colab-2d-to-3d` pipeline.
- The user wants to generate low/mid-poly 3D models optimized for Godot 4.
- The user requests automated remote Colab GPU inference without opening a web browser.

--------------------------------------------------------------------------------

## Pre-Flight Verification

Before running inference, verify the execution environment:

1. **Check Colab CLI Authentication**:
   ```bash
   colab --auth=oauth2 auth status
   ```
   *If unauthenticated, run `colab --auth=oauth2 auth login` or notify the user.*

2. **Verify Python Dependencies**:
   ```bash
   pip install -q -r requirements.txt
   ```

3. **Verify Input Images**:
   Confirm that all three view images exist and are readable:
   - Front view (`--front`): Centered, facing forward.
   - Back view (`--back`): Facing backward.
   - Side view (`--side`): Facing left or right (if right, apply horizontal mirror).

--------------------------------------------------------------------------------

## Execution Workflows

### Mode 1: Fully Automated Colab CLI (Recommended)

Spawns a headless Google Colab session on an NVIDIA L4 GPU, uploads assets, executes the Hunyuan3D-2 MV ComfyUI pipeline, downloads the generated `.glb`, and cleans up the session to conserve compute units.

```bash
./run_colab_inference.sh \
  --front <path/to/front.png> \
  --back <path/to/back.png> \
  --side <path/to/side.png> \
  --output godot_project/assets/models/<model_name>.glb \
  --faces 15000 \
  --gpu L4
```

#### Key Options:
- `--faces <COUNT>`: Target triangle count (see Polygon Count Budget below). Default is `15000`.
- `--gpu <MODEL>`: `L4` (24GB VRAM, recommended) or `A100`.
- `--keep-session`: Keep the Colab VM alive for rapid batch testing (prevents repeated session startup overhead).
- `--session <NAME>`: Reuse an existing named session.

---

### Mode 2: Direct ComfyUI API Execution

Used when a remote ComfyUI instance is already running (e.g., via Cloudflare tunnel `trycloudflare.com` or local LAN).

```bash
export COMFY_ENDPOINT="https://xxxx-xxxx.trycloudflare.com"

python3 agent_3d_skill.py \
  <path/to/front.png> \
  <path/to/back.png> \
  <path/to/side.png> \
  godot_project/assets/models/<model_name>.glb \
  15000 \
  "$COMFY_ENDPOINT" \
  --side-orientation left
```

*Note: If the side image faces right, specify `--side-orientation right` to automatically mirror it to the standard Hunyuan3D-2 left coordinate system.*

---

### Mode 3: Godot 4 Asset Ingestion & Refresh

After saving the `.glb` model into `godot_project/assets/models/`:

1. **Trigger Headless Reimport**:
   ```bash
   godot --headless --path ./godot_project --import --quit
   ```

2. **Verify Asset Status**:
   - Check file size: A normal decimated `.glb` with 2048x2048 PBR maps should be between 5 MB and 25 MB.
   - Check Godot `.import` file: Ensure `godot_project/assets/models/<model_name>.glb.import` is created.

--------------------------------------------------------------------------------

## Polygon Count Budget & Settings

Configure `--faces` according to asset category:

| Asset Type | Target Faces (`--faces`) | Texture Res | Use Case |
| :--- | :--- | :--- | :--- |
| **Props / Collectibles** | 3,000 – 5,000 | 1024×1024 | Potions, weapons, crates, pickups |
| **Environment / Vehicles** | 8,000 – 12,000 | 2048×2048 | Buildings, obstacles, vehicles, props |
| **Hero Characters / Bosses** | 15,000 – 25,000 | 2048×2048 | Main playable characters, major NPCs |

--------------------------------------------------------------------------------

## Troubleshooting & Failure Modes

1. **Colab Session Creation Timeout / Out of Quota**:
   - Error: `Quota exceeded` or `ResourceExhausted`.
   - Resolution: Switch to `--gpu A100` if L4 is unavailable, or report to the user that Colab compute units need replenishing.
2. **WebSocket Disconnection during Generation**:
   - The runner automatically falls back to HTTP polling via `/history/<prompt_id>`. Do not abort unless the timeout (3600s) expires.
3. **Black or Distorted Textures**:
   - Ensure input images have clear silhouettes and consistent lighting. RMBG will clean the background, but overlapping shadows may affect depth maps.
4. **Incorrect Orientation in Godot**:
   - Hunyuan3D-2 exports standard Y-up glTF format. Check if the model requires a $180^\circ$ yaw rotation around the Y-axis inside the Godot scene.

--------------------------------------------------------------------------------

## Reference Documentation

- [Pipeline Architecture & ComfyUI Nodes](./references/pipeline_architecture.md)
- [Godot 4 Integration & Collision Guidelines](./references/godot_guidelines.md)
