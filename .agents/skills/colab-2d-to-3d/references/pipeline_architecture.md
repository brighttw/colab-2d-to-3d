# Pipeline Architecture: Hybrid Cloud Geometry + Local Occlusion-Aware PBR Baking

This document details the architecture of the upgraded `colab-2d-to-3d` pipeline for generating commercial blind-box quality 3D models (`40,000` faces, `2048x2048` PBR textures) for Godot 4.

--------------------------------------------------------------------------------

## End-to-End Flow Diagram

```text
[Stage 1: Local 2D Preprocessing (scripts/preprocess_views.py)]
 3-View Turnaround Sheet (Front | Left Side | Back)
          │
          ├──> Exterior White Flood-Fill Alpha Masking (preserves white bellies & eye highlights)
          ├──> 2D Connected-Component View Extraction (prevents tail/arm horizontal clipping)
          └──> 1024x1024 Square Canvas Normalization ({prefix}_{front,side,back}.png)
          │
          ▼
[Stage 2: Cloud High-Res Geometry Synthesis (Colab L4 GPU / workflow_api.json)]
 [Nodes 10, 11, 12: LoadImage (Front, Back, Side)]
          │
          ▼
 [Nodes 13, 14, 15: ImageRemoveBackground+ (Patched with Exterior Flood-Fill + 1px Gaussian AA)]
          │
          ▼
 [Node 30: Hy3DGenerateMeshMultiView (steps=50, guidance_scale=6.0, DINOv2 Conditioner Patched)]
          │
          ▼
 [Node 31: Hy3DVAEDecode (octree_resolution=384, num_chunks=16000)]
          │
          ▼
 [Node 32: Hy3DPostprocessMesh (max_facenum=40000, remove_floaters=true, smooth_normals=true)]
          │
          ▼
 [Node 40: Hy3DExportMesh -> Downloads <prefix>.orig.glb (Raw 40K-face Watertight Mesh)]
          │
          ▼
[Stage 3: Local Occlusion-Aware Harmonic PBR Baker (scripts/bake_multiview_textures.py)]
 [Blender Headless (CYCLES CPU)]
          │
          ├──> Auto Smart UV Unwrap (if .orig.glb has no UVs)
          ├──> 2D Silhouette Distance Transform (ero_px=12) + 8-step Morphological Color Dilation
          ├──> 3D BVHTree Ray-Cast Self-Occlusion Rejection (vis_f, vis_b, vis_l, vis_r)
          ├──> Side-Arm / Flipper Torso Shadow Suppression (eliminates phantom limb stamps on belly)
          ├──> 36-Step Mesh Graph Laplacian Harmonic Inpainting (fills quill grooves & underarms)
          ├──> Cycles 2048x2048 Geometric AO Baking + Multi-View High-Res Shader Bake
          └──> Exports <prefix>.glb (Roughness=0.68, Specular IOR=0.22) + Headless Godot 4 Reimport
          │
          ▼
[Stage 4: Automated Verification (scripts/render_turntable.py)]
          └──> Renders 4-view Textured & Clay strips + 360° Turntable MP4 video
```

--------------------------------------------------------------------------------

## Key Architectural Decisions

1. **Why Bypass Cloud Single-View Texture Nodes (Nodes 33–39)?**
   - In `ComfyUI-Hunyuan3DWrapper`, `Hy3DSampleMultiView` only accepts a single reference image (`ref_image`), ignoring the side and back views during texture synthesis and causing severe side striping and color hallucination.
   - Offloading UV unwrapping and multi-view PBR baking to `scripts/bake_multiview_textures.py` on local Blender CPU takes only ~12 seconds per model, saves ~90 seconds of L4 GPU time per model, and utilizes all 3 views with exact 3D self-occlusion awareness.
2. **Why Inject the Exterior Flood-Fill Patch into `ComfyUI_essentials`?**
   - Default `u2net` background removal misclassifies bright white bellies (such as a penguin's white stomach) as background, carving holes in the 3D conditioning mask. Border-connected flood-fill preserves 100% of internal bright regions.
3. **Why Use 3D `BVHTree` + Mesh Graph Laplacian Harmonic Diffusion?**
   - Simple normal-based tri-planar blending projects A-pose arms onto the side of the torso and projects front belly colors onto the tips of back quills. Ray-casting against the 3D `BVHTree` rejects self-occluded rays, while 36 iterations of Graph Laplacian diffusion smoothly interpolate colors across occluded creases.
