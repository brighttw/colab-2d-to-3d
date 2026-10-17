# Pipeline Architecture & ComfyUI Nodes

This document details the internal architecture of the Hunyuan3D-2 MV to Godot 4 generation pipeline.

--------------------------------------------------------------------------------

## Flow Diagram

```
[2D Multi-View Inputs]
 (Front, Back, Side)
         │
         ▼
[Node 10, 11, 12: LoadImage]
         │
         ▼
[Node 13, 14, 15: ImageRemBg (RMBG)]  <-- Automatic background removal
         │
         ▼
[Node 16, 17, 18: ImageResize / Normalize] <-- Scale to 518x518 square
         │
         ▼
[Node 20: Hunyuan3DMVModelLoader]
         │
         ▼
[Node 21: Hunyuan3DGenerateMV]        <-- Multi-view conditioned latent diffusion
         │
         ▼
[Node 30: Hunyuan3DDecodeToMesh]      <-- Watertight volumetric mesh extraction
         │
         ▼
[Node 32: MeshDecimate]               <-- Polygon reduction (max_facenum)
         │
         ▼
[Node 35: MeshUVUnwrap (xatlas)]      <-- Non-overlapping atlas generation
         │
         ▼
[Node 38: Hunyuan3DPaint / Bake]      <-- Multi-view texture projection & PBR bake
         │
         ▼
[Node 40: Hy3DExportMesh (.glb)]      <-- Standard Godot 4 compatible glTF export
```

--------------------------------------------------------------------------------

## Key Node Descriptions

1. **RMBG / Background Removal**:
   - Isolates the character/object from surrounding background noise to avoid geometry artifacts.
2. **Hunyuan3D-2 MV Geometry Synthesis**:
   - Synthesizes consistent 3D volume using multi-view priors (Front, Back, Left).
3. **Decimation (`max_facenum`)**:
   - Ingests raw high-density mesh (often 300,000+ polygons) and simplifies topology down to the target polygon count (`3000` to `25000`) while preserving feature contours and sharp edges.
4. **xatlas UV Parameterization**:
   - Generates clean, non-overlapping UV layout with appropriate padding, ensuring texture seams are minimized.
5. **Texture Baking & PBR Material**:
   - Projects diffuse textures and bakes normal and roughness attributes into standard glTF 2.0 PBR format.
