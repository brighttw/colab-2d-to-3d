# Godot 4 Integration & Collision Guidelines

This guide details best practices for importing, configuring, and utilizing `.glb` assets produced by the `colab-2d-to-3d` pipeline inside Godot 4.

--------------------------------------------------------------------------------

## Asset Ingestion

When a new `.glb` model is written to `godot_project/assets/models/<name>.glb`:

1. **Headless Import**:
   ```bash
   godot --headless --path ./godot_project --import --quit
   ```
2. **Default Import Settings in Godot 4**:
   - Godot 4 automatically imports `.glb` as a `PackedScene`.
   - Materials are mapped to `StandardMaterial3D`.
   - Normal map format is direct compatibility (OpenGL Y+).

--------------------------------------------------------------------------------

## Collision Body Generation

Depending on the role of the 3D asset in Godot 4:

1. **Dynamic Objects / Characters (`CharacterBody3D` / `RigidBody3D`)**:
   - **Recommended**: Generate a **Convex Polygon Shape** (or multiple simplified convex hulls) or wrap with a basic `CapsuleShape3D` / `CylinderShape3D`.
   - Avoid using Trimesh collision for dynamic bodies to prevent physics performance drops and tunnelling glitches.
   - In Godot Editor: Select imported MeshInstance3D -> Menu `Mesh` -> `Create Single Convex Collision Sibling`.

2. **Static Environment / Props (`StaticBody3D`)**:
   - For non-moving structures or intricate buildings:
   - In Godot Editor: Menu `Mesh` -> `Create Trimesh Static Body`.
   - If decimated to under 10,000 faces, Trimesh collision evaluates efficiently in Godot 4's physics engine.

--------------------------------------------------------------------------------

## PBR Material Adjustments

Hunyuan3D-2 exports embedded PBR textures inside the `.glb` container. If adjustments are needed:

- **Roughness / Metallic**: If the surface appears overly glossy, add an override material or adjust `roughness` slider in the Inspector.
- **Backface Culling**: Enabled by default. For thin cloth or double-sided elements, switch Cull Mode to `Disabled` in the material inspector.
- **Shadow Casting**: Ensure `Cast Shadow` is set to `On` on the `MeshInstance3D`.
