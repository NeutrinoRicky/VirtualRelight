# Codex Prompt: Implement Step 2 Optimization Script for Material and Lighting Decomposition

Please implement the **next stage after SOP initialization**, i.e. the actual optimization stage for:

**STEP 2: MATERIAL AND LIGHTING DECOMPOSITION**

Assume the previous SOP initialization stage is already completed and produced files such as:

- `sop_query_init.pt`
- `sop_query_init.npz`
- preview images for probe textures

We now want a **new standalone training script** that uses these initialized SOP assets and performs the **Step 2 optimization**.

Do **not** redesign the whole project. Reuse the current codebase as much as possible.

---

## Goal

Create a new script, for example:

- `train_stage2_sop_decomposition.py`

that performs Step 2 optimization for the relightable object using:

- G-buffer rendering from the current Gaussian object
- deferred PBR shading
- SOP-based efficient querying for indirect radiance and occlusion
- image reconstruction loss on the PBR rendering
- regularization losses for material decomposition
- optional SOP supervision from traced targets if easy to reuse

This stage should optimize the object's material and lighting-related parameters.

---

## High-level design

This Step 2 script should do the following:

1. Load the object reconstruction result from Stage 1
2. Load initialized SOP data from `sop_query_init.pt`
3. Build / load the environment map
4. Render current object G-buffers
5. Recover shading points from unbiased depth
6. Query SOPs at shading points to get:
   - indirect radiance
   - occlusion
7. Perform deferred PBR shading using:
   - albedo
   - roughness
   - metallic
   - direct light from envmap
   - indirect light and occlusion from SOP query
8. Compute Step 2 losses
9. Optimize relevant parameters
10. Save checkpoints / visualizations / logs

---

## Script requirements

Please create a **new training script** rather than overloading the old SOP initialization script.

Suggested name:

```python
train_stage2_sop_decomposition.py
```

If the repository already has a Stage 2 trace script or similar file, reuse its structure heavily.

---

## Inputs

The new script should support arguments like:

- Stage 1 checkpoint path
- SOP init file path (`sop_query_init.pt`)
- output directory
- iteration count
- learning rates
- envmap settings
- rendering sample count
- SOP query radius / topk
- whether to enable optional traced SOP supervision

---

## Required loaded data

From `sop_query_init.pt`, load at least:

- `probe_xyz`
- `probe_normal`
- `probe_lin_tex`
- `probe_occ_tex`

If available, also load:

- `probe_albedo_tex`
- `probe_roughness_tex`
- `probe_metallic_tex`
- `oct_dirs`

These extra fields can remain optional.

---

## Step 2 forward pipeline

Implement the forward pass in a clean and debuggable way.

### 1. Multi-target rendering

Render the current object and obtain at least:

- RGB
- weight
- depth
- normal
- albedo
- roughness
- metallic

Reuse existing rendering code if already available.

### 2. Unbiased depth

Use:

```python
depth_unbiased = depth / (weight + eps)
```

### 3. Build shading points

For valid object pixels:

- backproject depth to 3D world-space shading points
- get world-space normal
- compute outgoing view direction

Everything should be in a consistent coordinate system, preferably world space.

### 4. Hemisphere sampling

Use low-discrepancy or a simple stable hemisphere sampler around the shading normal.

A first correct version is more important than fancy sampling.

### 5. Direct lighting

Query environment map along sampled directions.

### 6. SOP querying

Use the already implemented SOP query path to get:

- indirect radiance
- occlusion

At minimum, support:

```python
Lin_x, Occ_x = query_sops(...)
```

If needed, extend the current SOP query helper to support per-direction querying for shading samples.
But do not overcomplicate the first version.

### 7. Illumination model

Use the paper-style decomposition:

- direct lighting = envmap
- indirect lighting = SOP radiance texture
- visibility modulation via SOP occlusion

Use a simple formulation:

```python
Li = (1 - Occ) * Ldir + Lin
```

### 8. Deferred PBR

Use current material buffers / attributes:

- albedo
- roughness
- metallic

Evaluate a simple and stable physically-based BRDF.
If the codebase already contains the Disney-style BRDF utilities, reuse them.

Then do Monte Carlo integration to get the Step 2 rendered image:

```python
rgb_pbr
```

---

## Step 2 losses

Implement the following losses.

### 1. Main PBR image loss

Use the rendered PBR image against GT object image:

- L1
- optionally SSIM if already available and cheap to reuse

Example:

```python
L_pbr = L1(rgb_pbr, gt_rgb) + 0.2 * (1 - SSIM(...))
```

Mask the loss to valid object pixels if appropriate.

### 2. Lambertian regularization

Encourage stable decomposition:

- roughness toward 1
- metallic toward 0

Example:

```python
L_lam = L1(roughness, 1) + L1(metallic, 0)
```

### 3. Optional SOP supervision

If tracing utilities are already reusable without too much extra complexity, support an optional loss:

- trace reference indirect radiance / occlusion
- supervise queried SOP results against traced targets

Example:

```python
L_sops = L1(Lin_query, Lin_trace) + L1(Occ_query, Occ_trace)
```

This should be behind a flag and can be disabled by default if it is not yet stable.

### 4. Geometry consistency reuse

If the existing Stage 2 / reconstruction pipeline already has reusable:

- depth-to-normal consistency
- mask loss

then preserve them as optional terms instead of deleting them.

---

## Optimization targets

The new script should optimize the material / lighting side only.

### Recommended default behavior

By default:

- optimize environment map
- optimize albedo-related parameters
- optimize roughness-related parameters
- optimize metallic-related parameters

Optionally:

- optimize SOP textures with a small learning rate
- keep object geometry mostly frozen at first
- keep position / scale / rotation frozen unless explicitly enabled

Please expose flags to control this.

---

## Optimizer design

Use separate parameter groups when possible:

- envmap LR
- material LR
- SOP texture LR

Suggested defaults:

- envmap / albedo / roughness / metallic: `1e-2`
- SOP textures: `1e-3`

These can be arguments.

---

## Checkpoint / logging behavior

Please implement practical outputs.

### Save checkpoints containing at least:

- iteration
- envmap state / capture
- optionally optimized SOP textures
- references to source Stage 1 checkpoint
- optimizer state if convenient

### Save visualizations periodically:

- current PBR render
- GT image
- albedo
- roughness
- metallic
- queried indirect radiance
- queried occlusion

### Save a small JSON or text summary of config and running statistics.

---

## Implementation style

Please follow these principles:

- keep the implementation concise
- prefer a correct baseline over a highly optimized one
- reuse current tracing / PBR / rendering modules if available
- do not change unrelated code paths
- add small helper functions if needed
- make tensor shapes explicit in comments
- avoid NaNs
- clamp roughness / metallic / occlusion to valid ranges
- keep all new code easy to inspect and debug

---

## Suggested file structure

Adapt to the repo, but something like this is fine:

- `train_stage2_sop_decomposition.py`
- possibly small helpers in:
  - `utils/sop_utils.py`
  - `utils/deferred_pbr_comgs.py`
  - `utils/tracing_comgs.py`

Only add what is truly needed.

---

## Concrete implementation tasks

Please complete the following:

1. create the new Step 2 training script
2. load Stage 1 object checkpoint
3. load SOP init data from `sop_query_init.pt`
4. build Step 2 forward pass
5. compute losses
6. build optimizer with parameter groups
7. run training loop
8. save checkpoints and visualizations
9. add enough comments so the logic is easy to follow

---

## Important scope control

Do not try to solve scene lighting estimation or object-scene composition in this step.
This script is only for the **object-side Step 2 optimization**:
**Material + Lighting Decomposition with SOP-based efficient querying**.

Keep the implementation focused on this stage only.
