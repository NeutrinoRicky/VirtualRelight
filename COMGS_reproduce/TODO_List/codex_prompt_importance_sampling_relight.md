# Codex Prompt: Replace uniform hemisphere sampling with importance sampling in `render_stage2_sop_comgs.py`

Please modify the current Stage2 SOP rendering code so that object relighting uses **importance sampling** from the environment lighting instead of the current **uniform hemisphere Hammersley sampling**.

Target file:
- `render_stage2_sop_comgs.py`

Relevant existing utilities already available:
- `LatLongEnvMap.sample_light_directions(...)`
- `LatLongEnvMap.light_pdf(...)`
- `integrate_incident_radiance(...)`
- `query_sops_directional(...)`

Do **not** redesign the whole pipeline. Keep the current deferred PBR + SOP querying structure, and only replace the light-direction sampling / Monte Carlo estimator path.

---

## Goal

Current code path:

- recover shading points
- uniformly sample hemisphere directions using `sample_hemisphere_hammersley(...)`
- query envmap for direct light
- query SOP for indirect radiance + occlusion
- integrate BRDF

We want to change this to:

- sample directions from the **environment lighting PDF**
- keep only directions above the local surface hemisphere
- use the corresponding **direction PDF** in the Monte Carlo estimator
- preserve SOP querying and deferred PBR structure

This should better match the paper-style rendering logic:
- direct light comes from the new HDR environment map
- SOP provides cached indirect radiance / occlusion
- Monte Carlo integration uses importance sampling from the lighting distribution

---

## Core idea to implement

For each shading point `x` with normal `n`:

1. Sample candidate directions from the environment map importance sampler:
   - use `envmap.sample_light_directions(batch_size, sample_num, training=False or training flag)`
   - this returns directions and PDFs based on envmap intensity

2. Keep only directions in the visible local hemisphere:
   - require `dot(n, wi) > 0`
   - because relighting integral is over the upper hemisphere

3. For the kept directions:
   - query direct light from envmap
   - query SOP indirect radiance + occlusion
   - build incident radiance:
     - `Li = (1 - Occ) * Ldir + Lin`

4. Evaluate BRDF and integrate using the Monte Carlo estimator:
   - for each sample:
     - contribution = `f * Li * max(dot(n, wi), 0) / pdf(wi)`
   - average over valid samples

5. Return final relit RGB

Important:
- the estimator must use the **importance sampling PDF**
- do not use the previous constant solid-angle factor from uniform hemisphere sampling
- keep everything numerically stable

---

## What to change

### 1. Add a new integration helper

Please add a helper in `deferred_pbr_comgs.py` or locally in `render_stage2_sop_comgs.py`, something like:

```python
def integrate_incident_radiance_importance(
    albedo,         # [N, 3]
    roughness,      # [N, 1]
    metallic,       # [N, 1]
    normals,        # [N, 3]
    viewdirs,       # [N, 3]
    lightdirs,      # [N, S, 3]
    incident_radiance,  # [N, S, 3]
    light_pdf,      # [N, S, 1]
    eps=1e-6,
):
    ...
```

Implementation idea:

1. Reuse `evaluate_microfacet_brdf(...)`
2. Compute:
   - `n_dot_l = max(dot(n, wi), 0)`
3. Per-sample contribution:
   - `contrib = brdf * incident_radiance * n_dot_l / clamp(light_pdf, min=eps)`
4. Average over valid samples:
   - `rgb = contrib.mean(dim=1)`

Return:
- `rgb`
- optional debug dict with diffuse/specular terms

This is the correct importance-sampling-style estimator for directions sampled from `pdf(wi)`.

---

### 2. Replace uniform hemisphere sampling in `render_stage2_sop_view`

Current code uses:

```python
lightdirs, _pdf, sample_solid_angle = sample_hemisphere_hammersley(...)
```

Please replace this part with an envmap importance sampling path.

Suggested implementation sketch:

```python
candidate_dirs, candidate_pdf = envmap.sample_light_directions(
    batch_size=pts.shape[0],
    sample_num=num_shading_samples,
    training=randomized_samples,
)
```

Then compute:

```python
n_dot_l = (nrm[:, None, :] * candidate_dirs).sum(dim=-1, keepdim=True)
valid_hemi = n_dot_l > 0
```

Because the envmap sampler is global (not restricted to each point's hemisphere), please handle lower-hemisphere samples safely.

Two acceptable implementations:

#### Preferred simple version
Keep all sampled directions, but zero out contributions where `n_dot_l <= 0`.
That is:
- still query SOP / envmap on all sampled directions
- integration uses `max(n_dot_l, 0)`
- contributions from invalid hemisphere directions automatically become zero

This is simplest and keeps tensor shapes stable.

#### Optional improved version
Resample until enough positive-hemisphere directions are obtained.
Only do this if it stays concise.

For now, prefer the simple version.

---

### 3. Use envmap PDF in integration

Current code:

```python
pbr_rgb, _aux = integrate_incident_radiance(..., sample_solid_angle=...)
```

Replace it with the new importance estimator:

```python
pbr_rgb, _aux = integrate_incident_radiance_importance(
    albedo=albedo,
    roughness=roughness,
    metallic=metallic,
    normals=nrm,
    viewdirs=viewdirs,
    lightdirs=lightdirs,
    incident_radiance=incident_radiance,
    light_pdf=light_pdf,
)
```

Make sure the PDF is the one corresponding to the sampled directions.

---

### 4. Keep SOP logic unchanged

Do **not** remove SOP usage.

This line should remain conceptually the same:

```python
query_indirect, query_occlusion = query_sops_directional(...)
```

And incident radiance remains:

```python
incident_radiance = (1.0 - query_occlusion) * direct_radiance + query_indirect
```

This is important:
- the **new HDR envmap** changes `direct_radiance`
- SOP still provides cached **indirect radiance** and **occlusion**
- the final relighting uses both

---

### 5. Add profiling split

Please refine timing breakdown in `render_stage2_sop_view`:

- `gbuffer_render_sec`
- `env_sampling_sec`
- `env_lookup_sec`
- `sop_query_sec`
- `brdf_integrate_sec`
- `sop_query_shading_sec` as the total shading block

This helps compare the cost before/after switching to importance sampling.

---

### 6. Add optional comparison flag

Please add a render flag like:

```python
--sampling_mode {uniform,env_importance}
```

Default:
- `env_importance`

Behavior:
- `uniform`: keep old Hammersley path
- `env_importance`: use new envmap importance sampling path

This is useful for direct A/B comparison.

---

## Important conceptual note

The relighting logic is:

- SOP stores cached directional information from the original captured scene/object reconstruction:
  - indirect radiance `Lin`
  - occlusion `Occ`
- The **new HDR environment map** does **not** overwrite SOP
- Instead, it changes the **direct lighting term**:
  - `Ldir = envmap(wi)`
- Final incident lighting becomes:
  - `Li = (1 - Occ) * Ldir + Lin`

So:
- changing the HDR envmap directly changes the relit appearance through `Ldir`
- SOP modulates how much of this direct light reaches the point and adds cached indirect light
- this is why a new HDR can relight the object even though SOP is still loaded from prior data

Please keep this logic unchanged in code and mention it in comments where helpful.

---

## Output expectations

Please make the code:
- concise
- easy to debug
- numerically stable
- minimally invasive to the current structure

Please also:
- keep current visualization outputs working
- preserve existing saved debug images
- update any labels or summary logs if needed

Do not over-engineer.
