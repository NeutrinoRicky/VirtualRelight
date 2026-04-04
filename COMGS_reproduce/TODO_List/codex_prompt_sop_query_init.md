# Codex Prompt: Implement SOP Efficient Querying + Texture Initialization

Please implement **only** the following two parts for Phase B in the ComGS-style pipeline:

1. **Efficient Querying through SOPs**
2. **Initialization of SOP radiance and occlusion textures**

Do **not** modify unrelated training logic, rendering architecture, or loss definitions unless strictly necessary for these two parts.

---

## Goal

We already have:
- surface points / normals for SOP placement
- SOP centers `p_k`
- SOP normals `n_k`
- one octahedral texture per probe
- object geometry / relightable rendering pipeline basics

We now need:

- a clean **SOP query function** that returns **indirect radiance** and **occlusion** at arbitrary shading points
- a clean **initialization pipeline** for each SOP's **radiance texture** `L_in` and **occlusion texture** `O`

The implementation should follow the paper logic:

- each SOP stores directional **radiance** and **occlusion** in an **octahedral texture**
- at a shading point `x`, use nearby SOPs and interpolate their queried values
- interpolation uses **spatial weight** and **back-face weight**
- SOP textures are initialized by **ray tracing** from each probe over octahedral directions

---

## Part 1: Efficient Querying through SOPs

### Required API

Implement a function with roughly this interface:

```python
Lin_x, Occ_x = query_sops(
    x_world,          # [P, 3] shading points
    probe_xyz,        # [K, 3]
    probe_normal,     # [K, 3]
    probe_lin_tex,    # [K, Ht, Wt, 3]
    probe_occ_tex,    # [K, Ht, Wt, 1]
    radius=None,
    topk=8,
    eps=1e-8,
)
```

Return:
- `Lin_x`: `[P, 3]` interpolated indirect radiance
- `Occ_x`: `[P, 1]` interpolated occlusion

### Query logic

For each shading point `x`:

1. Find neighboring SOPs around `x`
   - prefer radius search if available
   - otherwise use KNN / top-k nearest probes
   - if radius search returns empty, fallback to nearest top-k

2. For each neighbor SOP `k`:
   - compute direction
     `d_k = p_k - x`
   - normalize direction
     `dir_k = d_k / (||d_k|| + eps)`

3. Compute interpolation weights:

- **spatial weight**
  `w_s = 1 / (||d_k|| + eps)`

- **back-face weight**
  `w_b = 0.5 * (1 + dot(dir_k, n_k)) + 0.01`

- combined weight
  `w = w_s * w_b`

4. Use `dir_k` to query the probe's octahedral texture:
   - sample `probe_lin_tex[k]` to get directional indirect radiance
   - sample `probe_occ_tex[k]` to get directional occlusion

5. Interpolate:
   - `Lin(x) = sum_k w_k * Lin_k / sum_k w_k`
   - `Occ(x) = sum_k w_k * Occ_k / sum_k w_k`

### Important details

- everything should be in **world space**
- probe normals should be normalized
- clamp final occlusion to `[0, 1]`
- if neighbor count is zero even after fallback, return zeros safely
- avoid NaNs
- keep the implementation vectorized where possible
- write the code so it is easy to debug

### Also implement

A helper for octahedral lookup, e.g.

```python
sample_octahedral_texture(texture, dirs)
```

that:
- maps 3D unit direction to octahedral UV
- bilinearly samples the texture
- supports batched directions

---

## Part 2: Initialization of SOP Radiance and Occlusion Textures

### Required API

Implement something like:

```python
probe_lin_tex, probe_occ_tex = init_sop_textures(
    probe_xyz,        # [K, 3]
    probe_normal,     # [K, 3]
    tex_h=16,
    tex_w=16,
    ...
)
```

Return:
- `probe_lin_tex`: `[K, tex_h, tex_w, 3]`
- `probe_occ_tex`: `[K, tex_h, tex_w, 1]`

### Initialization logic

Each probe stores a directional field over its octahedral texture.

For every probe `k` and every texel `(u, v)`:

1. Convert octahedral texel center to a unit 3D direction `omega`
2. Cast a ray from the probe position `p_k` along `omega`
3. Use ray tracing / scene intersection / existing tracing utility to estimate:

- **occlusion** `O(omega)`
  - scalar in `[0, 1]`
  - 0 means unoccluded, 1 means fully occluded
  - any consistent convention is fine, but keep it consistent with later querying

- **indirect radiance** `L_in(omega)`
  - RGB value along that direction
  - this is the cached indirect lighting term for the SOP

4. Store them into the probe textures

### Practical constraints

- if full high-quality tracing is too slow, first implement a clean version that is correct
- allow chunked processing over probes and/or texels to avoid OOM
- add a small ray origin bias along the ray direction or probe normal to reduce self-intersection
- if a ray misses everything:
  - occlusion should be 0
  - indirect radiance can be 0

### Important

Please keep this initialization code independent and reusable.
It should be possible to:
- run it once before training
- save the initialized textures
- reload them later

---

## Expected file/function organization

Please add or update only the minimum necessary files. A reasonable structure is:

- `utils/sop_utils.py`
  - `sample_octahedral_texture(...)`
  - `query_sops(...)`
  - `oct_uv_to_dir(...)`
  - `dir_to_oct_uv(...)`

- `utils/sop_init.py`
  - `init_sop_textures(...)`
  - helper(s) for tracing per probe / per texel

If the project already has better file locations, adapt to the existing style instead of forcing new files.

---

## Implementation requirements

- keep the code concise and readable
- include clear docstrings
- include shape comments for tensors
- avoid changing unrelated code paths
- if a tracing backend already exists, reuse it instead of inventing a new renderer
- prefer a first version that is **correct and debuggable**
- add a tiny smoke test or debug utility if easy

---

## Deliverables

Please complete all of the following:

1. implement the SOP query path
2. implement octahedral texture sampling
3. implement SOP texture initialization
4. wire them minimally so other code can call them
5. briefly explain any assumptions in comments

Do not over-engineer. Focus only on these two features.
