from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from gaussian_renderer import render_multitarget
from utils.deferred_pbr_comgs import shade_secondary_points
from utils.irgs_trace_adapter import IRGS2DGaussianTraceAdapter



def _safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm

@dataclass
class TraceBackendConfig:
    backend: str = "auto"
    trace_bias: float = 1e-3
    secondary_num_samples: int = 16
    rebuild_every: int = 0
    open3d_voxel_size: float = 0.004
    open3d_sdf_trunc: float = 0.02
    open3d_depth_trunc: float = 0.0
    open3d_mask_background: bool = True
    native_alpha_min: float = 1.0 / 255.0
    native_transmittance_min: float = 0.03


class BaseTraceBackend:
    backend_name = "base"

    def rebuild(self, *args, **kwargs):
        raise NotImplementedError

    def _trace_hits_flat(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        camera_center: Optional[torch.Tensor] = None,
        trace_feature_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def trace(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        envmap,
        secondary_num_samples: int = 16,
        randomized_secondary: bool = False,
        camera_center: Optional[torch.Tensor] = None,
        trace_feature_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        prefix = ray_origins.shape[:-1]
        rays_o = ray_origins.reshape(-1, 3)
        rays_d = F.normalize(ray_directions.reshape(-1, 3), dim=-1, eps=1e-6)
        hits = self._trace_hits_flat(
            rays_o,
            rays_d,
            camera_center=camera_center,
            trace_feature_mode=trace_feature_mode,
        )

        hit_mask = hits["hit_mask"]
        incident = torch.zeros((rays_o.shape[0], 3), device=rays_o.device, dtype=rays_o.dtype)
        if int(hit_mask.sum().item()) > 0:
            shaded, _ = shade_secondary_points(
                envmap=envmap,
                albedo=hits["albedo"][hit_mask],
                roughness=hits["roughness"][hit_mask],
                metallic=hits["metallic"][hit_mask],
                normals=hits["hit_normal"][hit_mask],
                viewdirs=-rays_d[hit_mask],
                num_samples=secondary_num_samples,
                randomized=randomized_secondary,
            )
            incident[hit_mask] = shaded

        occlusion = hit_mask.to(dtype=rays_o.dtype).unsqueeze(-1)
        outputs = {
            "occlusion": occlusion.view(*prefix, 1),
            "incident_radiance": incident.view(*prefix, 3),
            "hit_mask": hit_mask.view(*prefix),
            "hit_distance": hits["hit_distance"].view(*prefix),
            "hit_points": hits["hit_points"].view(*prefix, 3),
            "hit_normal": hits["hit_normal"].view(*prefix, 3),
            "hit_albedo": hits["albedo"].view(*prefix, 3),
            "hit_roughness": hits["roughness"].view(*prefix, 1),
            "hit_metallic": hits["metallic"].view(*prefix, 1),
        }
        if "trace_color" in hits:
            outputs["trace_color"] = hits["trace_color"].view(*prefix, 3)
        return outputs

    def trace_occlusion(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rays_o = ray_origins.reshape(-1, 3)
        rays_d = F.normalize(ray_directions.reshape(-1, 3), dim=-1, eps=1e-6)
        hits = self._trace_hits_flat(rays_o, rays_d, camera_center=camera_center)
        return hits["hit_mask"].to(dtype=ray_origins.dtype).view(*ray_origins.shape[:-1], 1)

    def trace_incident_radiance(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        envmap,
        secondary_num_samples: int = 16,
        randomized_secondary: bool = False,
        camera_center: Optional[torch.Tensor] = None,
        trace_feature_mode: Optional[str] = None,
    ) -> torch.Tensor:
        return self.trace(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            envmap=envmap,
            secondary_num_samples=secondary_num_samples,
            randomized_secondary=randomized_secondary,
            camera_center=camera_center,
            trace_feature_mode=trace_feature_mode,
        )["incident_radiance"]


class IRGSNativeGaussianTraceBackend(BaseTraceBackend):
    backend_name = "irgs_native"

    def __init__(self, gaussians, alpha_min: float = 1.0 / 255.0, transmittance_min: float = 0.03):
        self.gaussians = gaussians
        self.adapter = IRGS2DGaussianTraceAdapter(
            gaussians=gaussians,
            alpha_min=alpha_min,
            transmittance_min=transmittance_min,
        )

    @torch.no_grad()
    def rebuild(self, gaussians=None, **_unused):
        if gaussians is not None:
            self.gaussians = gaussians
        self.adapter.rebuild(gaussians=self.gaussians)

    def _trace_hits_flat(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        camera_center: Optional[torch.Tensor] = None,
        trace_feature_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        trace_outputs = self.adapter.trace(
            ray_origins=rays_o,
            ray_directions=rays_d,
            camera_center=camera_center,
            feature_mode=trace_feature_mode or "comgs_pbr",
        )

        return {
            "hit_mask": trace_outputs["hit_mask"],
            "hit_distance": trace_outputs["depth"],
            "hit_points": trace_outputs["position"],
            "hit_normal": _safe_normalize(trace_outputs["normal"]),
            "albedo": trace_outputs["albedo"],
            "roughness": trace_outputs["roughness"],
            "metallic": trace_outputs["metallic"],
            "trace_color": torch.clamp(trace_outputs["color"], 0.0, 1.0),
        }


class IRGSAdapterTraceBackend(BaseTraceBackend):
    backend_name = "irgs_adapter"
    default_feature_mode = "comgs_pbr"

    def __init__(self, gaussians, alpha_min: float = 1.0 / 255.0, transmittance_min: float = 0.03):
        self.gaussians = gaussians
        self.adapter = IRGS2DGaussianTraceAdapter(
            gaussians=gaussians,
            alpha_min=alpha_min,
            transmittance_min=transmittance_min,
        )

    @torch.no_grad()
    def rebuild(self, gaussians=None, **_unused):
        if gaussians is not None:
            self.gaussians = gaussians
        self.adapter.rebuild(gaussians=self.gaussians)

    def _resolve_feature_mode(self, trace_feature_mode: Optional[str]) -> str:
        return trace_feature_mode or self.default_feature_mode

    def _adapter_trace(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        camera_center: Optional[torch.Tensor] = None,
        trace_feature_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.adapter.trace(
            ray_origins=rays_o,
            ray_directions=rays_d,
            camera_center=camera_center,
            feature_mode=self._resolve_feature_mode(trace_feature_mode),
        )

    def _trace_hits_flat(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        camera_center: Optional[torch.Tensor] = None,
        trace_feature_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        trace_outputs = self._adapter_trace(
            rays_o=rays_o,
            rays_d=rays_d,
            camera_center=camera_center,
            trace_feature_mode=trace_feature_mode,
        )
        return {
            "hit_mask": trace_outputs["hit_mask"],
            "hit_distance": trace_outputs["depth"],
            "hit_points": trace_outputs["position"],
            "hit_normal": _safe_normalize(trace_outputs["normal"]),
            "albedo": trace_outputs["albedo"],
            "roughness": trace_outputs["roughness"],
            "metallic": trace_outputs["metallic"],
            "trace_color": trace_outputs["color"],
            "trace_alpha": trace_outputs["alpha"],
        }

    def trace(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        envmap,
        secondary_num_samples: int = 16,
        randomized_secondary: bool = False,
        camera_center: Optional[torch.Tensor] = None,
        trace_feature_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del envmap, secondary_num_samples, randomized_secondary
        trace_outputs = self._adapter_trace(
            rays_o=ray_origins,
            rays_d=ray_directions,
            camera_center=camera_center,
            trace_feature_mode=trace_feature_mode,
        )
        return {
            "occlusion": trace_outputs["alpha"].unsqueeze(-1),
            "incident_radiance": trace_outputs["color"],
            "hit_mask": trace_outputs["hit_mask"],
            "hit_distance": trace_outputs["depth"],
            "hit_points": trace_outputs["position"],
            "hit_normal": _safe_normalize(trace_outputs["normal"]),
            "hit_albedo": trace_outputs["albedo"],
            "hit_roughness": trace_outputs["roughness"],
            "hit_metallic": trace_outputs["metallic"],
            "trace_color": trace_outputs["color"],
            "trace_alpha": trace_outputs["alpha"].unsqueeze(-1),
        }

    def trace_occlusion(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        camera_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        trace_outputs = self.adapter.trace(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            camera_center=camera_center,
        )
        return trace_outputs["alpha"].unsqueeze(-1)


class IRGSAdapterCompatTraceBackend(IRGSAdapterTraceBackend):
    backend_name = "irgs_adapter_compat"
    default_feature_mode = "irgs_base_rough"


class Open3DMeshTraceBackend(BaseTraceBackend):
    backend_name = "open3d_mesh"

    def __init__(
        self,
        scene,
        gaussians,
        pipe,
        background,
        voxel_size: float = 0.004,
        sdf_trunc: float = 0.02,
        depth_trunc: float = 0.0,
        mask_background: bool = True,
    ):
        self.scene = scene
        self.gaussians = gaussians
        self.pipe = pipe
        self.background = background
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.depth_trunc = depth_trunc
        self.mask_background = mask_background
        self._o3d = None
        self._ray_scene = None
        self._triangles = None
        self._vertex_albedo = None
        self._vertex_roughness = None
        self._vertex_metallic = None

    def _import_open3d(self):
        if self._o3d is None:
            import open3d as o3d

            self._o3d = o3d
        return self._o3d

    def _make_attribute_render(self, attribute_name: str):
        def _render(viewpoint_camera, gaussians, pipe, bg_color):
            render_pkg = render_multitarget(viewpoint_camera, gaussians, pipe, bg_color)
            if attribute_name == "albedo":
                rgb = render_pkg["albedo"]
            elif attribute_name == "roughness":
                rgb = render_pkg["roughness"].repeat(3, 1, 1)
            elif attribute_name == "metallic":
                rgb = render_pkg["metallic"].repeat(3, 1, 1)
            else:
                raise ValueError(f"Unsupported mesh attribute render target: {attribute_name}")
            return {
                "render": rgb,
                "surf_depth": render_pkg["surf_depth"],
                "surf_normal": render_pkg["surf_normal"],
                "rend_alpha": render_pkg["rend_alpha"],
                "rend_normal": render_pkg["rend_normal"],
            }

        return _render

    @torch.no_grad()
    def _extract_attribute_mesh(self, attribute_name: str):
        from utils.mesh_utils import GaussianExtractor

        bg_list = self.background.detach().cpu().tolist() if torch.is_tensor(self.background) else self.background
        extractor = GaussianExtractor(
            self.gaussians,
            self._make_attribute_render(attribute_name),
            self.pipe,
            bg_color=bg_list,
        )
        extractor.reconstruction(self.scene.getTrainCameras())
        depth_trunc = self.depth_trunc if self.depth_trunc > 0 else max(extractor.radius * 2.0, self.scene.cameras_extent * 2.0)
        mesh = extractor.extract_mesh_bounded(
            voxel_size=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            depth_trunc=depth_trunc,
            mask_backgrond=self.mask_background,
        )
        mesh.compute_vertex_normals()
        return mesh

    @torch.no_grad()
    def rebuild(self, scene=None, gaussians=None, pipe=None, background=None):
        if scene is not None:
            self.scene = scene
        if gaussians is not None:
            self.gaussians = gaussians
        if pipe is not None:
            self.pipe = pipe
        if background is not None:
            self.background = background

        o3d = self._import_open3d()
        albedo_mesh = self._extract_attribute_mesh("albedo")
        roughness_mesh = self._extract_attribute_mesh("roughness")
        metallic_mesh = self._extract_attribute_mesh("metallic")

        triangles = torch.from_numpy(__import__("numpy").asarray(albedo_mesh.triangles)).long()
        vertices = __import__("numpy").asarray(albedo_mesh.vertices, dtype="float32")
        rough_vertices = __import__("numpy").asarray(roughness_mesh.vertices, dtype="float32")
        metal_vertices = __import__("numpy").asarray(metallic_mesh.vertices, dtype="float32")
        if vertices.shape != rough_vertices.shape or vertices.shape != metal_vertices.shape:
            raise RuntimeError("Open3D trace meshes produced incompatible topology across attributes.")

        if not __import__("numpy").allclose(vertices, rough_vertices, atol=1e-5) or not __import__("numpy").allclose(vertices, metal_vertices, atol=1e-5):
            raise RuntimeError("Open3D trace meshes drifted across attributes; topology-aligned attribute interpolation is not reliable.")

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(albedo_mesh)
        ray_scene = o3d.t.geometry.RaycastingScene()
        ray_scene.add_triangles(tmesh)

        self._ray_scene = ray_scene
        self._triangles = triangles.cpu().numpy()
        self._vertex_albedo = __import__("numpy").asarray(albedo_mesh.vertex_colors, dtype="float32")
        self._vertex_roughness = __import__("numpy").asarray(roughness_mesh.vertex_colors, dtype="float32")[:, :1]
        self._vertex_metallic = __import__("numpy").asarray(metallic_mesh.vertex_colors, dtype="float32")[:, :1]

    def _trace_hits_flat(self, rays_o: torch.Tensor, rays_d: torch.Tensor, camera_center: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self._ray_scene is None:
            self.rebuild()

        np = __import__("numpy")
        o3d = self._import_open3d()
        rays = np.concatenate([
            rays_o.detach().cpu().numpy().astype(np.float32),
            rays_d.detach().cpu().numpy().astype(np.float32),
        ], axis=-1)
        ans = self._ray_scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

        t_hit = ans["t_hit"].numpy()
        hit_mask_np = np.isfinite(t_hit)
        primitive_ids = ans["primitive_ids"].numpy()
        primitive_uvs = ans["primitive_uvs"].numpy()
        primitive_normals = ans["primitive_normals"].numpy()

        hit_points = torch.zeros_like(rays_o)
        hit_distance = torch.zeros(rays_o.shape[0], device=rays_o.device, dtype=rays_o.dtype)
        hit_normal = torch.zeros_like(rays_o)
        albedo = torch.zeros_like(rays_o)
        roughness = torch.zeros((rays_o.shape[0], 1), device=rays_o.device, dtype=rays_o.dtype)
        metallic = torch.zeros((rays_o.shape[0], 1), device=rays_o.device, dtype=rays_o.dtype)

        if hit_mask_np.any():
            tri = self._triangles[primitive_ids[hit_mask_np]]
            uv = primitive_uvs[hit_mask_np]
            bary = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=-1).astype(np.float32)

            albedo_np = (self._vertex_albedo[tri] * bary[..., None]).sum(axis=1)
            rough_np = (self._vertex_roughness[tri] * bary[..., None]).sum(axis=1)
            metal_np = (self._vertex_metallic[tri] * bary[..., None]).sum(axis=1)
            points_np = rays_o.detach().cpu().numpy()[hit_mask_np] + rays_d.detach().cpu().numpy()[hit_mask_np] * t_hit[hit_mask_np, None]
            normals_np = primitive_normals[hit_mask_np]

            hit_idx = torch.from_numpy(np.nonzero(hit_mask_np)[0]).to(device=rays_o.device)
            hit_points[hit_idx] = torch.from_numpy(points_np).to(device=rays_o.device, dtype=rays_o.dtype)
            hit_distance[hit_idx] = torch.from_numpy(t_hit[hit_mask_np]).to(device=rays_o.device, dtype=rays_o.dtype)
            hit_normal[hit_idx] = F.normalize(torch.from_numpy(normals_np).to(device=rays_o.device, dtype=rays_o.dtype), dim=-1, eps=1e-6)
            albedo[hit_idx] = torch.from_numpy(albedo_np).to(device=rays_o.device, dtype=rays_o.dtype)
            roughness[hit_idx] = torch.from_numpy(rough_np).to(device=rays_o.device, dtype=rays_o.dtype)
            metallic[hit_idx] = torch.from_numpy(metal_np).to(device=rays_o.device, dtype=rays_o.dtype)

        hit_mask = torch.from_numpy(hit_mask_np).to(device=rays_o.device)
        return {
            "hit_mask": hit_mask,
            "hit_distance": hit_distance,
            "hit_points": hit_points,
            "hit_normal": hit_normal,
            "albedo": torch.clamp(albedo, 0.0, 1.0),
            "roughness": torch.clamp(roughness, 0.0, 1.0),
            "metallic": torch.clamp(metallic, 0.0, 1.0),
        }



def build_trace_backend(config: TraceBackendConfig, scene, gaussians, pipe, background):
    requested = (config.backend or "auto").lower()
    adapter_error = None
    native_error = None
    open3d_error = None

    if requested in ("irgs_adapter_compat",):
        backend = IRGSAdapterCompatTraceBackend(
            gaussians=gaussians,
            alpha_min=config.native_alpha_min,
            transmittance_min=config.native_transmittance_min,
        )
        backend.rebuild(gaussians)
        print("[Stage2-Trace] Using IRGS adapter compatibility backend.")
        return backend

    if requested in ("auto", "irgs", "irgs_adapter"):
        try:
            backend = IRGSAdapterTraceBackend(
                gaussians=gaussians,
                alpha_min=config.native_alpha_min,
                transmittance_min=config.native_transmittance_min,
            )
            backend.rebuild(gaussians)
            print("[Stage2-Trace] Using IRGS 2D Gaussian trace adapter backend.")
            return backend
        except Exception as exc:
            adapter_error = exc
            if requested not in ("auto",):
                raise

    if requested in ("auto", "irgs_native"):
        try:
            backend = IRGSNativeGaussianTraceBackend(
                gaussians=gaussians,
                alpha_min=config.native_alpha_min,
                transmittance_min=config.native_transmittance_min,
            )
            backend.rebuild(gaussians)
            print("[Stage2-Trace] Using legacy IRGS native gaussian tracer backend.")
            if adapter_error is not None:
                print(f"[Stage2-Trace][Info] IRGS adapter backend unavailable: {adapter_error}")
            return backend
        except Exception as exc:
            native_error = exc
            if requested not in ("auto",):
                raise

    if requested in ("auto", "open3d", "open3d_mesh"):
        try:
            backend = Open3DMeshTraceBackend(
                scene=scene,
                gaussians=gaussians,
                pipe=pipe,
                background=background,
                voxel_size=config.open3d_voxel_size,
                sdf_trunc=config.open3d_sdf_trunc,
                depth_trunc=config.open3d_depth_trunc,
                mask_background=config.open3d_mask_background,
            )
            backend.rebuild(scene=scene, gaussians=gaussians, pipe=pipe, background=background)
            print("[Stage2-Trace] Using Open3D mesh raycasting fallback backend.")
            if adapter_error is not None:
                print(f"[Stage2-Trace][Info] IRGS adapter backend unavailable: {adapter_error}")
            if native_error is not None:
                print(f"[Stage2-Trace][Info] Native backend unavailable: {native_error}")
            return backend
        except Exception as exc:
            open3d_error = exc
            if requested not in ("auto",):
                raise

    raise RuntimeError(
        "No usable tracing backend is available. "
        f"adapter_error={adapter_error!r}, native_error={native_error!r}, open3d_error={open3d_error!r}"
    )



def trace_occlusion(trace_backend: BaseTraceBackend, ray_origins: torch.Tensor, ray_directions: torch.Tensor, camera_center: Optional[torch.Tensor] = None) -> torch.Tensor:
    return trace_backend.trace_occlusion(ray_origins=ray_origins, ray_directions=ray_directions, camera_center=camera_center)



def trace_incident_radiance(
    trace_backend: BaseTraceBackend,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    envmap,
    secondary_num_samples: int = 16,
    randomized_secondary: bool = False,
    camera_center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return trace_backend.trace_incident_radiance(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        envmap=envmap,
        secondary_num_samples=secondary_num_samples,
        randomized_secondary=randomized_secondary,
        camera_center=camera_center,
    )
