import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, steps=self.n_pts_per_ray).cuda()

        # TODO (Q1.4): Sample points from z values
        directions = ray_bundle.directions.unsqueeze(1).expand(-1, len(z_vals), -1)
        origins = ray_bundle.origins.unsqueeze(1).expand(-1, len(z_vals), -1)
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1)
  
        sample_points = directions * z_vals + origins

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}


def sample_pdf_rays(z_vals_coarse, weights, ray_bundle , N_samples):

    # Get midpoints between coarse z-values
    z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])

    # Create bin edges by concatenating start, midpoints, and end
    bins = torch.cat(
        [z_vals_coarse[..., :1], z_vals_mid, z_vals_coarse[..., -1:]], -1
    )

    # Get pdf
    pdf = weights + 1e-5  # prevent nans
    pdf = pdf / torch.sum(pdf, -1, keepdim=True) # [n_rays, n_pts_coarse]

    # Get cdf
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [n_rays, n_pts_coarse + 1]

    u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)  # (n_rays, N_samples, 2)

    # Gather bin edges and cdf values
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)

    # Perform linear interpolation
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_vals_fine = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0]) # [n_rays, n_pts_fine]

    # Combine and sort all samples
    z_vals_all, _ = torch.sort(
        torch.cat([z_vals_coarse, z_vals_fine], -1), -1
    )
 
    samples = z_vals_all.unsqueeze(-1)
    
    directions = ray_bundle.directions.unsqueeze(1).expand(-1, samples.shape[1], -1)
    origins = ray_bundle.origins.unsqueeze(1).expand(-1, samples.shape[1], -1)
    
    sample_points = directions * samples + origins

    return ray_bundle._replace(
        sample_points=sample_points,
        sample_lengths=samples,
    )