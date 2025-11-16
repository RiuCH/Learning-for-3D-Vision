import torch
from pytorch3d.ops.knn import knn_points
import torch.nn.functional as F
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	loss = F.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	# point_cloud_src = point_cloud_src.unsqueeze(1) # (b, 1, n, 3)
	# point_cloud_tgt = point_cloud_tgt.unsqueeze(2) # (b, n, 1, 3)

	# pc_dist = torch.norm(point_cloud_src - point_cloud_tgt, dim=3) # (b, n, n)
	# loss_chamfer = torch.mean(torch.min(pc_dist, dim=1)[0]) + torch.mean(torch.min(pc_dist, dim=2)[0]) # torch min return val, idx
 
	dists1, _, _ = knn_points(point_cloud_src, point_cloud_tgt)
	dists2, _, _ = knn_points(point_cloud_tgt, point_cloud_src)
	loss_chamfer = torch.sum(dists1 + dists2)
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian