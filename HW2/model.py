from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import torch.nn.functional as F

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder =  VoxelDecoder()
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            self.decoder = PointDecoder(self.n_point)            
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.decoder = MeshDecoder(mesh_pred.verts_packed().shape[0])        
        
        elif args.type == "implicit":
            self.decoder = ImplicitDecoder(args)    
        
        elif args.type == "parametric":
            self.n_point = args.n_points
            self.decoder = ParametricDecoder(self.n_point) 

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            voxels_pred = self.decoder(encoded_feat)        
            return voxels_pred
        
        elif args.type == "point":
            pointclouds_pred = self.decoder(encoded_feat)   
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat)         
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred        
        
        elif args.type == "implicit":
            occ_pred = self.decoder(encoded_feat)
            return occ_pred
        
        elif args.type == "parametric":
            pointclouds_pred = self.decoder(encoded_feat, args.n_points)
            return pointclouds_pred

    def interpret(self, images, args):
        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images
        if args.type == "vox":
            xs = self.decoder.interpret(encoded_feat)
            return xs
        
class VoxelDecoder(nn.Module):
    def __init__(self):
        super(VoxelDecoder, self).__init__()
        self.fc_layers = nn.Sequential( nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                        )

        self.decoder = nn.Sequential( nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), 
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(),
                                        nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1), 
                                        nn.BatchNorm3d(16),
                                        nn.ReLU(),
                                        nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1), 
                                        nn.BatchNorm3d(8),
                                        nn.ReLU(),
                                        nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1), 
                                        nn.BatchNorm3d(4),
                                        nn.ReLU(),
                                        nn.ConvTranspose3d(4, 1, kernel_size=1, stride=1, padding=0), 
                                        nn.BatchNorm3d(1))
        
    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(-1, 64, 2, 2, 2)
        x = self.decoder(x)
        return x
    
    @torch.no_grad()
    def interpret(self, x):
        xs = []
        x = self.fc_layers(x)
        x = x.view(-1, 64, 2, 2, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            if (i- 2) % 3 == 0:
                avg_map = x.mean(dim=1, keepdim=True)
                # avg_map = x[0]
                vmin, vmax = avg_map.min(), avg_map.max()
                avg_map = (avg_map - vmin) / (vmax - vmin)
                xs.append(avg_map)
        xs.append(x)
        return xs

class PointDecoder(nn.Module):
    def __init__(self, n_points):
        super(PointDecoder, self).__init__()
        self.n_points = n_points
        self.decoder = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.Linear(512, n_points*3), nn.Tanh())
    
    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, self.n_points, 3)
        return x

                                        
class MeshDecoder(nn.Module):
    def __init__(self, n_verts):
        super(MeshDecoder, self).__init__()
        self.n_verts = n_verts
        self.decoder = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.Linear(512, n_verts*3), nn.Tanh())
        
    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, self.n_verts, 3)
        return x

class ImplicitDecoder(nn.Module):
    def __init__(self, args): 
        super().__init__()
        
        self.decoder = nn.Sequential(nn.Linear(515, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 1), nn.Tanh())
        
        xx, yy, zz = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
        grid_points = torch.stack([xx, yy, zz], dim=-1)  
        grid_points = grid_points.view(-1, 3)
        self.register_buffer('grid_points', grid_points)
        
    def forward(self, x):
        grid_points = self.grid_points.unsqueeze(0).expand(x.shape[0], self.grid_points.shape[0], 3) # (B, 32**3, 3)
        x = x.unsqueeze(1).expand(-1, grid_points.shape[1], -1) # (B, 32**3, 512)  
        x = torch.cat([x, grid_points], dim=2) # (B, 32**3, 515)
        x = x.reshape(-1, x.shape[-1]) # (B*32**3, 515)
        x = self.decoder(x)      
        x = x.view(-1, 1, 32, 32, 32)
        return x
        
class ParametricDecoder(nn.Module):
    def __init__(self, n_points):
        super(ParametricDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(514, 512), nn.ReLU(),
                                    nn.Linear(512, 512), nn.ReLU(),
                                    nn.Linear(512, 3), nn.Tanh())
        self.n_points = n_points
        
    def forward(self, x, n_points=None):
        n_points = self.n_points if n_points is None else n_points
        bs = x.shape[0]
        points_2d = (torch.rand(bs, n_points, 2,) * 2) - 1
        points_2d = points_2d.to(x.device)
        x = x.unsqueeze(1).expand(bs, n_points, -1)
        x = torch.cat([x, points_2d], dim=2)
        x = x.reshape(-1, x.shape[-1])
        x = self.decoder(x)
        x = x.view(-1, n_points, 3)
        return x
        
        
            
            
            
     