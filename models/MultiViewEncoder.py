
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from models.decoder import load_decoder

def backproject(voxel_dim, voxel_size, origin, projection, features):
    """
    Backprojects 2D features into a 3D voxel grid along camera rays.

    Implements equations 1 and 2 from https://arxiv.org/pdf/2003.10432.pdf.
    Each image pixel defines a 3D ray; its feature is copied to all voxels along that ray.

    Args:
        voxel_dim: (nx, ny, nz) dimensions of the output voxel grid
        voxel_size: physical size of each voxel (e.g., 0.04m)
        origin: 3D coordinates of voxel (0,0,0)
        projection: bx4x3 projection matrices (intrinsics @ extrinsics)
        features: b x c x h x w tensor of 2D features to backproject

    Returns:
        volume: b x c x nx x ny x nz 3D feature grid
        valid:  b x 1 x nx x ny x nz mask indicating voxels within view
    """

    batch = features.size(0)
    channels = features.size(1)
    device = features.device
    nx, ny, nz = voxel_dim

    origin = torch.Tensor([[-nx*voxel_size/2, -ny*voxel_size/2, -nz*voxel_size/2]])
    coords = coordinates(voxel_dim, device).unsqueeze(0).expand(batch,-1,-1) # bx3xhwd
    world = coords.type_as(projection) * voxel_size + origin.to(device).unsqueeze(2)
    world = torch.cat((world, torch.ones_like(world[:,:1]) ), dim=1)

    camera = torch.bmm(projection, world)
    px = (camera[:,0,:]/camera[:,2,:]).round().type(torch.long)
    py = (camera[:,1,:]/camera[:,2,:]).round().type(torch.long)
    pz = camera[:,2,:]

    height, width = features.size()[2:]
    valid = (px >= 0) & (py >= 0) & (px < width) & (py < height) & (pz>0) # bxhwd

    volume = torch.zeros(batch, channels, nx*ny*nz, dtype=features.dtype,  device=device)
    
    for b in range(batch):
        volume[b,:,valid[b]] = features[b,:,py[b,valid[b]], px[b,valid[b]]]

    volume = volume.view(batch, channels, nx, ny, nz)
    valid = valid.view(batch, 1, nx, ny, nz)

    return volume, valid


def coordinates(voxel_dim, device=torch.device('cuda')):
    """
    Creates a 3D meshgrid for a given volume size.

    Args:
        voxel_dim: (nx, ny, nz) dimensions of the voxel grid

    Returns:
        torch.LongTensor of shape (3, nx * ny * nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


class MultiViewEncoder(nn.Module):

    def __init__(self, config):

        super(MultiViewEncoder, self).__init__()
        self.feat_encoder = torch.hub.load('facebookresearch/dinov2', config.model.oddmatcher.dinov2_model_name) # Trying to use Dinov2 features directly... with no distillation
        
        for p in self.feat_encoder.parameters():
            p.requires_grad = False
            
        self.reduce_channels = nn.Conv2d(in_channels=384, out_channels=config.model.oddmatcher.input_channel_dim, kernel_size=1) # reduces channels from 384 to 32 
        self.decoder = load_decoder(config)
        self.voxel_dim_train = config.model.backbone.voxel_dim
        self.voxel_size = config.model.backbone.voxel_size
        self.origin = torch.tensor([0,0,0]).view(1,3)
        self.backbone2d_stride = config.model.backbone.backbone2d_stride

    def initialize_volume(self):
        """
        Resets accumulators.

        'self.volume' stores accumulated voxel features.  
        'self.valid' counts how many times each voxel is seen in the camera frustum.
        """

        self.volume = 0
        self.valid = 0


    def inference(self, projection, feature):
        """
        Backprojects 2D features into 3D and accumulates them.

        Args:
            projection matrix: b x 3 x 4  
            RGB image: b x 3 x h x w
            feature map: b x c x h' x w' (with h', w' = downsampled size)

        Accumulated output is stored in `self.volume` and `self.valid`.
        """

        projection = projection.clone()
        projection[:,:2,:] = projection[:,:2,:] / self.backbone2d_stride

        voxel_dim = self.voxel_dim_train

        with torch.no_grad():
            volume, valid = backproject(voxel_dim, self.voxel_size, self.origin, projection, feature)
        
        self.volume = self.volume + volume
        self.valid = self.valid + valid

        

    def forward(self, rgb_images, projection):
        
        self.initialize_volume()

        bs, num_views = rgb_images.shape[:2]

        stacked_rgb = rearrange(rgb_images, 'b n c fh fw -> (b n) c fh fw')
        rgb_images_resized = F.interpolate(stacked_rgb, size=(252, 252), mode='bilinear', align_corners=False)
        vit_feats = self.feat_encoder.forward_features(rgb_images_resized) # Dino features exploited directly

        vit_feats = vit_feats['x_norm_patchtokens']
        fh = fw = int(math.sqrt(vit_feats.shape[1]))
        vit_feats = rearrange(vit_feats, 'b (fh fw) d -> b d fh fw', fh=fh, fw=fw )
        vit_feats = F.interpolate(vit_feats, size=(64, 64), mode='bilinear', align_corners=False)
        vit_feats = self.reduce_channels(vit_feats)
        vit_feats =  rearrange(vit_feats, '(b n) c fh fw -> b n c fh fw', b = bs)
        
        for view_idx in range(num_views):
            self.inference(projection[:,view_idx], vit_feats[:,view_idx])

        volume = self.volume/self.valid
        volume = volume.transpose(0,1)
        volume[:,self.valid.squeeze(1)==0]=0
        volume = volume.transpose(0,1)

        [_, _, volume_feats] = self.decoder(volume.permute(0,1,4,3,2))

        return volume_feats
    