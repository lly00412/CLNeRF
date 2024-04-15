import torch
import numpy as np
import struct
import open3d as o3d
import os

MAX_BATCH_SIZE = 2048
def write_pointcloud(filename,xyz,rgb=None):

    """
    refer to: https://gist.github.com/Shreeyak/9a4948891541cb32b501d058db227fff
    creates a .pkl file of the point clouds generated
    """

    assert xyz.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb is None:
        rgb = np.ones(xyz.shape).astype(np.uint8)*255
    assert xyz.shape == rgb.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz[i,0],xyz[i,1],xyz[i,2],
                                        rgb[i,0].tobytes(),rgb[i,1].tobytes(),
                                        rgb[i,2].tobytes())))
    fid.close()
    print(f'Save points to {filename}!')

def extract_geometry_from_density_grid(model,resolution=100,density_threshold=0.5):
    scale = model.scale
    step = 2*scale/resolution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X,Y,Z = torch.meshgrid(torch.arange(-scale, scale, step),
                           torch.arange(-scale, scale, step),
                           torch.arange(-scale, scale,step))
    xyz_points = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),dim=1).contiguous()
    if xyz_points.shape[0]>MAX_BATCH_SIZE:
        pcd = torch.zeros(0,).to(device)
        for xyzs in torch.split(xyz_points,MAX_BATCH_SIZE):
            xyzs = xyzs.to(device)
            sigmas, _ = model.density(xyzs, return_feat=True)
            pcd = torch.cat((pcd,xyzs[sigmas>=density_threshold]))
    else:
        xyz_points = xyz_points.to(device)
        sigmas, _ = model.density(xyz_points, return_feat=True)
        pcd = xyz_points[sigmas>=density_threshold]
        pcd = pcd.detach().cpu().numpy()
    return pcd

def extract_geometry_from_depth_map(tgt_depth_map,c2w,K,depth_clip=None,rgbs=None,img_shape=None,device='cpu'):
    torch.cuda.empty_cache()

    depth_map = tgt_depth_map.clone().to(device)
    height, width = img_shape
    K_homo = torch.eye(4)
    K_homo[:3, :3] = K.clone().cpu()

    c2w_ = torch.eye(4)
    c2w_[:3] = c2w.clone().cpu()
    w2c = torch.inverse(c2w_)

    P_tgt = w2c.to(torch.float32)  # 4x4
    K_tgt = K_homo.clone().to(torch.float32)  # 4x4

    P_tgt = P_tgt.to(device)
    K_tgt = K_tgt.to(device)

    bwd_proj = torch.matmul(torch.inverse(P_tgt), torch.inverse(K_tgt)).to(torch.float32)
    bwd_rot = bwd_proj[:3, :3]
    bwd_trans = bwd_proj[:3, 3:4]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                           torch.arange(0, width, dtype=torch.float32)],
                          indexing='ij')
    y, x = y.contiguous(), x.contiguous()
    y, x = y.reshape(height * width), x.reshape(height * width)
    homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)

    # get world coords
    world_coords = torch.matmul(bwd_rot, homog)
    world_coords = world_coords * depth_map.reshape(1, -1)
    world_coords = world_coords + bwd_trans.reshape(3, 1)
    world_coords = torch.movedim(world_coords, 0, 1)
    # world_coords = world_coords.reshape(height, width, 3)

    if depth_clip is not None:
        world_coords = world_coords[depth_map<=depth_clip]
        if rgbs is not None:
            rgbs = rgbs.to(device)
            rgbs = rgbs[depth_map<=depth_clip]
            rgbs = rgbs.detach().cpu()

    if rgbs is not None:
        rgbs = (rgbs * 255).numpy().astype(np.uint8)

    return world_coords.detach().cpu().numpy(),rgbs

def create_pcd_from_numpy(xyz):
    if not isinstance(xyz,np.ndarray):
        raise ValueError('Input must be numpy array!')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def create_pcd_from_ply(ply_file):
    if not os.path.isfile(ply_file):
        raise ValueError('ply file does not exist!')
    pcd = o3d.io.read_point_cloud(ply_file)
    return pcd

def mark_points_on_surface(pred_pcd,gt_pcd,threshold):
    # create c
    dists = pred_pcd.compute_point_cloud_distance(gt_pcd)

