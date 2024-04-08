import torch
import numpy as np
import struct

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
    print(f'Save points to {fid}!')

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