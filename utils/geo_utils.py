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
    else:
        rgb = (rgb * 255).astype(np.uint8)
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

    # if rgbs is not None:
    #     rgbs = (rgbs * 255).numpy().astype(np.uint8)

    return world_coords.detach().cpu().numpy(),rgbs.numpy()

def create_pcd_from_numpy(xyz,rgb=None):
    if not isinstance(xyz,np.ndarray):
        raise ValueError('Input must be numpy array!')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def create_pcd_from_ply(ply_file):
    if not os.path.isfile(ply_file):
        raise ValueError('ply file does not exist!')
    pcd = o3d.io.read_point_cloud(ply_file)
    return pcd

def mark_points_on_surface(pred_pcd,gt_pcd,threshold):
    aabb = gt_pcd.get_oriented_bounding_box()
    pred_pcd = pred_pcd.crop(aabb)
    dists = pred_pcd.compute_point_cloud_distance(gt_pcd)
    dists = np.asarray(dists)
    xyzs_ = np.asarray(pred_pcd.points)
    rgbs_ = np.asarray(pred_pcd.colors)
    mask = (dists<=threshold)
    return mask,xyzs_[mask],rgbs_[mask]

def warp_tgt_to_ref(tgt_depth, ref_c2w, tgt_c2w, K, pixl_ids=None, img_shape=None,device='cpu'):
    depth_map = tgt_depth.clone()
    height, width = img_shape

    K_homo = torch.eye(4)
    K_homo[:3,:3] = K.clone().cpu()

    rc2w = torch.eye(4)
    rc2w[:3] = ref_c2w.clone().cpu()
    rw2c = torch.inverse(rc2w)

    tc2w = torch.eye(4)
    tc2w[:3] = tgt_c2w.clone().cpu()
    tw2c = torch.inverse(tc2w)

    torch.cuda.empty_cache()
    # warp tgt depth map to ref view
    # grab intrinsics and extrinsics from reference view
    P_ref = rw2c.to(torch.float32) # 4x4
    K_ref = K_homo.clone().to(torch.float32) # 4x4

    P_ref = P_ref.to(device)
    K_ref = K_ref.to(device)

    R_ref = P_ref[:3, :3] # 3x3
    t_ref = P_ref[:3, 3:4] # 3x1

    C_ref = torch.matmul(-R_ref.transpose(0, 1), t_ref)
    z_ref = R_ref[2:3, :3].reshape(1, 1, 1, 3).repeat(height, width, 1, 1)
    C_ref = C_ref.reshape(1, 1, 3).repeat(height, width, 1)

    depth_map = depth_map.to(device)  # h,w

    # get intrinsics and extrinsics from target view
    P_tgt = tw2c.to(torch.float32)  #  4x4
    K_tgt = K_homo.clone().to(torch.float32)  #  4x4

    P_tgt = P_tgt.to(device)
    K_tgt = K_tgt.to(device)

    bwd_proj = torch.matmul(torch.inverse(P_tgt), torch.inverse(K_tgt)).to(torch.float32)
    fwd_proj = torch.matmul(K_ref, P_ref).to(torch.float32)
    bwd_rot = bwd_proj[:3, :3]
    bwd_trans = bwd_proj[:3, 3:4]
    proj = torch.matmul(fwd_proj, bwd_proj)
    rot = proj[:3, :3]
    trans = proj[:3, 3:4]

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
    world_coords = world_coords.reshape(height, width, 3)

    # get pixel projection
    rot_coords = torch.matmul(rot, homog)
    proj_3d = rot_coords * depth_map.reshape(1, -1)
    proj_3d = proj_3d + trans.reshape(3, 1)
    proj_2d = proj_3d[:2, :] / proj_3d[2:3, :]
    proj_2d = (torch.movedim(proj_2d, 0, 1)).to(torch.long)
    proj_2d = torch.flip(proj_2d, dims=(1,))

    # compute projected depth
    proj_depth = torch.sub(world_coords, C_ref).unsqueeze(-1)
    proj_depth = torch.matmul(z_ref, proj_depth).reshape(height, width)
    proj_depth = proj_depth.reshape(-1, 1)

    # mask out invalid indices
    mask = torch.where(proj_2d[:, 0] < height, 1, 0) * \
           torch.where(proj_2d[:, 0] >= 0, 1, 0) * \
           torch.where(proj_2d[:, 1] < width, 1, 0) * \
           torch.where(proj_2d[:, 1] >= 0, 1, 0)
    inds = torch.where(mask)[0]
    proj_2d = torch.index_select(proj_2d, dim=0, index=inds)
    proj_2d = (proj_2d[:, 0] * width) + proj_2d[:, 1]
    proj_depth = torch.index_select(proj_depth, dim=0, index=inds).squeeze()

    proj_depth, indices = torch.sort(proj_depth, 0) # ascending oreder
    proj_2d = proj_2d[indices]
    proj_depth = proj_depth.flip(0)
    proj_2d = proj_2d.flip(0)

    warped_depth = torch.zeros(height*width).to(proj_depth)
    warped_depth[proj_2d] = proj_depth
    warped_depth = warped_depth.reshape(height, width)

    del proj_depth

    return warped_depth, proj_2d

class GetVirtualCam:
    def __init__(self, kwargs):
        super(GetVirtualCam, self).__init__()
        self.ref_c2w = kwargs['ref_c2w'] # 3x4
        self.K = kwargs['K'] # 3x3
        self.ref_depth_map = kwargs['ref_depth_map']
        self.device = kwargs['device']
        self.pixl_ids = kwargs['pix_ids']
        self.img_h = kwargs['img_h']
        self.img_w = kwargs['img_w']
        self.dense = kwargs['dense_map']
        self.opacity = kwargs['opacity']
        if self.dense:
            self.ref_depth_map = self.ref_depth_map.reshape(self.img_h,self.img_w)

        self.scene_center = self.get_scene_center()

    def get_scene_center(self):
        if not self.dense:
            return self.get_scene_center_sparse()
        else:
            return self.get_scene_center_dense()

    def get_scene_center_dense(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        height, width = self.ref_depth_map.shape

        ref_c2w = torch.eye(4)
        ref_c2w[:3] = self.ref_c2w.clone().cpu()
        ref_c2w = ref_c2w.to(device=self.device, dtype=torch.float32)
        ref_w2c = torch.inverse(ref_c2w)

        K = torch.eye(4)
        K[:3, :3] = self.K.clone().cpu()
        K = K.to(ref_w2c)

        bwd_proj = torch.matmul(ref_c2w, torch.inverse(K)).to(torch.float32)
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
        world_coords = torch.movedim(world_coords, 0, 1) # (h w) 3

        world_coords_real = world_coords[self.opacity>0]
        scene_center = world_coords_real.mean(0)

        return scene_center.cpu()

    def get_scene_center_sparse(self):
        depth_map = self.ref_depth_map.clone().to(self.device)
        if len(depth_map) > len(self.pixl_ids):
            depth_map = depth_map[self.pixl_ids]
        height, width = self.img_h, self.img_w

        ref_c2w = torch.eye(4)
        ref_c2w[:3] = self.ref_c2w.clone().cpu()
        ref_c2w = ref_c2w.to(device=self.device, dtype=torch.float32)
        ref_w2c = torch.inverse(ref_c2w)

        K = torch.eye(4)
        K[:3, :3] = self.K.clone().cpu()
        K = K.to(ref_w2c)

        bwd_proj = torch.matmul(ref_c2w, torch.inverse(K)).to(torch.float32)
        bwd_rot = bwd_proj[:3, :3]
        bwd_trans = bwd_proj[:3, 3:4]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                               torch.arange(0, width, dtype=torch.float32)],
                              indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height * width), x.reshape(height * width)
        homog = torch.stack((x, y, torch.ones_like(x))).to(bwd_rot)
        homog = homog[...,self.pixl_ids]

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords * depth_map.reshape(1, -1)
        world_coords = world_coords + bwd_trans.reshape(3, 1)
        world_coords = torch.movedim(world_coords, 0, 1) # (n_rays) 3

        world_coords_real = world_coords[self.opacity > 0]
        scene_center = world_coords_real.mean(0)

        return scene_center.cpu()

    def get_near_c2w(self, c2w, theta=5, axis='x'):
        cam_center = c2w[:3, 3:4].clone().to(self.scene_center)
        cam_center = cam_center.squeeze()
        trans_c2s = self.get_translation_matrix(cam_center,self.scene_center)
        rot = self.get_rotation_matrix(theta, axis)

        c2w_homo = torch.eye(4)
        c2w_homo[:3] = c2w.clone().cpu()
        c2w_homo = c2w_homo.to(torch.float32)
        w2c = torch.inverse(c2w_homo)

        w2c = torch.mm(trans_c2s,w2c)
        w2c = torch.mm(rot,w2c)
        w2c = torch.mm(torch.inverse(trans_c2s),w2c)

        new_c2w = torch.inverse(w2c)
        return new_c2w[:3]

    def get_rotation_matrix(self, theta=5, axis='x'): # rot theta degree across x axis
        phi = (theta * (np.pi / 180.))
        rot = torch.eye(4)
        if axis=='x':
            rot[:3,:3] = torch.Tensor([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
            ])
        elif axis == 'y':
            rot[:3,:3] = torch.Tensor([
                [np.cos(phi), 0, -np.sin(phi)],
                [0, 1, 0],
                [np.sin(phi), 0, np.cos(phi)]
            ])
        elif axis=='z':
            rot[:3,:3] = torch.Tensor([
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ])
        return rot

    def get_translation_matrix(self,origin,destination): # both should be (x,y,z)
        trans = torch.eye(4).to(destination)
        trans[:3,3] = destination-origin
        return trans