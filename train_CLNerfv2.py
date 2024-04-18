import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils.utils import slim_ckpt, load_ckpt
from utils.geo_utils import *

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        print("running exp on {}".format(self.hparams.exp_name))
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'task_number': self.hparams.task_number,
                  'task_curr': self.hparams.task_curr,
                  'task_split_method': self.hparams.task_split_method,
                  'rep_size': self.hparams.rep_size,
                  'rep_dir': f'results/lb/{self.hparams.dataset_name}/{self.hparams.exp_name}/rep',
                  'nerf_rep': self.hparams.nerf_rep}

        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        self.rep_dataset = dataset(split='rep', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        if not self.hparams.val_only:
            self.register_buffer('directions', self.train_dataset.directions.to(self.device))
            self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        else:
            # define additional parameters
            self.register_buffer('directions', self.test_dataset.directions.to(self.device))
            self.register_buffer('poses', self.test_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.rep_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)
        # TODO: comparing hamming distance of the bitmask for density grid across current and last step
        # TODO: comparing self.model.density_bitfield

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')
            # density_threshold = 0.01*MAX_SAMPLES/3**0.5

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/CLNerf/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)
        if self.hparams.save_density_pcd:
            self.pcd_dir = f'results/lb/{self.hparams.dataset_name}/{self.hparams.exp_name}/pcd'
            os.makedirs(self.pcd_dir, exist_ok=True)
        if self.hparams.save_depth_pcd:
            self.pcd_dir = f'results/lb/{self.hparams.dataset_name}/{self.hparams.exp_name}/pcd_clip_colmap/v{self.hparams.task_curr}'
            os.makedirs(self.pcd_dir, exist_ok=True)
        if self.hparams.mark_points_on_surface:
            os.makedirs(f'{self.pcd_dir}/on_surface2', exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if self.hparams.save_depth_pcd:
            idx = batch['img_idxs']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            img_w, img_h = self.test_dataset.img_wh
            xyzs,rgbs = extract_geometry_from_depth_map(tgt_depth_map=results['depth'].cpu(),
                                                        rgbs=results['rgb'].cpu(),
                                                  c2w=batch['pose'],
                                                  K=self.test_dataset.K,
                                                  img_shape=(img_h, img_w),
                                                  depth_clip=None,
                                                  device=device) # N*3, N*3, np.ndarrray

            # align_m = np.array([[1.138071816345332055e+00,-7.300598829132702583e-02,6.497527927834608752e-01,-2.323759808085589018e+00],
            #                     [-6.536992969315738033e-01,-9.985495936658358995e-02,1.133764632318774668e+00,1.657361383604702088e+00],
            #                     [-1.363067303782293342e-02,-1.306680962583968819e+00,-1.229434503706130222e-01,-7.163783983342566941e+00],
            #                     [0,0,0,1]])
            # xyzs = np.concatenate((xyzs,np.ones((xyzs.shape[0],1))),axis=-1)
            # xyzs = np.expand_dims(xyzs,axis=2)
            # align_m = np.expand_dims(align_m,axis=0)
            # xyzs = align_m.repeat(xyzs.shape[0],axis=0) @ xyzs

            pcd_file = f'{self.pcd_dir}/{idx:03d}.ply'
            # write_pointcloud(pcd_file, xyz=xyzs, rgb=rgbs)
            write_pointcloud(pcd_file, xyz=xyzs[:,:3], rgb=rgbs)

            if self.hparams.mark_points_on_surface:
                gt_pcd = create_pcd_from_ply(self.hparams.gt_pcd)
                pred_pcd = create_pcd_from_numpy(xyz=xyzs[:,:3],rgb=rgbs)
                # mask,xyzs_ = mark_points_on_surface(pred_pcd,gt_pcd,self.hparams.distance_threshold)
                mask, xyzs_, rgbs_ = mark_points_on_surface(pred_pcd, gt_pcd, threshold=self.hparams.distance_threshold)
                mark_pcd_file = f'{self.pcd_dir}/on_surface/{idx:03d}_on_surface.ply'
                write_pointcloud(mark_pcd_file, xyz=xyzs_, rgb=rgbs_)
                osr = mask.sum()/len(mask)
                logs['on_surface'] = osr
                frame_file = f'{self.pcd_dir}/on_surface/val_log.txt'
                #print(f'Among {len(mask)} points, {mask.sum()} points are on surface!')
                with open(frame_file,'a') as f:
                    f.write(f'frame {idx:03d}: valid points: {len(mask)} on_surface_rate: {osr:.4f}\n')
                    f.close()
                del mask,xyzs_,gt_pcd,pred_pcd
            # del xyzs,rgbs,align_m
            del xyzs, rgbs

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim, True)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips, True)
        if self.hparams.save_density_pcd:
            xyzs = extract_geometry_from_density_grid(self.model,
                                                      resolution=128,
                                                      density_threshold=0.05*MAX_SAMPLES/3**0.5)
            pcd_file = f'{self.pcd_dir}/epoch={hparams.num_epochs-1}-v{self.hparams.task_curr}.ply'
            write_pointcloud(pcd_file, xyz=xyzs, rgb=None)
            del xyzs
        if self.hparams.mark_points_on_surface:
            on_surface = np.stack([x['on_surface'] for x in outputs])
            mean_on_surface = on_surface.mean()
            std_on_surface = on_surface.std()
            self.log('test/on_surface_rate', mean_on_surface, True)
            print(f'On surface rate:{mean_on_surface}')
            txt_log = f'results/lb/{self.hparams.dataset_name}/{self.hparams.exp_name}/pcd_clip_colmap/on_surface_rate.txt'
            with open(txt_log,'a') as f:
                f.write(f'Task {self.hparams.task_curr} on surface rate: {mean_on_surface:.4f} \t std: {std_on_surface:.4f}\n')
                f.close()

    def on_test_start(self):
        torch.cuda.empty_cache()
        self.rep_dir = f'results/lb/{self.hparams.dataset_name}/{self.hparams.exp_name}/rep'
        os.makedirs(self.rep_dir, exist_ok=True)


    def test_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        fname = batch['fname']
        results = self(batch, split='test')
        
        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)

        rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        rgb_pred = (rgb_pred*255).astype(np.uint8)
        imageio.imsave(os.path.join(self.rep_dir, fname), rgb_pred)
        return None


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    # if hparams.val_only and (not hparams.ckpt_path):
    #     raise ValueError('You need to provide a @ckpt_path for validation!')
    if hparams.val_only and (not hparams.weight_path):
        raise ValueError('You need to provide a @weight_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/lb/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/lb/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    if (hparams.task_curr != hparams.task_number - 1) or hparams.val_only:
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs+1,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=DDPPlugin(find_unused_parameters=False)
                                if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=-1 if hparams.val_only else 0,
                        precision=16)
    else:  
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=DDPPlugin(find_unused_parameters=False)
                                if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=-1 if hparams.val_only else 0,
                        precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)
    # if not hparams.val_only:
    #     trainer.fit(system, ckpt_path=hparams.ckpt_path)
    # else:
    #     trainer.validate(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/lb/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/lb/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)

    if not hparams.val_only:
        if hparams.task_curr != (hparams.task_number -1):
            trainer.test(system)