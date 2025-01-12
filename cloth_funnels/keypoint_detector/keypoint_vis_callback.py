import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import Subset, DataLoader
import wandb
import numpy as np
import cv2
from matplotlib import cm

def to_numpy(x):
    return x.detach().to('cpu').numpy()

def dict_to(x, device):
    new_x = dict()
    for key, value in x.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        new_x[key] = value
    return new_x

class KeypointCallback(pl.Callback):
    def __init__(self, 
            val_dataset, 
            num_samples=4, seed=0):
        super().__init__()
        rs = np.random.RandomState(seed=seed)
        vis_idxs = rs.choice(len(val_dataset), num_samples)
        vis_subset = Subset(val_dataset, vis_idxs)
        vis_dataloader = DataLoader(
            vis_subset, batch_size=num_samples)
        vis_batch = next(iter(vis_dataloader))
        self.vis_batch = vis_batch
        self.vis_idxs = vis_idxs
        
    def on_validation_epoch_end(self, 
            trainer: pl.Trainer, 
            pl_module: pl.LightningModule) -> None:
        vis_batch_device = dict_to(
            self.vis_batch,
            device=pl_module.device)

        batch = vis_batch_device
        input_img = batch['input_image']
        # target_scoremap = batch['target_scoremap'] # (N,3,H,W)
        target_keypoints = batch['target_keypoint'] # (N,2) float32

        result = pl_module.forward(input_img)

        scoremaps = result['scoremap']
        scoremap_probs = torch.sigmoid(scoremaps)

        imgs = list()
        # compute keypoint distance
        for j in range(scoremaps.shape[1]):

            scoremap_prob = scoremap_probs[:, j]
            scoremap = scoremaps[:, j]
            target_keypoint = target_keypoints[:, j]

            # compute keypoint distance
            target_keypoint_np = to_numpy(target_keypoint) # (N, K, 2)

            pred_idx_np = to_numpy(torch.argmax(
                scoremap.reshape(scoremap.shape[0], -1), 
                dim=-1, keepdim=False)) #(4, HxW)

            pred_keypoint_np = np.stack(np.unravel_index(
                pred_idx_np, shape=scoremap.shape[1:])).T.astype(np.float32)[:,::-1]
        
            # two images
            def draw_keypoint(img, pred_keypoint, target_keypoint):
                cv2.drawMarker(img, pred_keypoint, 
                        color=(255,0,0), markerType=cv2.MARKER_CROSS,
                        markerSize=20, thickness=1)
                cv2.drawMarker(img, target_keypoint, 
                        color=(0,0,255), markerType=cv2.MARKER_CROSS,
                        markerSize=20, thickness=1)

            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            for i in range(len(self.vis_idxs)):
                vis_idx = self.vis_idxs[i]
                input_rgb_img = np.moveaxis(to_numpy(input_img[i]), 0, -1)
                input_rgb_img = np.ascontiguousarray((input_rgb_img * std + mean) * 255).astype(np.uint8)
                # print("Scoremap vis shape", scoremap_prob[i,0].shape)
                scoremap_vis = np.ascontiguousarray(cm.get_cmap('jet')(to_numpy(
                    scoremap_prob[i]))[:,:,:3] * 255).astype(np.uint8)
                
                pred_p = np.round(pred_keypoint_np[i]).astype(np.int32)
                target_p = np.round(target_keypoint_np[i]).astype(np.int32)
                draw_keypoint(input_rgb_img, pred_p, target_p)
                draw_keypoint(scoremap_vis, pred_p, target_p)

                img_pair = np.concatenate([input_rgb_img, scoremap_vis], axis=0)
                imgs.append(wandb.Image(
                    img_pair, caption=f"val-k{i}-{vis_idx}"
                ))

        trainer.logger.experiment.log({
            "val/vis": imgs,
            "global_step": trainer.global_step
        })