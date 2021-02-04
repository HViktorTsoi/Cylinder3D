# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from builder import model_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import cart2polar
from utils.load_save_util import load_checkpoint


class Cylinder3D:
    def __init__(self, config_path):
        self.devide = torch.device('cuda:0')

        self.configs = load_config_data(config_path)

        self.val_dataloader_config = self.configs['val_data_loader']

        self.val_batch_size = self.val_dataloader_config['batch_size']

        self.model_config = self.configs['model_params']

        self.model = model_builder.build(self.model_config)

        model_path = self.configs['train_params']['model_load_path']
        self.model = load_checkpoint(model_path, self.model)

        self.model.to(self.devide)
        self.model.eval()

    def preprocess_pointcloud(self, pc, model_config, dataset_config, ):

        # convert coordinate into polar coordinates
        xyz = pc[:, :3]
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if dataset_config['fixed_volume_space']:
            max_bound = np.asarray(dataset_config['max_volume_space'])
            min_bound = np.asarray(dataset_config['min_volume_space'])
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = np.array(model_config['output_shape'])
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # add voxel center feature
        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        # add intensity feature
        return_fea = np.concatenate((return_xyz, pc[:, 3][..., np.newaxis]), axis=1)

        return grid_ind, return_fea, pc[:, :3]

    def inference(self, point_cloud):
        """
        point_cloud: N x 4 array, [x, y, z, intensity]
        Note that , the range of intensity is [0, 1]
        """
        with torch.no_grad():
            val_grid, val_pt_fea, raw_coord = self.preprocess_pointcloud(
                pc=point_cloud, model_config=self.model_config, dataset_config=self.configs['dataset_params']
            )
            val_pt_fea_ten = torch.from_numpy(val_pt_fea).type(torch.FloatTensor).to(self.devide)
            val_grid_ten = torch.from_numpy(val_grid).to(self.devide)

            val_pt_fea_ten = [val_pt_fea_ten]
            val_grid_ten = [val_grid_ten]

            tic = time.time()
            predict_labels = self.model(val_pt_fea_ten, val_grid_ten, self.val_batch_size)
            toc = time.time()
            print('Time: {:.6f}s'.format(toc - tic))
            # aux_loss = loss_fun(aux_outputs, point_label_tensor)

            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()

            # get predicted label for each point
            predict_labels = predict_labels[0][[val_grid[:, 0], val_grid[:, 1], val_grid[:, 2]]]
            results = np.column_stack([raw_coord, predict_labels.reshape(-1, 1)])

        return results


def vis(pc, color_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('hot')(color_data / color_data.max())[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('gist_rainbow')(color_data / color_data.max())[:, :3])
    # pcd = pcd.voxel_down_sample(voxel_size=0.1)

    visualizer.add_geometry(pcd)
    opt = visualizer.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2


if __name__ == '__main__':

    segmentator = Cylinder3D(config_path='config/semantickitti_inference.yaml')

    # root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/02/velodyne.full'
    root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/KITTI/object/data_object_velodyne/training/velodyne/'
    # root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/KITTI/tracking/data_tracking_velodyne/testing/velodyne/0000'
    # root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/bottle_data/data_20210202/horizon/KITTI/'
    # root = '/home/hviktortsoi/map/KITTI.map'

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window('segmentation_result')
    for path in sorted(os.listdir(root)):
        # load data
        pc = np.fromfile(os.path.join(root, path), dtype=np.float32).reshape((-1, 4))
        pc[:, 3] /= 255

        # inference
        segment_results = segmentator.inference(pc)

        vis(segment_results[:, :3], segment_results[:, 3])
        visualizer.run()
        visualizer.clear_geometries()
    visualizer.destroy_window()
