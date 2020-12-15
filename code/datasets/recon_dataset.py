import torch.utils.data as data
from utils.general import *
import trimesh
from trimesh.sample import sample_surface
from scipy.spatial import cKDTree
from tqdm import tqdm
import utils.general as utils
import numpy as np
import torch


class ReconDataSetPrep(data.Dataset):

    def __init__(self,split,dataset_path,dist_file_name,normalize_data=False):

        model = trimesh.load(dataset_path)
        self.is_pc = type(model) == trimesh.PointCloud
        if not self.is_pc:
            model = utils.as_mesh(trimesh.load(dataset_path))
            if (model.faces.shape[0] > 0):
                self.points = sample_surface(model,250000)[0]
            else:
                self.points = model.vertices
        else:
            self.points = model.vertices

        if normalize_data:
            self.points = self.points - self.points.mean(0,keepdims=True)
            scale = np.abs(self.points).max()
            self.points = self.points / scale

        d = (self.points**2).sum(-1,keepdims=True) - 2*np.matmul(self.points,self.points.T) + (self.points**2).sum(-1,keepdims=True).T
        sigma_big = np.sqrt(d.max())
        sigmas = []

        ptree = cKDTree(self.points)

        for p in tqdm(np.array_split(self.points, 10, axis=0)):
            d = ptree.query(p, np.int(51.0))
            sigmas.append(d[0][:, -1])

        sigmas = np.concatenate(sigmas)
        sigmas = np.tile(sigmas, [250000 // sigmas.shape[0]])
        sigmas_big = sigma_big * np.ones_like(sigmas)
        pnts = np.tile(self.points, [250000 // self.points.shape[0], 1])

        sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                 pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)],
                                axis=0)

        dists = []

        for np_query in tqdm(sample):
            dist = ptree.query(np_query)[0]
            dists.append(dist)

        dists = np.array(dists)
        self.dists = np.concatenate([sample, np.expand_dims(dists, axis=-1)], axis=-1)
        self.npyfiles_mnfld = [dataset_path]

    def __getitem__(self, index):

        point_set_mnlfld = torch.from_numpy(self.points).float()
        sample_non_mnfld = torch.from_numpy(self.dists).float()
        random_idx = (torch.rand(128**2) * point_set_mnlfld.shape[0]).long()
        point_set_mnlfld = torch.index_select(point_set_mnlfld,0,random_idx)

        random_idx = (torch.rand(128**2) * sample_non_mnfld.shape[0]).long()
        sample_non_mnfld = torch.index_select(sample_non_mnfld,0,random_idx)

        return point_set_mnlfld,sample_non_mnfld,index


    def __len__(self):
        return 1


class ReconDataSet(data.Dataset):

    def __init__(self,split,dataset_path,dist_file_name,normalize_data=False):

        model = trimesh.load(dataset_path)
        self.is_pc = type(model) == trimesh.PointCloud
        if not self.is_pc:
            model = utils.as_mesh(trimesh.load(dataset_path))
            if (model.faces.shape[0] > 0):
                self.points = sample_surface(model,250000)[0]
            else:
                self.points = model.vertices
        else:
            self.points = model.vertices

        if normalize_data:
            self.points = self.points - self.points.mean(0,keepdims=True)
            scale = np.abs(self.points).max()
            self.points = self.points / scale

        d = (self.points**2).sum(-1,keepdims=True) - 2*np.matmul(self.points,self.points.T) + (self.points**2).sum(-1,keepdims=True).T
        self.sigma_big = np.sqrt(d.max()).item()
        sigmas = []

        ptree = cKDTree(self.points)

        for p in tqdm(np.array_split(self.points, 10, axis=0)):
            d = ptree.query(p, np.int(51.0))
            sigmas.append(d[0][:, -1])

        self.sigmas = torch.from_numpy(np.concatenate(sigmas)).float()
        self.npyfiles_mnfld = [dataset_path]
        self.points = torch.from_numpy(self.points).float()

    def __getitem__(self, index):

        sample_non_mnfld = torch.cat([self.points + self.sigmas.unsqueeze(-1) * torch.randn_like(self.points),
                   self.points.mean(dim=0,keepdim=True) + self.sigma_big*torch.randn_like(self.points)],
                  dim=0)
        sample_non_mnfld = torch.cat([sample_non_mnfld,torch.ones([sample_non_mnfld.shape[0],1])],dim=-1)
        point_set_mnlfld = self.points

        return point_set_mnlfld,sample_non_mnfld,index


    def __len__(self):
        return 1
