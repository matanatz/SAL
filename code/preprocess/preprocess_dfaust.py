from __future__ import print_function
import sys
import torch
sys.path.append('../../')
import argparse

import utils.general as utils
import trimesh
from trimesh.sample import sample_surface
import os
import numpy as np
import json

from scipy.spatial import cKDTree
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True, help="Path of unzipped d-faust scans.")
    parser.add_argument('--split', type=str, default='../confs/splits/dfaust/train_all_every5.json', help="Split file name.")
    parser.add_argument('--shapeindex', type=int,required=True, help="Shape index to be preprocessed. -1 for all")
    parser.add_argument('--skip', action="store_true",default=False)
    parser.add_argument('--sigma', type=float,default=0.2)


    opt = parser.parse_args()


    with open(opt.split, "r") as f:
        train_split = json.load(f)

    shapeindex = opt.shapeindex

    global_shape_index = 0
    for human,scans in train_split['scans'].items():
        for pose,shapes in scans.items():

            source = '{0}/{1}/{2}/{3}'.format(opt.datapath,'scans',human,pose)
            output = opt.datapath + '_processed'
            utils.mkdir_ifnotexists(output)
            utils.mkdir_ifnotexists(os.path.join(output,human))
            utils.mkdir_ifnotexists(os.path.join(output,human,pose))

            for shape in shapes:


                if (shapeindex == global_shape_index or shapeindex == -1):
                    print ("found!")
                    output_file = os.path.join(output,human,pose,shape)
                    print (output_file)
                    if (not opt.skip or  not os.path.isfile(output_file + '_dist_triangle.npy')):

                        print ('loading : {0}'.format(os.path.join(source,shape)))
                        mesh = trimesh.load(os.path.join(source,shape) + '.ply')
                        sample = sample_surface(mesh,250000)
                        center = np.mean(sample[0],axis=0)

                        pnts = sample[0]

                        pnts = pnts - np.expand_dims(center,axis=0)
                        scale = 1
                        pnts = pnts/scale
                        triangles = []
                        for tri in mesh.triangles:

                            a = Point_3((tri[0][0] - center[0])/scale, (tri[0][1]- center[1])/scale, (tri[0][2]- center[2])/scale)
                            b = Point_3((tri[1][0]- center[0])/scale, (tri[1][1]- center[1])/scale, (tri[1][2]- center[2])/scale)
                            c = Point_3((tri[2][0]- center[0])/scale, (tri[2][1]- center[1])/scale, (tri[2][2]- center[2])/scale)
                            triangles.append(Triangle_3(a, b, c))
                        tree = AABB_tree_Triangle_3_soup(triangles)

                        sigmas = []
                        ptree = cKDTree(pnts)
                        i = 0
                        for p in np.array_split(pnts,100,axis=0):
                            d = ptree.query(p,51)
                            sigmas.append(d[0][:,-1])

                            i = i+1

                        sigmas = np.concatenate(sigmas)
                        sigmas_big = opt.sigma * np.ones_like(sigmas)

                        np.save(output_file + '.npy', pnts)


                        sample = np.concatenate([pnts + np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                         pnts + np.expand_dims(sigmas_big,-1) * np.random.normal(0.0,1.0, size=pnts.shape)], axis=0)

                        dists = []
                        for np_query in sample:
                            cgal_query = Point_3(np_query[0].astype(np.double), np_query[1].astype(np.double), np_query[2].astype(np.double))

                            cp = tree.closest_point(cgal_query)
                            cp = np.array([cp.x(), cp.y(), cp.z()])
                            dist = np.sqrt(((cp - np_query)**2).sum(axis=0))

                            dists.append(dist)
                        dists = np.array(dists)
                        np.save(output_file + '_dist_triangle.npy',
                                np.concatenate([sample, np.expand_dims(dists, axis=-1)], axis=-1))

                        np.save(output_file + '_normalization.npy',
                                {"center":center,"scale":scale})

                global_shape_index = global_shape_index + 1

    print ("end!")