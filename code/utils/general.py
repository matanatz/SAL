import os
import numpy as np
import torch
import trimesh
import logging
from scipy.spatial import cKDTree as KDTree

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

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

def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)

def get_item(list,idx):
    if (len(list) > 0):
        return list[idx]
    else:
        return None
def threshold_min_max(tensor, min_vec, max_vec):
    return torch.min(max_vec, torch.max(tensor, min_vec))

def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_cuda_ifavailable(torch_obj):
    if (torch.cuda.is_available()):
        return torch_obj.cuda()
    else:
        return torch_obj

def get_dist_matrix(a,b):
    x, y = a, b

    if (type(a) == torch.Tensor):
        x_square = (x ** 2).sum(dim=-1, keepdim=True)
        y_square = (y ** 2).sum(dim=-1).unsqueeze(0)
        zz = torch.mm(x, y.transpose(1, 0))
        P_mine = x_square + y_square - 2 * zz
    else:
        x_square = (x ** 2).sum(axis=-1,keepdims=True)
        y_square = np.expand_dims((y ** 2).sum(axis=-1),axis=0)
        zz = np.matmul(x, y.T)
        P_mine = x_square + y_square - 2 * zz
    return P_mine

def get_batch_dist_matrix(a, b):
    x, y = a, b
    x_square = (x ** 2).sum(dim=-1, keepdim=True)
    y_square = (y ** 2).sum(dim=-1).unsqueeze(1)
    zz = torch.bmm(x, y.transpose(2, 1))
    P_mine = x_square + y_square - 2 * zz

    return P_mine

def fps_2( points, B):

    r = np.sum(points * points, 1)
    r = np.expand_dims(r, axis=1)
    distance = r - 2 * np.matmul(points, np.transpose(points, [1, 0])) + np.transpose(r, [1, 0])

    def getGreedyPerm(D,B):
        """
        A Naive O(N^2) algorithm to do furthest points sampling

        Parameters
        ----------
        D : ndarray (N, N)
            An NxN distance matrix for points
        Return
        ------
        tuple (list, list)
            (permutation (N-length array of indices),
            lambdas (N-length array of insertion radii))
        """
        a = np.copy(D)
        np.fill_diagonal(a,np.inf)
        a = np.min(a,axis=0)
        print (a.mean())
        print(a.max())
        idx = (a < 0.0015).squeeze()
        D = D[idx][:,idx]
        N = D.shape[0]
        # By default, takes the first point in the list to be the
        # first point in the permutation, but could be random
        perm = np.zeros(B, dtype=np.int64)
        lambdas = np.zeros(B)
        perm[0] = np.random.choice(np.arange(D.shape[0]),1).item()
        ds = D[perm[0], :]
        for i in range(1, B):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            ds = np.minimum(ds, D[idx, :])
        return (perm, lambdas)

    idx,_ = getGreedyPerm(distance.squeeze(),B)
    return idx


def load_srb_range_scan(file_name):
    """
    Load a range scan point cloud from the Surface Reconstruction Benchmark dataset
    :param file_name: The file containing the point cloud
    :return: A pair (v, f) of vertices and normals both with shape [n, 3]
    """
    v = []
    n = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            x, y, z, nx, ny, nz = [float(p) for p in line.split()]
            v.append((x, y, z))
            n.append((nx, ny, nz))
    return np.array(v), np.array(n)

def load_point_cloud_by_file_extension(file_name, compute_normals=False):
    import point_cloud_utils as pcu
    if file_name.endswith(".obj"):
        v, f, n = pcu.read_obj(file_name, dtype=np.float32)
    elif file_name.endswith(".off"):
        v, f, n = pcu.read_off(file_name, dtype=np.float32)
    elif file_name.endswith(".ply"):
        v, f, n, _ = pcu.read_ply(file_name, dtype=np.float32)
    elif file_name.endswith(".npts"):
        v, n = load_srb_range_scan(file_name)
        f = []
    else:
        raise ValueError("Invalid file extension must be one of .obj, .off, .ply, or .npts")

    if compute_normals and f.shape[0] > 0:
        n = pcu.per_vertex_normals(v, f)
    return v, n

def configure_logging(debug,quiet,logfile):
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("SAL - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000,one_side=False):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    if (one_side):
        return gen_to_gt_chamfer
    else:
        return gt_to_gen_chamfer + gen_to_gt_chamfer
