import argparse
import sys
sys.path.append('../code')
import utils.general as utils
import os
import json
import trimesh
import utils.general as utils
import logging
from datasets.dfaust_dataset import DFaustDataSet
from datasets.recon_dataset import ReconDataSet
import torch
from pyhocon import ConfigFactory
import utils.plots as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import os
import GPUtil

def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def optimize_latent(latent, ds, itemindex, decoder, path, epoch,resolution,conf):
    latent.detach_()
    latent.requires_grad_()
    lr = 1.0e-3
    optimizer = torch.optim.Adam([latent], lr=lr)
    loss_func = utils.get_class(conf.get_string('network.loss.loss_type'))(
        **conf.get_config('network.loss.properties'))

    num_iterations = 800

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    for e in range(num_iterations):
        input_pc,sample_nonmnfld,_ = ds[itemindex]
        input_pc = utils.get_cuda_ifavailable(input_pc).unsqueeze(0)
        sample_nonmnfld = utils.get_cuda_ifavailable(sample_nonmnfld).unsqueeze(0)

        non_mnfld_pnts = sample_nonmnfld[:,:, :3]
        dist_nonmnfld = sample_nonmnfld[:,:, 3].reshape(-1)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()
        non_mnfld_pnts_with_latent = torch.cat(
            [latent.unsqueeze(1).repeat(1, non_mnfld_pnts.shape[1], 1), non_mnfld_pnts], dim=-1)
        nonmanifold_pnts_pred = decoder(
            non_mnfld_pnts_with_latent.view(-1, non_mnfld_pnts_with_latent.shape[-1]))

        loss_res = loss_func(manifold_pnts_pred=None,
                             nonmanifold_pnts_pred=nonmanifold_pnts_pred,
                             nonmanifold_gt=dist_nonmnfld,
                             weight=None)
        loss = loss_res["loss"]

        loss.backward()
        optimizer.step()
        print("iteration : {0} , loss {1}".format(e, loss.item()))
        print("mean {0} , std {1}".format(latent.mean().item(), latent.std().item()))

    with torch.no_grad():
        reconstruction = plt.plot_surface(with_points=False,
                                          points=
                                          torch.cat([latent.unsqueeze(1).repeat(1, input_pc.shape[1], 1), input_pc],
                                                       dim=-1)[0],
                                          decoder=network.decoder,
                                          latent=latent,
                                          path=path,
                                          epoch=epoch,
                                          in_epoch=ds.npyfiles_mnfld[itemindex].split('/')[-3] + '_'  + ds.npyfiles_mnfld[itemindex].split('/')[-1].split('.npy')[0] + '_after',
                                          shapefile=ds.npyfiles_mnfld[itemindex],
                                          resolution=resolution,
                                          mc_value=0,
                                          is_uniform_grid=True,
                                          verbose=True,
                                          save_html=False,
                                          save_ply=True,
                                          overwrite=True)
        return  reconstruction

def evaluate(network,exps_dir,experiment_name,timestamp, split_filename, epoch, conf, with_opt,resolution,compute_dist_to_gt):

    utils.mkdir_ifnotexists(os.path.join('../', exps_dir, experiment_name, timestamp, 'evaluation'))
    utils.mkdir_ifnotexists(os.path.join('../', exps_dir, experiment_name, timestamp, 'evaluation',split_filename.split('/')[-1].split('.json')[0]))
    path = os.path.join('../', exps_dir, experiment_name, timestamp, 'evaluation',split_filename.split('/')[-1].split('.json')[0], str(epoch))
    utils.mkdir_ifnotexists(path)

    dataset_path = conf.get_string('train.dataset_path')
    train_data_split = conf.get_string('train.data_split')
    latent_size = conf.get_int('train.latent_size')

    if (train_data_split == 'none'):
        ds = ReconDataSet(split=None, dataset_path=dataset_path, dist_file_name=None)
    else:
        dist_file_name = conf.get_string('train.dist_file_name')
        with open(split_filename, "r") as f:
            split = json.load(f)

        chamfer_results = []
        plot_cmpr = True
        ds = DFaustDataSet(split=split, dataset_path=dataset_path, dist_file_name=dist_file_name, with_gt=True)
        total_files = len(ds)
        logging.info ("total files : {0}".format(total_files))
    counter = 0
    dataloader = torch.utils.data.DataLoader(ds,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=1, drop_last=False, pin_memory=True)

    for data in dataloader:

        counter = counter + 1

        logging.info("evaluating " + ds.npyfiles_mnfld[data[-1]])

        input_pc = data[0].cuda()
        if latent_size > 0:
            latent = network.encoder(input_pc)
            if (type(latent) is tuple):
                latent = latent[0]
            points = torch.cat([latent.unsqueeze(1).repeat(1,input_pc.shape[1],1),input_pc],dim=-1)[0]
        else:
            latent = None
            points = input_pc[0]

        reconstruction = plt.plot_surface(with_points=False,
                         points=points,
                         decoder=network.decoder,
                         latent=latent,
                         path=path,
                         epoch=epoch,
                         in_epoch=ds.npyfiles_mnfld[data[-1].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1].item()].split('/')[-1].split('.npy')[0] + '_before',
                         shapefile=ds.npyfiles_mnfld[data[-1].item()],
                         resolution=resolution,
                         mc_value=0,
                         is_uniform_grid=True,
                         verbose=True,
                         save_html=False,
                         save_ply=True,
                         overwrite=True)
        if (with_opt):
            recon_after_latentopt = optimize_latent(latent, ds, data[-1], network.decoder, path, epoch,resolution,conf)

        if compute_dist_to_gt:
            gt_mesh_filename = ds.gt_files[data[-1]]
            normalization_params_filename = ds.normalization_files[data[-1]]

            logging.debug(
                "normalization params are " + normalization_params_filename
            )

            ground_truth_points = trimesh.Trimesh(trimesh.sample.sample_surface(trimesh.load(
                gt_mesh_filename
            ),30000)[0])


            normalization_params = np.load(normalization_params_filename,allow_pickle=True)

            scale = normalization_params.item()['scale']
            center = normalization_params.item()['center']

            chamfer_dist = utils.compute_trimesh_chamfer(
                gt_points=ground_truth_points,
                gen_mesh=reconstruction,
                offset=-center,
                scale=1./scale,
            )

            chamfer_dist_scan = utils.compute_trimesh_chamfer(
                gt_points=trimesh.Trimesh(input_pc[0].cpu().numpy()),
                gen_mesh=reconstruction,
                offset=0,
                scale=1.,
                one_side=True
            )

            logging.debug("chamfer distance: " + str(chamfer_dist))

            if (with_opt):
                chamfer_dist_after_opt = utils.compute_trimesh_chamfer(
                    gt_points=ground_truth_points,
                    gen_mesh=recon_after_latentopt,
                    offset=-center,
                    scale=1. / scale,
                )

                chamfer_dist_scan_after_opt = utils.compute_trimesh_chamfer(
                    gt_points=trimesh.Trimesh(input_pc[0].cpu().numpy()),
                    gen_mesh=recon_after_latentopt,
                    offset=0,
                    scale=1.,
                    one_side=True
                )

                chamfer_results.append(
                    (
                        ds.gt_files[data[-1]],
                        chamfer_dist,
                        chamfer_dist_scan,
                        chamfer_dist_after_opt,
                        chamfer_dist_scan_after_opt
                    )
                )
            else:
                chamfer_results.append(
                    (
                        ds.gt_files[data[-1]],
                        chamfer_dist,
                        chamfer_dist_scan
                    )
                )

            if (plot_cmpr):
                if (with_opt):
                    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}],
                                                               [{"type": "scene"}, {"type": "scene"}]],
                                        subplot_titles=["Input", "Registration",
                                                        "Ours", "Ours after opt"])

                else:
                    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}, {"type": "scene"},{"type": "scene"}]],
                                        subplot_titles=("input pc", "Ours","Registration"))

                fig.layout.scene.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                             yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                             zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                             aspectratio=dict(x=1, y=1, z=1)))
                fig.layout.scene2.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                              yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                              zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                              aspectratio=dict(x=1, y=1, z=1)))
                fig.layout.scene3.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                              yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                              zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                              aspectratio=dict(x=1, y=1, z=1)))
                if (with_opt):
                    fig.layout.scene4.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                  yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                  zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                  aspectratio=dict(x=1, y=1, z=1)))

                scan_mesh = trimesh.load(ds.scans_files[data[-1]])

                scan_mesh.vertices = scan_mesh.vertices - center

                def tri_indices(simplices):
                    return ([triplet[c] for triplet in simplices] for c in range(3))

                I, J, K = tri_indices(scan_mesh.faces)
                color = '#ffffff'
                trace = go.Mesh3d(x=scan_mesh.vertices[:, 0], y=scan_mesh.vertices[:, 1],
                                  z=scan_mesh.vertices[:, 2],
                                  i=I, j=J, k=K, name='scan',
                                  color=color, opacity=1.0, flatshading=False,
                                  lighting=dict(diffuse=1, ambient=0, specular=0), lightposition=dict(x=0, y=0, z=-1))
                fig.add_trace(trace, row=1, col=1)


                I, J, K = tri_indices(reconstruction.faces)
                color = '#ffffff'
                trace = go.Mesh3d(x=reconstruction.vertices[:, 0], y=reconstruction.vertices[:, 1], z=reconstruction.vertices[:, 2],
                                       i=I, j=J, k=K, name='our',
                                       color=color, opacity=1.0,flatshading=False,lighting=dict(diffuse=1,ambient=0,specular=0),lightposition=dict(x=0,y=0,z=-1))
                if (with_opt):
                    fig.add_trace(trace, row=2, col=1)

                    I, J, K = tri_indices(recon_after_latentopt.faces)
                    color = '#ffffff'
                    trace = go.Mesh3d(x=recon_after_latentopt.vertices[:, 0], y=recon_after_latentopt.vertices[:, 1],
                                      z=recon_after_latentopt.vertices[:, 2],
                                      i=I, j=J, k=K, name='our_after_opt',
                                      color=color, opacity=1.0, flatshading=False,
                                      lighting=dict(diffuse=1, ambient=0, specular=0),
                                      lightposition=dict(x=0, y=0, z=-1))
                    fig.add_trace(trace, row=2, col=2)
                else:
                    fig.add_trace(trace,row=1,col=2)

                gtmesh = trimesh.load(gt_mesh_filename)
                gtmesh.vertices = gtmesh.vertices - center
                I, J, K = tri_indices(gtmesh.faces)
                trace = go.Mesh3d(x=gtmesh.vertices[:, 0], y=gtmesh.vertices[:, 1],
                                  z=gtmesh.vertices[:, 2],
                                  i=I, j=J, k=K, name='gt',
                                  color=color, opacity=1.0, flatshading=False,
                                   lighting=dict(diffuse=1, ambient=0, specular=0),
                                   lightposition=dict(x=0,y=0,z=-1))
                if (with_opt):
                    fig.add_trace(trace, row=1, col=2)
                else:
                    fig.add_trace(trace, row=1, col=3)


                div = offline.plot(fig, include_plotlyjs=False, output_type='div', auto_open=False)
                div_id = div.split('=')[1].split()[0].replace("'", "").replace('"', '')
                if (with_opt):
                    js = '''
                                                    <script>
                                                    var gd = document.getElementById('{div_id}');
                                                    var isUnderRelayout = false
    
                                                    gd.on('plotly_relayout', () => {{
                                                      console.log('relayout', isUnderRelayout)
                                                      if (!isUnderRelayout) {{
                                                            Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
                                                              .then(() => {{ isUnderRelayout = false }}  )
                                                            Plotly.relayout(gd, 'scene3.camera', gd.layout.scene.camera)
                                                              .then(() => {{ isUnderRelayout = false }}  )
                                                            Plotly.relayout(gd, 'scene4.camera', gd.layout.scene.camera)
                                                              .then(() => {{ isUnderRelayout = false }}  )
                                                          }}
    
                                                      isUnderRelayout = true;
                                                    }})
                                                    </script>'''.format(div_id=div_id)
                else:
                    js = '''
                                    <script>
                                    var gd = document.getElementById('{div_id}');
                                    var isUnderRelayout = false
        
                                    gd.on('plotly_relayout', () => {{
                                      console.log('relayout', isUnderRelayout)
                                      if (!isUnderRelayout) {{
                                            Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
                                              .then(() => {{ isUnderRelayout = false }}  )
                                            Plotly.relayout(gd, 'scene3.camera', gd.layout.scene.camera)
                                              .then(() => {{ isUnderRelayout = false }}  )
                                          }}
        
                                      isUnderRelayout = true;
                                    }})
                                    </script>'''.format(div_id=div_id)
                # merge everything
                div = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
                print (ds.shapenames[data[-1]])
                with open(os.path.join(path, "compare_{0}.html".format(ds.shapenames[data[-1]])),
                          "w") as text_file:
                    text_file.write(div)

    if compute_dist_to_gt:
        with open(os.path.join(path,"chamfer.csv"),"w",) as f:
            if (with_opt):
                f.write("shape, chamfer_dist, chamfer scan dist, after opt chamfer dist, after opt chamfer scan dist\n")
                for result in chamfer_results:
                    f.write("{}, {} , {}\n".format(result[0], result[1], result[2], result[3], result[4]))
            else:
                f.write("shape, chamfer_dist, chamfer scan dist\n")
                for result in chamfer_results:
                    f.write("{}, {} , {}\n".format(result[0], result[1], result[2]))


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_name", required=True, help='The experiment name to be evaluated.')
    arg_parser.add_argument("--exps_dir", default="exps",  help='The experiments directory.')
    arg_parser.add_argument("--timestamp", default="latest",help="The experiemnt timestamp to test.")
    arg_parser.add_argument("--conf", required=True)
    arg_parser.add_argument("--checkpoint", default="latest",help="The trained model checkpoint to test.")
    arg_parser.add_argument("--split", required=True,help="The split to evaluate.")
    arg_parser.add_argument("--parallel", default=False, action="store_true", help="Should be set to True if the loaded model was trained in parallel mode.")
    arg_parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto].')
    arg_parser.add_argument('--with_opt', default=False, action="store_true", help='If set, optimizing latent with reconstruction Loss versus input scan')
    arg_parser.add_argument('--resolution', default=512, type=int, help='Grid resolution')
    arg_parser.add_argument('--compute_dist_to_gt', default=False, action="store_true", help='Set to True for computing chamfer distance between reconstruction to gt meshes')

    args = arg_parser.parse_args()
    utils.configure_logging(True,False,None)

    if args.timestamp == 'latest':
        timestamps = os.listdir(os.path.join('../',args.exps_dir,args.exp_name))
        timestamp = sorted(timestamps)[-1]
    elif args.timestamp == 'find':
        timestamps = [x for x in os.listdir(os.path.join('../',args.exps_dir,args.exp_name))
                      if not os.path.isfile(os.path.join('../',args.exps_dir,args.exp_name,x))]
        for t in timestamps:
            cpts = os.listdir(os.path.join('../',args.exps_dir,args.exp_name,t,'checkpoints/ModelParameters'))

            for c in cpts:
                if args.epoch + '.pth' == c:
                    timestamp = t
    else:
        timestamp = args.timestamp

    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    base_dir = os.path.join('../',args.exps_dir,args.exp_name, timestamp)
    saved_model_state = torch.load(os.path.join(base_dir, 'checkpoints', 'ModelParameters', args.checkpoint + ".pth"))
    saved_model_epoch = saved_model_state["epoch"]
    conf = ConfigFactory.parse_file(args.conf)
    network = utils.get_class(conf.get_string('train.network_class'))(conf=conf.get_config('network'),latent_size=conf.get_int('train.latent_size'))

    if (args.parallel):
        network.load_state_dict(
            {'.'.join(k.split('.')[1:]): v for k, v in saved_model_state["model_state_dict"].items()})
    else:
        network.load_state_dict(saved_model_state["model_state_dict"])

    evaluate(
        network=network.cuda(),
        exps_dir=args.exps_dir,
        experiment_name=args.exp_name,
        timestamp=timestamp,
        split_filename=args.split,
        epoch=saved_model_epoch,
        conf=conf,
        with_opt=args.with_opt,
        resolution=args.resolution,
        compute_dist_to_gt=args.compute_dist_to_gt
    )


