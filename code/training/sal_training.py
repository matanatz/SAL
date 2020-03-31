
import sys
sys.path.append('../code')
from utils.plots import plot_surface
import torch
from training.base_training import BaseTrainRunner
import logging
import time

class SalTrainRunner(BaseTrainRunner):

    def run(self):
        timing_log = []
        for epoch in range(self.start_epoch,self.nepochs + 2):
            start = time.time()

            if epoch % 100 == 0:
                self.save_checkpoints(epoch)
            if epoch % self.conf.get_int('train.plot_frequency') == 0 and epoch >= 0:
                with torch.no_grad():

                    self.network.eval()

                    pnts,_,idx = next(iter(self.eval_dataloader))
                    pnts = pnts.cuda()

                    if (self.parallel):
                        decoder = self.network.module.decoder
                        encoder = self.network.module.encoder
                    else:
                        decoder = self.network.decoder
                        encoder = self.network.encoder

                    if self.latent_size > 0:
                        latent = encoder(pnts)[0]

                        if (type(latent) is tuple):
                            latent = latent[0]
                        pnts = torch.cat([latent.unsqueeze(1).repeat(1,pnts.shape[1],1),pnts],dim=-1)[0]
                    else:
                        latent = None
                        pnts = pnts[0]

                    plot_surface(with_points=True,
                                 points=pnts,
                                 decoder=decoder,
                                 latent=latent,
                                 path=self.plots_dir,
                                 epoch=epoch,
                                 in_epoch=0,
                                 shapefile=self.ds.npyfiles_mnfld[idx],
                                 **self.conf.get_config('plot'))
                    self.network.train()

            self.network.train()
            if (self.adjust_lr):
                self.adjust_learning_rate(epoch)
            for data_index,(pnts_mnfld,sample_nonmnfld,indices) in enumerate(self.dataloader):

                pnts_mnfld = pnts_mnfld.cuda()
                sample_nonmnfld = sample_nonmnfld.cuda()
                xyz_nonmnfld = sample_nonmnfld[:,:,:3]
                dist_nonmnfld = sample_nonmnfld[:,:,3].reshape(-1)

                outputs = self.network(xyz_nonmnfld,pnts_mnfld)
                loss_res = self.loss(manifold_pnts_pred = outputs['manifold_pnts_pred'],
                                 nonmanifold_pnts_pred = outputs['nonmanifold_pnts_pred'],
                                 nonmanifold_gt = dist_nonmnfld,
                                 weight=None,
                                 latent_reg=outputs["latent_reg"])
                loss = loss_res["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logging.debug("expname : {0}".format(self.expname))
                logging.debug("timestamp: {0} , epoch : {1}, data_index : {2} , loss : {3}, reconstruction loss : {4} , vae loss : {5} ".format(self.timestamp,
                                                                                                                                                epoch,
                                                                                                                                                data_index,
                                                                                                                                                loss_res['loss'].item(),
                                                                                                                                                loss_res['recon_term'].item(),
                                                                                                                                                loss_res['reg_term'].item()))
                for param_group in self.optimizer.param_groups:
                    logging.debug("param group lr : {0}".format(param_group["lr"]))

            end = time.time()
            seconds_elapsed_epoch = end - start
            timing_log.append(seconds_elapsed_epoch)

