import argparse
import sys
sys.path.append('../code')
from training.sal_training import SalTrainRunner
import GPUtil



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size.')
    parser.add_argument('--nepoch', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--conf', type=str, default='./confs/dfaust.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='all', help='GPU to use [default: all].')
    parser.add_argument('--parallel', default=False,action="store_true", help='If set, indicaties running on multiple gpus.')
    parser.add_argument('--workers', type=int, default=1, help='Data loader number of workers.')
    parser.add_argument('--is_continue', default=False,action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', type=str,help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument("--debug",default=True,action="store_true",help="If set, debugging messages will be printed.")
    parser.add_argument("--quiet",dest="quiet",default=False,action="store_true",help="If set, only warnings will be printed.")

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu



    trainrunner = SalTrainRunner(conf=opt.conf,
                                      batch_size=opt.batch_size,
                                      nepochs=opt.nepoch,
                                      expname=opt.expname,
                                      gpu_index=gpu,
                                      exps_folder_name='exps',
                                      parallel=opt.parallel,
                                      workers=opt.workers,
                                      is_continue=opt.is_continue,
                                      timestamp=opt.timestamp,
                                      checkpoint=opt.checkpoint,
                                debug=opt.debug,
                                 quiet=opt.quiet)

    trainrunner.run()