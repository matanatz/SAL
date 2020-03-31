import utils.general as utils
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import numpy as np
import json
import logging



class BaseTrainRunner():
    def __init__(self,**kwargs):

        if (type(kwargs['conf']) == str):
            self.conf = ConfigFactory.parse_file(kwargs['conf'])
            self.conf_filename = kwargs['conf']
        else:
            self.conf = kwargs['conf']
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.expnameraw = self.conf.get_string('train.expname')
        self.expname = self.conf.get_string('train.expname') +  kwargs['expname']

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.adjust_lr = self.conf.get_bool('train.adjust_lr')
        self.GPU_INDEX = kwargs['gpu_index']
        self.exps_folder_name = kwargs['exps_folder_name']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))

        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        log_dir = os.path.join(self.expdir, self.timestamp, 'log')
        self.log_dir = log_dir
        utils.mkdir_ifnotexists(log_dir)
        utils.configure_logging(kwargs['debug'],kwargs['quiet'],os.path.join(self.log_dir,'log.txt'))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path,self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        if (not self.GPU_INDEX == 'all'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        # Backup code
        self.code_path = os.path.join(self.expdir, self.timestamp, 'code')
        utils.mkdir_ifnotexists(self.code_path)
        for folder in ['training','preprocess','utils','model','datasets','confs']:
            utils.mkdir_ifnotexists(os.path.join(self.code_path, folder))
            os.system("""cp -r ./{0}/* "{1}" """.format(folder,os.path.join(self.code_path, folder)))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.code_path, 'confs/runconf.conf')))

        logging.info('shell command : {0}'.format(' '.join(sys.argv)))

        if (self.conf.get_string('train.data_split') == 'none'):
            self.ds = utils.get_class(self.conf.get_string('train.dataset'))(split=None, dataset_path=self.conf.get_string('train.dataset_path'), dist_file_name=None)
        else:
            train_split_file = './confs/splits/{0}'.format(self.conf.get_string('train.data_split'))

            with open(train_split_file, "r") as f:
                train_split = json.load(f)

            self.ds = utils.get_class(self.conf.get_string('train.dataset'))(split=train_split,
                                                                             dataset_path=self.conf.get_string('train.dataset_path'),
                                                                             dist_file_name=self.conf.get_string('train.dist_file_name'))

        self.dataloader = torch.utils.data.DataLoader(self.ds,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=kwargs['workers'],drop_last=True,pin_memory=True)
        self.eval_dataloader = torch.utils.data.DataLoader(self.ds,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           num_workers=0, drop_last=True)

        self.latent_size = self.conf.get_int('train.latent_size')

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(conf=self.conf.get_config('network'),
                                                                                    latent_size=self.latent_size)
        if kwargs['parallel']:
            self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()

        self.parallel = kwargs['parallel']
        self.loss = utils.get_class(self.conf.get_string('network.loss.loss_type'))(**self.conf.get_config('network.loss.properties'))
        self.lr_schedules = BaseTrainRunner.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))

        self.optimizer = torch.optim.Adam(
        [
            {
                "params": self.network.parameters(),
                "lr": self.lr_schedules[0].get_learning_rate(0),
            }
        ])

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.start_epoch = saved_model_state['epoch']

    def get_learning_rate_schedules(schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self,epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)),1.0e-5)

