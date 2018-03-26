import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from scipy import misc

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import json
from bunch import Bunch

import source.models, source.losses, source.datasets
from utils import flow_utils, tools
import utils.frame_utils as frame_utils
import torch.utils.data as data

from source.datasets import StaticRandomCrop as StaticRandomCrop
from source.datasets import StaticCenterCrop as StaticCenterCrop


from networks.resample2d_package.modules.resample2d import Resample2d
from networks.channelnorm_package.modules.channelnorm import ChannelNorm

from networks.submodules import *

class FlowNet2API(object):
    def __init__(self, name_model="FlowNet2"):

        self.name_model = name_model

        self.get_config()
        self.load_model()



    def flow_estimate(self, img0, img1, save_flow=True, flow_visualize=False, is_cropped =False, inference_batch_size = 1):

        self.args.save_flow = save_flow
        self.flow_visualize = flow_visualize
        self.args.inference_batch_size = inference_batch_size
        self.args.is_cropped = is_cropped

        data = self.flow_dataloader(img0, img1)
        self.inference(data, self.model_and_loss, offset=1)


    def flow_warping(self,img0, img1, flow,warping_batch_size = 1):

        # if img0 == None:
        #     self.is_warpingLoss = False
        self.args.warping_batch_size = warping_batch_size
        self.args.is_cropped = False

        data = self.warping_dataloader( img0, img1, flow)
        self.warping_inference(data, self.warping_model, offset=1)




























    def flow_dataloader(self, img0, img1):
        dataset = FlowEstimateDataset(self.args, img0, img1)
        inference_dataloader = DataLoader(dataset, batch_size=self.args.inference_batch_size, shuffle=False, **self.gpuargs)

        return inference_dataloader



    def inference(self, data_loader, model, offset=0):
        model.eval()

        if self.args.save_flow :
            flow_folder = self.args.inference_dir
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)

        self.args.inference_n_batches = np.inf if self.args.inference_n_batches < 0 else self.args.inference_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), self.args.inference_n_batches),
                        desc='Inferencing ',
                        leave=True, position=offset)

        for batch_idx, (data, target) in enumerate(progress):
            if self.args.cuda:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
            data, target = [Variable(d, volatile=True) for d in data], [Variable(t, volatile=True) for t in target]

            losses, output = model(data[0], target[0], inference=True)

            if self.args.save_flow:
                for i in range(self.args.inference_batch_size):
                    _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                    flow_utils.writeFlow(join(flow_folder, '%06d.flo' % (batch_idx * self.args.inference_batch_size + i)),
                                         _pflow)

                    if self.flow_visualize:
                        flowX = _pflow[:, :, 0]
                        plt.imshow(flowX)
                        plt.savefig(fname= join(flow_folder, '%06d_x.png' % (batch_idx * self.args.inference_batch_size + i)))

                        flowY = _pflow[:, :, 1]
                        plt.imshow(flowY)
                        plt.savefig(
                            fname=join(flow_folder, '%06d_y.png' % (batch_idx * self.args.inference_batch_size + i)))


            progress.update(1)

        progress.close()


        return



    def warping_dataloader(self, img0, img1, flow):
        dataset = WarpingDataset(self.args, img0, img1, flow)
        warping_dataloader = DataLoader(dataset, batch_size=self.args.warping_batch_size, shuffle=False,
                                          **self.gpuargs)
        return warping_dataloader

    def warping_inference(self, data_loader, model, offset=1):
        model.eval()

        self.args.warping_n_batches = np.inf if self.args.warping_n_batches < 0 else self.args.warping_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), self.args.warping_n_batches),
                        desc='Warping ',
                        leave=True, position=offset)

        statistics = []
        total_loss = 0
        for batch_idx, (data, target) in enumerate(progress):
            if self.args.cuda:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
            data, target = [Variable(d, volatile=True) for d in data], [Variable(t, volatile=True) for t in target]

            warped_data, losses = model(data[0], target[0])
            # losses = torch.norm(losses, p=2, dim=1).mean()
            #
            # total_loss += losses.data[0]
            # loss_values = [v.data[0] for v in losses]
            #
            # statistics.append(loss_values)
            for i in range(self.args.warping_batch_size):
                warped_data = warped_data[i].data.cpu().numpy().transpose(1, 2, 0)
                #misc.imshow(warped_data)
                misc.imsave('warped_image'+str(batch_idx)+'.png', warped_data)


            progress.update(1)

        progress.close()

        return






    def get_config(self):

        json_file = "config.json"
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)

        # convert the dictionary to a namespace using bunch lib
        self.args = Bunch(config_dict)
        self.args.name_model = self.name_model
        self.args.model = self.name_model.replace('-', '')

        self.args.resume = os.path.join(self.args.dir_checkpoints,(self.args.name_model + "_checkpoint.pth.tar"))

        self.args.model_class = tools.module_to_dict(source.models)[self.args.model]
        self.args.loss_class = tools.module_to_dict(source.losses)[self.args.loss]
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        self.args.inference_dir = os.path.join("{}/inference".format(self.args.save), self.args.name_model)


    def load_model(self):

        self.gpuargs = {'num_workers': self.args.number_workers, 'pin_memory': True} if self.args.cuda else {}

        with tools.TimerBlock("Building {} model".format(self.args.model)) as block:
            class ModelAndLoss(nn.Module):
                def __init__(self, args):
                    super(ModelAndLoss, self).__init__()
                    kwargs = tools.kwargs_from_args(args, 'model')
                    self.model = args.model_class(args, **kwargs)
                    kwargs = tools.kwargs_from_args(args, 'loss')
                    self.loss = args.loss_class(args, **kwargs)

                def forward(self, data, target, inference=False):
                    output = self.model(data)

                    loss_values = self.loss(output, target)

                    if not inference:
                        return loss_values
                    else:
                        return loss_values, output

            self.model_and_loss = ModelAndLoss(self.args)

            # block.log('Effective Batch Size: {}'.format(self.args.effective_batch_size))
            block.log('Number of parameters: {}'.format(
                sum([p.data.nelement() if p.requires_grad else 0 for p in self.model_and_loss.parameters()])))

            ## # assing to cuda or wrap with dataparallel, model and loss
            if self.args.cuda and (self.args.number_gpus > 0) and self.args.fp16:
                block.log('Parallelizing')
                model_and_loss = nn.parallel.DataParallel(self.model_and_loss, device_ids=list(range(self.args.number_gpus)))

                block.log('Initializing CUDA')
                model_and_loss = model_and_loss.cuda().half()
                torch.cuda.manual_seed(self.args.seed)
                param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in
                              model_and_loss.parameters()]

            elif self.args.cuda and self.args.number_gpus > 0:
                block.log('Initializing CUDA')
                model_and_loss = self.model_and_loss.cuda()
                block.log('Parallelizing')
                model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(self.args.number_gpus)))
                torch.cuda.manual_seed(self.args.seed)

            else:
                block.log('CUDA not being used')
                torch.manual_seed(self.args.seed)
            cwd = os.getcwd()
            print(cwd)
            # Load weights if needed, otherwise randomly initialize
            if self.args.resume and os.path.isfile(self.args.resume):
                block.log("Loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                # if (not args.inference) and (not args.test):
                #     args.start_epoch = checkpoint['epoch']
                # best_err = checkpoint['best_EPE']
                model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
                block.log("Loaded checkpoint '{}' (at epoch {})".format(self.args.resume, checkpoint['epoch']))

            elif self.args.resume:
                block.log("No checkpoint found at '{}'".format(self.args.resume))
                quit()

            else:
                block.log("Random initialization")

            block.log("Initializing save directory: {}".format(self.args.save))
            if not os.path.exists(self.args.save):
                os.makedirs(self.args.save)

        self.warping_model = FlowWarping()
        print("Warping Model initialized")



class FlowEstimateDataset(data.Dataset):
    def __init__(self, args, img0, img1):
        self.args = args
        self.is_cropped = self.args.is_cropped
        self.crop_size = self.args.crop_size
        self.render_size = self.args.inference_size

        for i in range(len(img0)):
            assert (img0[i].shape == img1[i].shape), "a pair of images should have same shape"

        self.image_list = []
        self.flow_list = []

        self.frame_size = img0[0].shape


        for i in range(len(img0)):
            self.image_list += [[img0[i], img1[i]]]
            self.flow_size = img0[i][:, :, 0:2].shape
            flow = np.zeros(self.flow_size, dtype=float)
            self.flow_list += [flow]

        self.size = len(self.image_list)

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) / 64) * 64
            self.render_size[1] = ((self.frame_size[1]) / 64) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))


    def __getitem__(self, index):
        index = index % self.size

        img1 = self.image_list[index][0]
        img2 = self.image_list[index][1]

        flow = self.flow_list[index]

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = map(cropper, images)
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size


class WarpingDataset(data.Dataset):
    def __init__(self, args, img0, img1, flow):
        self.args = args
        self.is_cropped = self.args.is_cropped
        self.crop_size = self.args.crop_size
        self.render_size = self.args.inference_size

        # for i in range(len(img1)):
        #     assert (img1[i][:,:,0].shape == flow[i][:,:,0].shape), " image and flow should have same shape"

        self.warp_list = []
        self.target_list = []

        self.frame_size = img1[0].shape


        for i in range(len(img1)):
            self.warp_list += [[img1[i], flow[i]]]
            self.target_list += [img0[i]]

        self.size = len(self.warp_list)

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) / 64) * 64
            self.render_size[1] = ((self.frame_size[1]) / 64) * 64

        args.inference_size = self.render_size

        assert (len(self.warp_list) == len(self.target_list))


    def __getitem__(self, index):
        index = index % self.size

        img1 = self.warp_list[index][0]
        flow = self.warp_list[index][1]

        img0 = self.target_list[index]

        images = [img0, img1]

        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = map(cropper, images)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]


    def __len__(self):
        return self.size



class FlowWarping(nn.Module):

    def __init__(self):
        super(FlowWarping, self).__init__()

        self.channelnorm = ChannelNorm()
        self.resample1 = Resample2d()


    def forward(self, input, target):

        img0 = input[:,:,0,:,:]
        img1 = input[:,:,1,:,:]
        flow = target

        frame_size = img0.shape


        resampled_img1 = self.resample1(img1, flow)
        diff_img0 = img0 - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0) / (frame_size[1] * frame_size[2])

        return resampled_img1, norm_diff_img0