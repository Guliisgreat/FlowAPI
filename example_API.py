
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from FlowNetAPI import FlowNet2API as FlowNet2
import utils.frame_utils as frame_utils



if __name__ == '__main__':

    FlowNet2 = FlowNet2(name_model= "FlowNet2")

    img0_name = "0.jpg"
    img1_name = "1.jpg"
    img2_name = "2.jpg"
    img3_name = "3.jpg"

    img0_list = []
    img1_list = []

    img0 = frame_utils.read_gen(img0_name)
    img2 = frame_utils.read_gen(img2_name)
    img0_list += [img0]
    img0_list += [img2]

    img1 = frame_utils.read_gen(img1_name)
    img3 = frame_utils.read_gen(img3_name)
    img1_list += [img1]
    img1_list += [img3]



    FlowNet2.flow_estimate( img0_list, img1_list,\
                            save_flow= True, \
                            flow_visualize= True,\
                            is_cropped= False)
    img0_list = []
    img1_list = []
    flow_list = []


    flow0_name = "./save_flow/inference/FlowNet2/000000.flo"
    img0_name = "0.jpg"
    img1_name = "1.jpg"


    flow1_name = "./save_flow/inference/FlowNet2/000001.flo"
    img2_name = "2.jpg"
    img3_name = "3.jpg"


    img0 = frame_utils.read_gen(img0_name)
    img1 = frame_utils.read_gen(img1_name)
    img0_list += [img0]
    img1_list += [img1]

    flow0 = frame_utils.read_gen(flow0_name)
    flow_list += [flow0]


    img2 = frame_utils.read_gen(img2_name)
    img3 = frame_utils.read_gen(img3_name)
    img0_list += [img2]
    img1_list += [img3]

    flow1 = frame_utils.read_gen(flow1_name)
    flow_list += [flow1]

    FlowNet2.flow_warping(img0_list, img1_list, flow_list)

