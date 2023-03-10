import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import torch
from models.archs.CARN_arch import CARN_M, EE_CARN_M
import numpy as np
import time

class CDNSR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(CDNSR, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = CARN_M(in_nc=in_nc, out_nc=out_nc, nf=36, scale=4, multi_scale=False, group=4)
        self.net2 = CARN_M(in_nc=in_nc, out_nc=out_nc, nf=52, scale=4, multi_scale=False, group=4)
        self.net3 = CARN_M(in_nc=in_nc, out_nc=out_nc, nf=64, scale=4, multi_scale=False, group=4)        
        self.tau = 1

    def forward(self, x,is_train):
        self.net1.eval() # must freeze bn
        self.net2.eval()
        self.net3.eval()
        outs_res = None
        if is_train:
            for i in range(len(x)):
                # print(x[i].unsqueeze(0).shape)
                type = self.classifier(x[i].unsqueeze(0))
                # p = F.softmax(type, dim=1)
                # print(type)
                p = arch_util.gumbel_softmax(type, dim=1, tau=self.tau)
                # p = arch_util.onehot_softmax(type, dim=1)
                p1 = p[0][0]
                p2 = p[0][1]
                p3 = p[0][2]

                out1 = self.net1(x[i].unsqueeze(0))
                out2 = self.net2(x[i].unsqueeze(0))
                out3 = self.net3(x[i].unsqueeze(0))

                out = out1 * p1 + out2 * p2 + out3 * p3
                outs = [out1, out2, out3]
                if i == 0:
                    out_res = out
                    outs_res = outs
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    outs_res = [torch.cat((outs_res[i], outs[i]), 0) for i in range(len(outs_res))]
                    type_res = torch.cat((type_res, p), 0)
            return [out_res, outs_res], type_res #兼容代码
        else:

            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))

                flag = torch.max(type, 1)[1].data.squeeze()
                p = F.softmax(type, dim=1)
                # p = arch_util.gumbel_softmax(type, dim=1, tau=self.tau)
                #flag=np.random.randint(0,2)
                #flag=1
                if flag == 0:
                    out = self.net1(x[i].unsqueeze(0))
                elif flag==1:
                    out = self.net2(x[i].unsqueeze(0))
                elif flag==2:
                    out = self.net3(x[i].unsqueeze(0))
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)

            return out_res, type_res

        return out_res,type_res

    def set_tau(self, tau):
        self.tau = tau

    def eval_bn(self):
        self.net1.eval()
        self.net2.eval()
        self.net3.eval()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))
        arch_util.initialize_weights([self.lastOut, self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out
