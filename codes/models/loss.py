from turtle import forward
import torch
import torch.nn as nn
from torchvision import models

from torchvision.models.vgg import vgg19
import torch.nn.functional as F
# just for single branch
class branch_flops_loss(nn.Module):
    def __init__(self, target_flops = None):
        super(branch_flops_loss, self).__init__()
        assert(target_flops is not None)
        self.target_flops = target_flops

    def forward(self, ch_mask): # ch_mask:[1, a, b, 2] a: channels num, b: layers num
        probs = ch_mask.softmax(dim=3)
        masks = (probs > 0.5).float().detach() - \
            probs.detach() + probs
        return max(0, (masks[:,:,:,1:].sum()) / (masks.size(1) * masks.size(2)) - self.target_flops)

class class_L1Lose(nn.Module):
    def __init__(self):
        super(class_L1Lose, self).__init__()
        self.l1_loss_func = nn.L1Loss()

    def forward(self, fakes, real, p):
        # fakes: [[tensor(96, 3, 128, 128)], [[tensor(96, 3, 128, 128)]], [[tensor(96, 3, 128, 128)]]]
        # p: tensort(96, 3)
        l_pix = 0
        for i in range(len(fakes)):
            for j in range(fakes[i].size(0)):
                l_pix += self.l1_loss_func(fakes[i][j], real[j]) * p[j][i]
        return l_pix

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        if torch.cuda.is_available():
            self.loss_network.cuda()
        self.l1_loss = nn.L1Loss()

    def forward(self, outs_all, type):
        # assert(len(outs_all) == 3)
        # assert(type.size(0) == 96 and type.size(1) == 3)
        net0 = self.loss_network(outs_all[0])
        net1 = self.loss_network(outs_all[1])
        net2 = self.loss_network(outs_all[2])
        for i in range(type.size(0)):
            if i == 0:
                perception_loss = (self.l1_loss(net0[i:i+1,::], net2[i:i+1,::]) + self.l1_loss(net0[i:i+1,::], net1[i:i+1,::])) * type[i][0]
            else:
                perception_loss += (self.l1_loss(net0[i:i+1,::], net2[i:i+1,::]) + self.l1_loss(net0[i:i+1,::], net1[i:i+1,::])) * type[i][0]
        # perception_loss = self.l1_loss(self.loss_network(outs_all), self.loss_network(fake_high_resolution))
        return perception_loss

    # def forward(self, high_resolution, fake_high_resolution):
    #     perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
    #     return perception_loss

class KDL1Loss(nn.Module):
    def __init__(self, cri_pix):
        super(KDL1Loss, self).__init__()
        self.cri_pix = cri_pix # 直接把外面的L1拿进来，懒得再自己生成一个了

    def forward(self, grond_truths, outs_all, type):
        # assert(len(outs_all) == 3)
        # assert(type.size(0) == 96 and type.size(1) == 3)
        kd_l1_loss = torch.tensor(0).float().cuda()
        for i in range(type.size(0)):
            # 计算每一个分支的l1 loss,如果教师l1loss优于学生，才蒸馏学生
            l1s = [self.cri_pix(outs_all[j][i:i+1,::], grond_truths[i:i+1,::]) for j in range(3)]
            # if l1s[0] > l1s[1]:
            kd_l1_loss += self.cri_pix(outs_all[0][i:i+1,::], outs_all[1][i:i+1,::].detach()) * type[i][0]
            # if l1s[0] > l1s[2]:
            kd_l1_loss += self.cri_pix(outs_all[0][i:i+1,::], outs_all[2][i:i+1,::].detach()) * type[i][0]
            # if l1s[1] > l1s[2]:
            kd_l1_loss += self.cri_pix(outs_all[1][i:i+1,::], outs_all[2][i:i+1,::].detach()) * type[i][1]
            # kd_l1_loss += (self.cri_pix(outs_all[0][i:i+1,::], outs_all[1][i:i+1,::].detach()) + self.cri_pix(outs_all[0][i:i+1,::], outs_all[2][i:i+1,::].detach())) * type[i][0]
            # kd_l1_loss += self.cri_pix(outs_all[1][i:i+1,::], outs_all[2][i:i+1,::].detach()) * type[i][1]
        assert(kd_l1_loss != 0)
        return kd_l1_loss

    # def forward(self, high_resolution, fake_high_resolution):
    #     perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
    #     return perception_loss

class EE_flops_loss(nn.Module):
    def __init__(self, branch_flops, target_flops=None):
        super(EE_flops_loss, self).__init__()
        assert(target_flops is not None)
        assert(target_flops >= 0 and target_flops <= 1)
        assert(len(branch_flops) > 0)
        self.branch_flops = branch_flops
        self.target_flops = target_flops

    def forward(self, type_res):
        sum_flops = torch.tensor(0.0).cuda()
        for i in range(len(type_res)):
            for j in range(len(self.branch_flops)):
                sum_flops += type_res[i][j] * self.branch_flops[j]
        ExpectedFlop = sum_flops / (len(type_res) * max(self.branch_flops))
        return max(torch.tensor(0.0).cuda(), (ExpectedFlop - self.target_flops)/(max(self.branch_flops)-min(self.branch_flops)))

class class_loss_3class(nn.Module):
    #Class loss
    def __init__(self):
        super(class_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0]) - 1
        type_all = type_res
        loss = 0
        for i in range(n):
            sum_re = abs(type_all[i][0]-type_all[i][1]) + abs(type_all[i][0]-type_all[i][2]) + abs(type_all[i][1]-type_all[i][2])
            loss += (m - sum_re)
        return loss / n


class average_loss_3class(nn.Module):
    #Average loss
    def __init__(self):
        super(average_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0])
        type_all = type_res
        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(n):
            sum1 += type_all[i][0]
            sum2 += type_all[i][1]
            sum3 += type_all[i][2]

        return (abs(sum1-n/m) + abs(sum2-n/m) + abs(sum3-n/m)) / ((n/m)*4)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

# contrastive loss
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        # for i in range(len(out)):
        #     print("i: {}, shape: {}".format(i, out[i].shape))
        #     nums = out[i].size(0) * out[i].size(1) * out[i].size(2) * out[i].size(3)
        #     zeros_nums = (out[i] == 0).sum().float()
        #     print("zeros_nums = {}, nums = {}, percent = {}".format(zeros_nums, nums, zeros_nums/nums))
        # exit(0)
        return out

class ContrastLoss(nn.Module):
    def __init__(self, weights, d_func, t_detach = False, is_one=False, save_path = None,loadpath=None):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = weights
        self.d_func = d_func
        self.is_one = is_one
        self.t_detach = t_detach

        self.backbone = self.vgg
        feat_dim = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).cuda()
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
            ).cuda()
        if loadpath is not None:
            self.load_proj_head(loadpath)
        assert(save_path is not None)
        self.save_path = save_path

    def forward(self, teacher, student, neg, blur_neg=None):
        teacher_vgg, student_vgg, neg_vgg, = self.vgg(teacher), self.vgg(student), self.vgg(neg)
        teacher_vgg = self.avgpool(teacher_vgg[-1])
        teacher_vgg = teacher_vgg.view(teacher_vgg.size(0), -1)

        student_vgg = self.avgpool(student_vgg[-1])
        student_vgg = student_vgg.view(student_vgg.size(0), -1)

        neg_vgg = self.avgpool(neg_vgg[-1])
        neg_vgg = neg_vgg.view(neg_vgg.size(0), -1)

        teacher_vgg, student_vgg, neg_vgg, = [self.proj_head(teacher_vgg)], [self.proj_head(student_vgg)], [self.proj_head(neg_vgg)]
        blur_neg_vgg = None
        if blur_neg is not None:
            blur_neg_vgg = self.vgg(blur_neg)
        if self.d_func == "L1":
            self.forward_func = self.L1_forward
        elif self.d_func == 'cos':
            self.forward_func = self.cos_forward

        return self.forward_func(teacher_vgg, student_vgg, neg_vgg, blur_neg_vgg)

    def L1_forward(self, teacher, student, neg, blur_neg=None):
        """
        :param teacher: 5*batchsize*color*patchsize*patchsize
        :param student: 5*batchsize*color*patchsize*patchsize
        :param neg: 5*negnum*color*patchsize*patchsize
        :return:
        """
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1)
            neg_i = neg_i.permute(1, 0, 2)### batchsize*negnum*color*patchsize*patchsize
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))


            if self.t_detach:
                d_ts = self.l1(teacher[i].detach(), student[i])
            else:
                d_ts = self.l1(teacher[i], student[i])
            d_sn = torch.mean(torch.abs(neg_i.detach() - student[i]).sum(0))

            contrastive = d_ts / (d_sn + 1e-7)
            loss += self.weights[i] * contrastive
        return loss


    def cos_forward(self, teacher, student, neg, blur_neg=None):
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1)
            neg_i = neg_i.permute(1, 0, 2)
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))

            if self.t_detach:
                d_ts = torch.cosine_similarity(teacher[i].detach(), student[i], dim=0).mean()
            else:
                d_ts = torch.cosine_similarity(teacher[i], student[i], dim=0).mean()
            d_sn = self.calc_cos_stu_neg(student[i], neg_i.detach())

            contrastive = -torch.log(torch.exp(d_ts)/(torch.exp(d_sn)+1e-7))
            loss += self.weights[i] * contrastive
        return loss

    def calc_cos_stu_neg(self, stu, neg):
        n = stu.shape[0]
        m = neg.shape[0]

        stu = stu.view(n, -1)
        neg = neg.view(m, n, -1)
        # normalize
        stu = F.normalize(stu, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=2)
        # multiply
        d_sn = torch.mean((stu * neg).sum(0))
        return d_sn

    def save_proj_head(self, iter_label):
        network = self.proj_head
        save_filename = '{}_proj_head.pth'.format(iter_label)
        save_path = os.path.join(self.save_path, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path) 

    def load_proj_head(self, load_path, strict=True):
        network = self.proj_head
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            print("k: ", k)
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)
   

class CSDLoss(nn.Module):
    def __init__(self, weights, d_func, t_detach = False, is_one=False):
        super(CSDLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = weights
        self.d_func = d_func
        self.is_one = is_one
        self.t_detach = t_detach

    def forward(self, teacher, student, neg, blur_neg=None):
        teacher_vgg, student_vgg, neg_vgg, = self.vgg(teacher), self.vgg(student), self.vgg(neg)
        blur_neg_vgg = None
        if blur_neg is not None:
            blur_neg_vgg = self.vgg(blur_neg)
        if self.d_func == "L1":
            self.forward_func = self.L1_forward

        return self.forward_func(teacher_vgg, student_vgg, neg_vgg, blur_neg_vgg)

    def L1_forward(self, teacher, student, neg, blur_neg=None):
        """
        :param teacher: 5*batchsize*color*patchsize*patchsize
        :param student: 5*batchsize*color*patchsize*patchsize
        :param neg: 5*negnum*color*patchsize*patchsize
        :return:
        """
        loss = 0
        for i in range(len(teacher)):
            neg_i = neg[i].unsqueeze(0)
            neg_i = neg_i.repeat(student[i].shape[0], 1, 1, 1, 1)
            neg_i = neg_i.permute(1, 0, 2, 3, 4)### batchsize*negnum*color*patchsize*patchsize
            if blur_neg is not None:
                blur_neg_i = blur_neg[i].unsqueeze(0)
                neg_i = torch.cat((neg_i, blur_neg_i))


            if self.t_detach:
                d_ts = self.l1(teacher[i].detach(), student[i])
            else:
                d_ts = self.l1(teacher[i], student[i])
            d_sn = torch.mean(torch.abs(neg_i.detach() - student[i]).sum(0))

            contrastive = d_ts / (d_sn + 1e-7)
            loss += self.weights[i] * contrastive
        return loss

import os
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel


if __name__ == '__main__':
    import torch
    cs = ContrastLoss([1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0], d_func="L1",t_detach=True)
    teacher = torch.rand([64, 3, 128, 128]).cuda()
    student = torch.rand([64, 3, 128, 128]).cuda()
    neg = teacher[torch.randperm(5), :, :, :].cuda()
    cs(teacher, student, neg)
    