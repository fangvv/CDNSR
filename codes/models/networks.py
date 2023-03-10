import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.CDNSR_fsrcnn_arch as CDNSR_fsrcnn_arch
import models.archs.CDNSR_rcan_arch as CDNSR_rcan_arch
import models.archs.CDNSR_carn_arch as CDNSR_carn_arch
import models.archs.CDNSR_srresnet_arch as CDNSR_srresnet_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.FSRCNN_arch as FSRCNN_arch
import models.archs.CARN_arch as CARN_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])

    elif which_model == 'RCAN':
        netG = RCAN_arch.RCAN(n_resblocks=opt_net['n_resblocks'], n_feats=opt_net['n_feats'],
                              res_scale=opt_net['res_scale'], n_colors=opt_net['n_colors'],rgb_range=opt_net['rgb_range'],
                              scale=opt_net['scale'],reduction=opt_net['reduction'],n_resgroups=opt_net['n_resgroups'])

    elif which_model == 'CARN_M':
        netG = CARN_arch.CARN_M(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])

    elif which_model == 'EE_CARN_M':
        netG = CARN_arch.EE_CARN_M(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])

    elif which_model == 'fsrcnn':
        netG = FSRCNN_arch.FSRCNN_net(input_channels=opt_net['in_nc'],upscale=opt_net['scale'],d=opt_net['d'],
                                      s=opt_net['s'],m=opt_net['m'])

    elif which_model == 'classSR_3class_fsrcnn_net':
        netG = CDNSR_fsrcnn_arch.CDNSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'classSR_3class_rcan':
        netG = CDNSR_rcan_arch.CDNSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'classSR_3class_srresnet':
        netG = CDNSR_srresnet_arch.CDNSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'classSR_3class_carn':
        netG = CDNSR_carn_arch.CDNSR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG