import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from nets.backbone.vgg import vgg as add_vgg


def add_extras(in_channels: int, backbone_name: str):
    layers = []
    if backbone_name == 'vgg':
        # block 6
        # 19, 19, 1024 -> 19, 19, 256 -> 10, 10, 512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # block 7
        # 10, 10, 512 -> 10, 10, 128 -> 5, 5, 256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # block 8
        # 5,5,256 -> 5,5,128 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

        # block 9
        # 3,3,256 -> 3,3,128 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return nn.ModuleList(layers)


class L2Norm(nn.Module):
    def __init__(self, n_channels: int, scale: float):
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD300(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        if backbone_name == 'vgg':
            self.vgg = add_vgg()
            self.extras = add_extras(in_channels=1024, backbone_name=backbone_name)
            self.L2Norm = L2Norm(512, 20)

            # ??????feature map??????????????????????????????box???number
            mbox = [4, 6, 6, 6, 4, 4]

            loc_layers = []
            conf_layers = []
            backbone_source = [21, -2]

            # ---------------------------------------------------#
            #   ???add_vgg?????????????????????
            #   ???21??????-2???????????????????????????????????????????????????
            #   ?????????conv4-3(38,38,512)???conv7(19,19,1024)?????????
            # ---------------------------------------------------#
            for k, v in enumerate(backbone_source):
                loc_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

            # -------------------------------------------------------------#
            #   ???add_extras?????????????????????
            #   ???1?????????3?????????5?????????7???????????????????????????????????????????????????
            #   shape?????????(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

            self.loc = nn.ModuleList(loc_layers)
            self.conf = nn.ModuleList(conf_layers)
            self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor):
        sources = list()
        loc = list()
        conf = list()

        for k in range(23):
            x = self.vgg[k](x)

        # ---------------------------#
        #   conv4_3?????????
        #   ????????????L2?????????
        # ---------------------------#
        s = self.L2Norm(x)
        sources.append(s)

        # --------------------------- #
        #   ??????conv7?????????
        #   shape???19,19,1024
        # --------------------------- #
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # -------------------------------------------------------------#
        #   ???add_extras?????????????????????
        #   ???1?????????3?????????5?????????7???????????????????????????????????????????????????
        #   shape?????????(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # -------------------------------------------------------------#
        #   ????????????6???????????????????????????????????????????????????
        #   sources???input shape???38,38,512???  (19,19,1024) (10,10,512) (5,5,256) (3,3,256) (1,1,256)
        #   loc???conv shape      (512, 4 * 4)
        #   conf???conv shape     (512, 4 * 21)
        #   ???????????????loc shape    (38, 38, 16)
        # -------------------------------------------------------------#
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # (1, 38 16 38) (1, )
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # -------------------------------------------------------------#
        #   ??????reshape????????????
        # -------------------------------------------------------------#
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # -------------------------------------------------------------#
        #   loc???reshape???batch_size, num_anchors, 4
        #   conf???reshape???batch_size, num_anchors, self.num_classes
        # -------------------------------------------------------------#
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        
        return output
