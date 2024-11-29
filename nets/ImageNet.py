import torch
from torch import nn
from torch.nn import functional as F
import math
from model.model import build_model

class ImageNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=8192, mid_num2=8192, hiden_layer=3):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ImageNet, self).__init__()
        self.module_name = "img_model"
        self.clipPath="./ViT-B-32.pt"
        self.clip = self.load_clip(self.clipPath)
        
        modules = [nn.Linear(y_dim, mid_num1),
                   #nn.Dropout(0.05),
                   nn.ReLU(inplace=True),
                   nn.Linear(mid_num1, mid_num2),
                   #nn.Dropout(0.05),
                   nn.ReLU(inplace=True),
                   nn.Linear(mid_num2, bit),
                   #nn.Dropout(0.05),
                   ]
        modules_dec = [nn.Linear(bit, mid_num2),
               nn.ReLU(inplace=True),
               nn.Linear(mid_num2, mid_num1),
               nn.ReLU(inplace=True),
               nn.Linear(mid_num1, y_dim),
               nn.Sigmoid(),
               ]
        modules_g = [nn.Linear(bit,bit),
                     nn.ReLU(inplace=True),]
        self.encoder = nn.Sequential(*modules)
        self.decoder = nn.Sequential(*modules_dec
            )
        self.g_net = nn.Sequential(*modules_g)
        #self.apply(weights_init)
        self.norm = norm
        self.alpha = 1.0
    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")
        
        return build_model(state_dict)
    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
    def encode(self,x):
        fea = self.clip.encode_image(x).detach()
        fea = fea.type(torch.float32)
        out = torch.tanh(self.alpha *self.encoder(fea))
        return out
    def decode(self,out):
        recon = self.decoder(out)
        return recon
    def forward(self, x):
        out = self.encode(x)
        recon = self.decode(out).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x

        return out,recon
