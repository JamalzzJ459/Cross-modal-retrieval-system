from torch import nn

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.softmax(1)#因为用了softmax，希望所有正样本的概率高也就是让其他的概率低
        return -x[:, 0].log().mean()