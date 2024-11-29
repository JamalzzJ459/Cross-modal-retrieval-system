import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import torch.nn.functional as F

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=True):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([K, T * math.sqrt(inputSize), -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        rnd = torch.randn(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', F.normalize(rnd.sign(), dim=1))
    
    
    def update_memory(self, data):
        memory = 0
        for i in range(len(data)):
            memory += data[i]
        memory /= memory.norm(dim=1, keepdim=True)
        self.memory.mul_(0).add_(memory)
    
    def forward(self, l, ab, y, idx=None, epoch=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()
    
        momentum = self.params[4].item() if (epoch is None) else (0 if epoch < 0 else self.params[4].item())

        batchSize = l.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)
    
        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        if momentum <= 0:
            weight = (l + ab) / 2.
            inx = torch.stack([torch.arange(batchSize)] * batchSize)
            inx = torch.cat([torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])], dim=1).to(weight.device).view([-1])
            weight = weight[inx].view([batchSize, batchSize, -1])
        else:
            weight = torch.index_select(self.memory, 0, idx.view(-1)).detach().view(batchSize, K + 1, inputSize)
    
        weight = weight.sign_()
        out_ab = torch.bmm(weight, ab.view(batchSize, inputSize, 1))
        # sample
        out_l = torch.bmm(weight, l.view(batchSize, inputSize, 1))
        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()
    
        # # update memory
        with torch.no_grad():
            l = (l + ab / 2)
            l.div_(l.norm(dim=1, keepdim=True))
            l_pos = torch.index_select(self.memory, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_pos = l_pos.div_(l_pos.norm(dim=1, keepdim=True))
            self.memory.index_copy_(0, y, l_pos)
    
        return out_l, out_ab

class NCEAverage2(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=True):
        '''
        inputSize:bit num
        outputSize:num_samples/batch_size
        K:negtive samples num --4096
        T:temperate
        momentum:factor of momentum use to update params
        '''
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)#多样式采样（均值采样？？？）
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([K, T * math.sqrt(inputSize), -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)#compute 标准差
        
        rnd = torch.randn(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)#从均值为0、标准差为stdv的正态分布中生成的。然后，将这个张量的值进行缩放和平移，使其范围落在[-stdv, stdv]之间
        #rnd shape(N,bit)  N是数据集总数
        #设置两个memory bank
        self.register_buffer('memory_l', F.normalize(rnd.sign(), dim=1))
        self.register_buffer('memory_ab', F.normalize(rnd.sign(), dim=1))

    def update_memory(self, data):
        memory = 0
        for i in range(len(data)):
            memory += data[i]
        memory /= memory.norm(dim=1, keepdim=True)
        self.memory.mul_(0).add_(memory)
    
    def forward(self, l, ab, y, idx=None, epoch=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item() if (epoch is None) else (0 if epoch < 0 else self.params[4].item())
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)#shape(batchsize,K+1) 这一步将开头的idx替换为正样本，其他均视为负样本
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach().view(batchSize, K + 1, inputSize)
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach().view(batchSize, K + 1, inputSize)
        
        
        #weight: (batchSize, K + 1, inputSize)
        #ab: (batchsize,inputSize)
        weight_l = weight_l.sign_()
        weight_ab = weight_ab.sign_()
        out_l_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))#(K+1,inputSize)
        out_ab_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        out_l_l = torch.bmm(weight_l, l.view(batchSize, inputSize, 1))#(K+1,inputSize)
        out_ab_ab = torch.bmm(weight_ab, ab.view(batchSize, inputSize, 1))
        if self.use_softmax:
            #first div T then keep memory contiguous
            out_ab_l = torch.div(out_ab_l, T)
            out_l_ab = torch.div(out_l_ab, T)
            out_ab_ab = torch.div(out_ab_ab, T)
            out_l_l = torch.div(out_l_l, T)
            out_l_ab = out_l_ab.contiguous()
            out_ab_l = out_ab_l.contiguous()
            out_l_l = out_l_l.contiguous()
            out_ab_ab = out_ab_ab.contiguous()
        else:
            out_ab_l = torch.exp(torch.div(out_ab_l, T))
            out_l_ab = torch.exp(torch.div(out_l_ab, T))
            out_ab_ab = torch.exp(torch.div(out_ab_ab, T))
            out_l_l = torch.exp(torch.div(out_l_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l_ab.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab_l.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l_ab = torch.div(out_l_ab, Z_l).contiguous()
            out_ab_l = torch.div(out_ab_l, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l.div_(l.norm(dim=1, keepdim=True))
            ab.div_(ab.norm(dim=1, keepdim=True))
            
            
            #更新memory_l
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_pos = l_pos.div_(l_pos.norm(dim=1, keepdim=True))
            
            #更新memory_ab
            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_pos = ab_pos.div_(ab_pos.norm(dim=1, keepdim=True))
            
            self.memory_l.index_copy_(0, y, l_pos)
            self.memory_ab.index_copy_(0, y, ab_pos)
            
        '''
        out_l:shape(batch_size,k+1)
        out_ab:shape(batch_size,k+1)
        '''
        return out_l_ab, out_ab_l
    
    
class new_contra(nn.Module):

    def __init__(self):
        super(new_contra, self).__init__()
        self.t = torch.tensor(0.3)
        self.accumu = 1
    
    def contra_loss(self,logits):
        logits = torch.exp(logits)
        dia = logits.diagonal()
        loss = -torch.mean(torch.log(dia/torch.sum(logits,dim=0)))-torch.mean(torch.log(dia/torch.sum(logits,dim=1)))
        return loss
    def compute_loss(self,logits,labels):
        loss  = F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels)
        return loss
    def forward(self, img, text, img2, text2,C=None):
        #如果i和j相似度高，则他们计算得得分也该更高，而我们希望所有得非coexist都低，所以反而会把h中得距离推的更远
        #那么如果我让相似度高的得分更低，是不是反过来可以制服那些相似度低的
        #S = torch.cosine_similarity(img.unsqueeze(0),text.unsqueeze(1),dim=2)
        #S = torch.cosine_similarity(torch.tanh(img*self.accumu).unsqueeze(0),torch.tanh(text*self.accumu).unsqueeze(1),dim=2)
        #print(torch.tanh(img*self.accumu))
        
        logits_it = img @ text.T *torch.exp(self.t)
        logits_ti = text @ img.T *torch.exp(self.t)
        logits_ii = img @ img2.T *torch.exp(self.t)
        logits_tt = text @ text2.T *torch.exp(self.t)
        labels = torch.arange(logits_it.shape[0])
        labels = labels.cuda()
        
        loss_it = self.contra_loss(logits_it)
        loss_ti = self.contra_loss(logits_ti)
        loss_ii = self.contra_loss(logits_ii)
        loss_tt = self.contra_loss(logits_tt)
        #loss_it  = self.compute_loss(logits,labels)
        #loss_ii = self.compute_loss(logits_ii,labels)
        #loss_tt = self.compute_loss(logits_tt,labels)
        #loss_recon = torch.sum((0.8*S - C)**2)
        loss = loss_it+ loss_ii+ loss_tt +loss_ti
        return loss
