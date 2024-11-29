seed = 123
import numpy as np
from sympy import arg
np.random.seed(seed)
import random as rn
rn.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils.config import args
import time
from datetime import datetime

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True
from torch.utils.data import SubsetRandomSampler
import nets as models
# from utils.preprocess import *
from utils.bar_show import progress_bar
import pdb
from src.cmdataset import CMDataset
import scipy
import scipy.spatial
import torch.nn as nn
import src.utils as utils
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCESoftmaxLoss
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

import wandb
# --pretrain --arch resnet18
#  CUDA_VISIBLE_DEVICES=5 python UCCH.py --data_name mscoco_fea --bit 32 --alpha 0.7 --num_hiden_layers 3 2 --margin 0.2 --max_epochs 20 --train_batch_size 256 --shift 0.1 --lr 0.0001 --optimizer Adam --momentum 0.9
'''
where is K
'''

device_ids = [0, 1]
teacher_device_id = [0, 1]
best_acc = 0  # best test accuracy
start_epoch = 0

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

def main():
    print('===>start wandb ..')
    '''
    wandb.init(
      project="multi-view-cross-modal-hashing",
      name = "对比学习多视角（4个） +跨模态检索任务模态间多视角",
      notes="只使用对比学习 + 无多视角（4个）",
      tags=["MVCMH-T", "-"],
      config=args,
    )'''
    
    print('===> Preparing data ..')
        # build data
    train_dataset = CMDataset(
        args.data_name,
        return_index=True,
        partition='train',
        two_crop=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    test_dataset = CMDataset(
        args.data_name,
        return_index=True,
        partition='test'
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    retrieval_dataset = CMDataset(
        args.data_name,
        return_index=True,
        partition= 're'
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print('===> Building ResNet..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'fea' in args.data_name:
        #if use pretrain fea  
        image_model = models.__dict__['ImageNet'](y_dim=512, bit=args.bit, hiden_layer=args.num_hiden_layers[0]).cuda()
        backbone = None
    else:
        backbone = models.__dict__[args.arch](pretrained=args.pretrain, feature=True).cuda()
        fea_net = models.__dict__['ImageNet'](y_dim=4096 if 'vgg' in args.arch.lower() else (512 if args.arch == 'resnet18' or args.arch == 'resnet34' else 2048), bit=args.bit, hiden_layer=args.num_hiden_layers[0]).cuda()
        image_model = nn.Sequential(backbone, fea_net)
    text_model = models.__dict__['TextNet'](y_dim=512, bit=args.bit, hiden_layer=args.num_hiden_layers[1]).cuda()

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    wd = args.wd
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)
    if args.ls == 'cos':
        lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.1)

    summary_writer = SummaryWriter(args.log_dir)
    
    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        image_model.load_state_dict(ckpt['image_model_state_dict'])
        text_model.load_state_dict(ckpt['text_model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    def set_train(is_warmup=False):
        image_model.train()
        if is_warmup and backbone:
            backbone.eval()
            backbone.requires_grad_(False)
        elif backbone:
            backbone.requires_grad_(True)
        text_model.train()

    def set_eval():
        image_model.eval()
        text_model.eval()
    criterion = utils.ContrastiveLoss(args.margin, shift=args.shift)
    n_data = len(train_loader.dataset)
    #print("n_data",n_data) 117218
    contrast = NCEAverage(args.bit, n_data, args.K, args.T, args.momentum)
    criterion_contrast = NCESoftmaxLoss()
    contrast = contrast.cuda()
    criterion_contrast = criterion_contrast.cuda()
    def compute_loss(y_pred):
        idxs = torch.arange(0,y_pred.shape[0],device='cuda')
        y_true = idxs + 1 - idxs % 2 * 2
        similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        #torch自带的快速计算相似度矩阵的方法
        similarities = similarities-torch.eye(y_pred.shape[0],device='cuda') * 1e12
        #屏蔽对角矩阵即自身相等的loss
        lamda = 0.05
        similarities = similarities / lamda
        #论文中除以 temperature 超参 0.05
        loss = F.cross_entropy(similarities,y_true)
        return torch.mean(loss)
    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train(epoch < args.warmup_epoch)
        # set_train(True)
        train_loss, correct, total = 0., 0., 0.
        for batch_idx, (idx, images, texts, _) in enumerate(train_loader):
            images = images.float()
            #print(images.shape)
            l, ab = torch.split(images,3, dim=1)
            l = l.cuda()
            ab = ab.cuda()
            #text_t = text_t.cuda()
            
            texts = texts.float()
            texts = texts.cuda()
            idx = idx.cuda()
            images_outputs,im_recon= image_model(l)
            images_outputs2,im_recon2= image_model(ab)
            texts_outputs,txt_recon = text_model(texts)
            texts_outputs2,txt_recon2 = text_model(texts)
            #l1_loss = nn.L1Loss()
            
            '''
            F_I = F.normalize(images)
            F_T = F.normalize(texts)
            S_I = F_I.mm(F_I.t())
            S_T = F_T.mm(F_T.t())
            F_I_D = F.normalize(im_recon)
            F_T_D = F.normalize(txt_recon)
            S_I_D = F_I_D.mm(F_I.t())
            S_T_D = F_T_D.mm(F_T.t())
            recon_loss = F.mse_loss(S_I_D ,S_I) +F.mse_loss(S_T_D ,S_T)
            '''
            out_l, out_ab = contrast(images_outputs, texts_outputs, idx *1,epoch=epoch-args.warmup_epoch )
            #out_l2, out_ab2 = contrast(images_outputs2, texts_outputs2, idx * B,epoch=epoch-args.warmup_epoch)
            #out_l3, out_ab3 = contrast(images_outputs, images_outputs2, idx * B,epoch=epoch-args.warmup_epoch)
            #out_l4, out_ab4 = contrast(texts_outputs, texts_outputs2, idx * B,epoch=epoch-args.warmup_epoch)
            l_loss = criterion_contrast(out_l)
            ab_loss = criterion_contrast(out_ab)
            '''
            l_loss2 = criterion_contrast(out_l2)
            ab_loss2 = criterion_contrast(out_ab2)
            l_loss3 = criterion_contrast(out_l3)
            ab_loss3 = criterion_contrast(out_ab3)
            l_loss4 = criterion_contrast(out_l4)
            ab_loss4 = criterion_contrast(out_ab4)
            '''
            Lc = l_loss + ab_loss 
            #Lc = l_loss + ab_loss +l_loss2 + ab_loss2 +l_loss3 +ab_loss3 + l_loss4 +ab_loss4
            #Lr = criterion(images_outputs, texts_outputs)
            #Lc = new_con(images_outputs, texts_outputs,images_outputs2,texts_outputs2)
            Lr = criterion(images_outputs, texts_outputs,images_outputs2,texts_outputs)
            #loss = Lc
            loss = (args.alpha *Lc + (1.0 - args.alpha) * Lr) 

            #loss = Lc2
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters, 1.)
            optimizer.step()
            train_loss += loss.item()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

            if batch_idx % args.log_interval == 0:  #every log_interval mini_batches...
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)

    def eval(data_loader):
        imgs, txts, labs = [], [], []
        with torch.no_grad():
            for batch_idx, (idx,images,texts,targets) in enumerate(data_loader):
                images = images.float()
                texts = texts.float()
                images = images.cuda()
                texts = texts.cuda()
                images_outputs= image_model.encode(images)
                texts_outputs = text_model.encode(texts)
                imgs.append(images_outputs)
                txts.append(texts_outputs)
                labs.append(targets)

            imgs = torch.cat(imgs).sign_().cpu().numpy()
            txts = torch.cat(txts).sign_().cpu().numpy()
            labs = torch.cat(labs).cpu().numpy()
        return imgs, txts, labs

    def test(epoch, is_eval=True):
        # pass
        global best_acc
        set_eval()
        # switch to evaluate mode
        
        if is_eval:
            (retrieval_imgs, retrieval_txts,retrieval_labs) = eval(query_loader)
            query_imgs, query_txts, query_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
            retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
        else:
            (retrieval_imgs, retrieval_txts,retrieval_labs) = eval(retrieval_loader)
            (query_imgs, query_txts,query_labs) = eval(query_loader)
        print("获取测试集完成")
        i2t ,i2t_all= fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0, metric='hamming')
        t2i ,t2i_all= fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0, metric='hamming')
        i2i ,i2i_all= fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_imgs, query_labs, k=0, metric='hamming')
        print("计算矩阵完成")
        #i2t = calc_map_k_matrix(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0)
        #t2i = calc_map_k_matrix(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0)
        avg = (i2t + t2i) / 2.
        
        '''
        wandb.log({'i2t': i2t_all, 't2i': t2i_all, 'i2t@5000': i2t,
                   't2i@5000': t2i,
                  })'''

        print('%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g' % ('Evaluation' if is_eval else 'Test',i2t, t2i, (i2t + t2i) / 2.))
        print('%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g' % ('Evaluation' if is_eval else 'Test',i2t_all, t2i_all, (i2t_all + t2i_all) / 2.))
        print('%s\nImg2Img: %g \t Img2Img: %g \t '% ('Evaluation' if is_eval else 'Test',i2i, i2i_all))
        if avg > best_acc:
            print('Saving..')
            state = {
                'image_model_state_dict': image_model.state_dict(),
                'text_model_state_dict': text_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Avg': avg,
                'Img2Txt': i2t,
                'Txt2Img': t2i,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(args.ckpt_dir, '%s_%d_best_checkpoint.t7' % (args.arch, args.bit)))
            best_acc = avg
        print("保存模型完成")
        return i2t, t2i

    lr_schedu.step(start_epoch)
    for epoch in range(start_epoch, args.max_epochs):
        image_model.train()
        text_model.train()
        image_model.set_alpha(epoch +1)
        text_model.set_alpha(epoch +1)
        train(epoch)
        print("训练完成")
        lr_schedu.step(epoch)
        i2t, t2i = test(epoch)
        print("测试完成")
        avg = (i2t + t2i) / 2.
        if avg == best_acc:
            image_model_state_dict = image_model.state_dict()
            image_model_state_dict = {key: image_model_state_dict[key].clone() for key in image_model_state_dict}
            text_model_state_dict = text_model.state_dict()
            text_model_state_dict = {key: text_model_state_dict[key].clone() for key in text_model_state_dict}
    print("保存字典完成")
    chp = torch.load(os.path.join(args.ckpt_dir, '%s_%d_best_checkpoint.t7' % (args.arch, args.bit)))
    print("导入最佳模型")
    image_model.load_state_dict(image_model_state_dict)
    text_model.load_state_dict(text_model_state_dict)
    print("保存最佳模型完成")
    test(chp['epoch'], is_eval=False)
    summary_writer.close()
    # pdb.set_trace()
    
    #wandb.finish()


def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    print(dist.shape)
    if k == 0:
        k = dist.shape[0]
        
    res = []
    res2 = []
    for i in range(numcases):
        order = ord[i].reshape(-1)[:3000]

        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    for i in range(numcases):
        order = ord[i].reshape(-1)[:5000]
        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res2 += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res),np.mean(res2)

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH
def calc_map_k_matrix(qB, rB, query_L, retrieval_L, k=None, rank=0):
    
    num_query = query_L.shape[0]
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        gnd = (query_L[iter].unsqueeze(0).mm(retrieval_L.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map
if __name__ == '__main__':
    main()