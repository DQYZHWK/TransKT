import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from utils import torch_utils
from model.TransKT import TransKT
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import pdb
import copy
import numpy as np
from model.modules import NCELoss

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics._regression")

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr): 
        torch_utils.change_lr(self.optimizer, new_lr)

    def reload(self, filename):  
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        for param in self.model.cid_emb.parameters():
            param.requires_grad = False
            
        for param in self.model.p_diffculty_emb.parameters():
            param.requires_grad = False    
            
        for param in self.model.GNN_encoder.parameters():
            param.requires_grad = False
            
        for param in self.model.response_emb.parameters():
            param.requires_grad = False
            
        for param in self.model.attention_dense.parameters():
            param.requires_grad = False
            
        for param in self.model.encoder.parameters():
            param.requires_grad = False
            
        for name, param in self.model.named_parameters():
            print(f"{name} requires_grad: {param.requires_grad}")

    
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class TransKTtainer(Trainer):
    def __init__(self, opt,adj,adj_single,seed):
        self.opt = opt
        self.model =TransKT(opt,adj,adj_single)
        self.mi_loss = 0
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        self.NCE_criterion = NCELoss(self.opt['temperature'],self.opt['device'])
        if opt['device']=="cuda":
            self.model.cuda()
            self.BCE_criterion.cuda()
            self.CS_criterion.cuda()
            self.NCE_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'],opt['weight_decay'])

    def unpack_batch_ps(self, batch):
        if self.opt["device"]=="cuda":
            inputs = [Variable(b.cuda()) for b in batch]
            set_p=inputs[0]
            set_px=inputs[1]
            set_py=inputs[2]
            set_r=inputs[3]
            set_rx=inputs[4]
            set_ry=inputs[5]
            set_mask=inputs[6]
            set_x_mask=inputs[7]
            set_y_mask=inputs[8]
            re_p=inputs[9]
            re_r=inputs[10]
            first=inputs[11]
            last=inputs[12]
            x_first=inputs[13]
            x_last=inputs[14]
            y_first=inputs[15]
            y_last=inputs[16]
        else:
            inputs = [Variable(b) for b in batch]
            set_p=inputs[0]
            set_px=inputs[1]
            set_py=inputs[2]
            set_r=inputs[3]
            set_rx=inputs[4]
            set_ry=inputs[5]
            set_mask=inputs[6]
            set_x_mask=inputs[7]
            set_y_mask=inputs[8]
            re_p=inputs[9]
            re_r=inputs[10]
            first=inputs[11]
            last=inputs[12]
            x_first=inputs[13]
            x_last=inputs[14]
            y_first=inputs[15]
            y_last=inputs[16]
            
        return set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,re_p,re_r,first,last,x_first,x_last,y_first,y_last

    def shift(self,emb,index):
        batch_size = emb.size(0)
        maxlen = emb.size(1)
        first_mask=torch.ones((batch_size,maxlen), dtype=torch.bool)
        first_mask[range(batch_size),index]=False
        emb=emb[first_mask].view(batch_size,maxlen-1,emb.size(2),emb.size(3))
        return emb
    
    def shift2(self,emb,index):
        batch_size = emb.size(0)
        maxlen = emb.size(1)
        first_mask=torch.ones((batch_size,maxlen), dtype=torch.bool)
        first_mask[range(batch_size),index]=False
        emb=emb[first_mask].view(batch_size,maxlen-1,emb.size(2))
        return emb

    def shift3(self,emb,index):
        batch_size = emb.size(0)
        maxlen = emb.size(1)
        first_mask=torch.ones((batch_size,maxlen), dtype=torch.bool)
        first_mask[range(batch_size),index]=False
        emb=emb[first_mask].view(batch_size,maxlen-1)
        return emb

    def IM_loss(self,prototype_x,prototype_y,cross_x,cross_y,cro_x,cro_y):
        x_pos_1=prototype_x
        x_pos_2=cross_x
        y_pos_1=prototype_y
        y_pos_2=cross_y
        
        x_neg1=prototype_x
        x_neg2=cro_x
        y_neg1=prototype_y
        y_neg2=cro_y
        m=nn.Sigmoid()
        x_pos_score=m(self.model.D_X(x_pos_1,x_pos_2))
        y_pos_score=m(self.model.D_Y(y_pos_1,y_pos_2))
        
        x_neg_score=m(self.model.D_X(x_neg1,x_neg2))
        y_neg_score=m(self.model.D_Y(y_neg1,y_neg2))
        pos_label = torch.ones_like(x_pos_score).cuda()
        neg_label = torch.zeros_like(x_neg_score).cuda()
        loss_pos = self.BCE_criterion(x_pos_score,pos_label)+self.BCE_criterion(y_pos_score,pos_label)
        loss_neg = self.BCE_criterion(x_neg_score,neg_label)+self.BCE_criterion(y_neg_score,neg_label)
        return loss_neg+loss_pos
    
    def get_prototype(self,H,H_X,H_Y,NEG_H,first,x_first,y_first,mask,x_mask,y_mask):
        single_x_score=self.model.prototype_att(H_X).squeeze(-1)
        single_y_score=self.model.prototype_att(H_Y).squeeze(-1)
        cross_score=self.model.prototype_att(H).squeeze(-1)
        cro_score=self.model.prototype_att(NEG_H).squeeze(-1)
        
        single_x_score = single_x_score.masked_fill(x_mask == 0, float('-inf'))
        single_y_score = single_y_score.masked_fill(y_mask == 0, float('-inf'))
        cross_score = cross_score.masked_fill(mask == 0, float('-inf'))
        cro_score = cro_score.masked_fill(mask == 0, float('-inf'))
        
        cross_x_score=cross_score.masked_fill(x_mask == 0, float('-inf'))
        cross_y_score=cross_score.masked_fill(y_mask == 0, float('-inf'))
        cro_x_score=cro_score.masked_fill(x_mask == 0, float('-inf'))
        cro_y_score=cro_score.masked_fill(y_mask == 0, float('-inf'))
        
        single_x_weights=F.softmax(single_x_score,dim=-1)
        single_y_weights=F.softmax(single_y_score,dim=-1)
        cross_weights=F.softmax(cross_score,dim=-1)
        cro_weights=F.softmax(cro_score,dim=-1)
        
        cross_x_wei=F.softmax(cross_x_score,dim=-1)
        cross_y_wei=F.softmax(cross_y_score,dim=-1)
        cro_x_wei=F.softmax(cro_x_score,dim=-1)
        cro_y_wei=F.softmax(cro_y_score,dim=-1)
        
        single_x_weights = torch.where(x_mask == 0, torch.tensor(0.0).cuda(), single_x_weights)
        single_y_weights = torch.where(y_mask == 0, torch.tensor(0.0).cuda(), single_y_weights)
        cross_weights = torch.where(mask == 0, torch.tensor(0.0).cuda(), cross_weights)
        cro_weights = torch.where(mask == 0, torch.tensor(0.0).cuda(), cro_weights)
        cross_x_wei = torch.where(x_mask == 0, torch.tensor(0.0).cuda(), cross_x_wei)
        cross_y_wei = torch.where(y_mask == 0, torch.tensor(0.0).cuda(), cross_y_wei)
        cro_x_wei = torch.where(x_mask == 0, torch.tensor(0.0).cuda(), cro_x_wei)
        cro_y_wei = torch.where(y_mask == 0, torch.tensor(0.0).cuda(), cro_y_wei)
        
        prototype_x = torch.einsum('bmk,bm->bk',H_X,single_x_weights) 
        prototype_y = torch.einsum('bmk,bm->bk',H_Y,single_y_weights)
        cross_prototype= torch.einsum('bmk,bm->bk',H,cross_weights)
        cro_prototype= torch.einsum('bmk,bm->bk',NEG_H,cro_weights)
        
        
        cross_x_prototype= torch.einsum('bmk,bm->bk',H,cross_x_wei)
        cross_y_prototype= torch.einsum('bmk,bm->bk',H,cross_y_wei)
        cro_x_prototype= torch.einsum('bmk,bm->bk',NEG_H,cro_x_wei)
        cro_y_prototype= torch.einsum('bmk,bm->bk',NEG_H,cro_y_wei)


        return prototype_x,prototype_y,cross_prototype,cro_prototype,cross_x_prototype,cross_y_prototype,cro_x_prototype,cro_y_prototype

        
    def train_batch(self, batch,do_icl=True,epoch=0):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.graph_convolution()
        set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,re_p,re_r,first,last,x_first,x_last,y_first,y_last=self.unpack_batch_ps(batch)
        H,H_X,H_Y,p_emb,p_x_emb,p_y_emb =self.model(set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,first,last,x_first,x_last,y_first,y_last)
        
        
        NEG_H=self.model.cro(re_p,re_r,set_mask,x_first,y_first,x_last,y_last)
        
        l2=self.opt['eta']
        maxproblem=self.opt['maxproblem']
        
        
        single_x=(l2*H_X+(1-l2)*H)
        single_y=(l2*H_Y+(1-l2)*H)
        
        concat_x=torch.cat([single_x,p_x_emb],dim=-1)
        concat_y=torch.cat([single_y,p_y_emb],dim=-1)
        predicate_x=self.model.out_X(concat_x).squeeze(-1)   
        predicate_y=self.model.out_Y(concat_y).squeeze(-1)
        
        
        # prototype 
        prototype_x,prototype_y,cross_prototype,cro_prototype,cross_x,cross_y,cro_x,cro_y=self.get_prototype(H,H_X,H_Y,NEG_H,first,x_first,y_first,set_mask,set_x_mask,set_y_mask)

        
        # predicate (r_2,...,r_n)
        predicate_x=self.shift3(predicate_x,x_first)
        predicate_y=self.shift3(predicate_y,y_first)
        set_rx=self.shift3(set_rx,x_first)
        set_ry=self.shift3(set_ry,y_first)
        set_x_mask=self.shift3(set_x_mask,x_first)
        set_y_mask=self.shift3(set_y_mask,y_first)
        
        predicat_x_mask_select=torch.masked_select(predicate_x,set_x_mask.bool())
        x_ground_mask_select=torch.masked_select(set_rx,set_x_mask.bool())
        predicat_y_mask_select=torch.masked_select(predicate_y,set_y_mask.bool())
        y_ground_mask_select=torch.masked_select(set_ry,set_y_mask.bool())
        loss1 =  self.BCE_criterion(predicat_x_mask_select, x_ground_mask_select.float())+self.BCE_criterion(predicat_y_mask_select, y_ground_mask_select.float())
        l1=self.opt['lambda']
        loss=loss1*l1
     
        
        if self.opt['IM']==1:
            im_loss=self.IM_loss(prototype_x,prototype_y,cross_x,cross_y,cro_x,cro_y)
            loss=loss+im_loss*(1-l1)
            loss2=im_loss.item()
        elif self.opt['ICL']==1:
            icl_loss=self.ICL_loss(cross_prototype)
            loss=loss+icl_loss*(1-l1)
            loss2=icl_loss.item()
        else:
            loss2=0
        
        loss.backward()
        self.optimizer.step()

        
        return loss.item(),loss1.item(),loss2

    def eval_batch(self, batch):
        self.model.eval()
        l2=self.opt['eta']
        maxproblem=self.opt['maxproblem']
        set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,re_p,re_r,first,last,x_first,x_last,y_first,y_last=self.unpack_batch_ps(batch)
        H,H_X,H_Y,p_emb,p_x_emb,p_y_emb =self.model(set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,first,last,x_first,x_last,y_first,y_last)
        
        single_x=(l2*H_X+(1-l2)*H)
        single_y=(l2*H_Y+(1-l2)*H)
        
        concat_x=torch.cat([single_x,p_x_emb],dim=-1)
        concat_y=torch.cat([single_y,p_y_emb],dim=-1)
        
        
        MX=nn.Sigmoid()
        MY=nn.Sigmoid()
        predicate_x=MX(self.model.out_X(concat_x).squeeze(-1))
        predicate_y=MY(self.model.out_Y(concat_y).squeeze(-1))
        

        
        predicat_x_mask_select=torch.masked_select(predicate_x,set_x_mask.bool()).cpu().detach().numpy()
        x_ground_mask_select=torch.masked_select(set_rx,set_x_mask.bool()).cpu().detach().numpy()
        predicat_y_mask_select=torch.masked_select(predicate_y,set_y_mask.bool()).cpu().detach().numpy()
        y_ground_mask_select=torch.masked_select(set_ry,set_y_mask.bool()).cpu().detach().numpy()
        
        acc_x = accuracy_score(x_ground_mask_select, predicat_x_mask_select>0.5)
        acc_y = accuracy_score(y_ground_mask_select, predicat_y_mask_select>0.5)
        
        
        auc_x = roc_auc_score(x_ground_mask_select, predicat_x_mask_select)
        auc_y = roc_auc_score(y_ground_mask_select, predicat_y_mask_select)
        
        rmse_x = mean_squared_error(x_ground_mask_select, predicat_x_mask_select,squared=False)
        rmse_y = mean_squared_error(y_ground_mask_select, predicat_y_mask_select,squared=False)

        return acc_x, acc_y, auc_x,auc_y,rmse_x, rmse_y