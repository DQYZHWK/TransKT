import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
import copy
from model.SimpleKT_Encoder import SimpleKT_Encoder
from model.GNN import GCNLayer

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransKT(torch.nn.Module):
    def __init__(self, opt,adj,adj_single):
        super(TransKT, self).__init__()
        self.opt = opt
        self.adj = adj
        self.adj_single = adj_single
        
        
        # self.merge_size=self.opt["merge_size"]
        # TODO embdding 
        # concept embedding(c)
        self.cid_emb = torch.nn.Embedding(self.opt["cidnum"]+1, 256,padding_idx=self.opt["cidnum"])
        self.cid_emb_X = torch.nn.Embedding(self.opt["cidnum"]+1, 256,padding_idx=self.opt["cidnum"])
        self.cid_emb_Y = torch.nn.Embedding(self.opt["cidnum"]+1, 256, padding_idx=self.opt["cidnum"])
        
        # problem difficulty embedding(mq)
        self.p_diffculty_emb = torch.nn.Embedding(self.opt["pidnum"]+1,256,padding_idx=self.opt["pidnum"])
        self.p_diffculty_emb_X = torch.nn.Embedding(self.opt["pidnum"]+1,256,padding_idx=self.opt["pidnum"])
        self.p_diffculty_emb_Y = torch.nn.Embedding(self.opt["pidnum"]+1,256,padding_idx=self.opt["pidnum"])
        
        # response embedding(r)
        self.response_emb = torch.nn.Embedding(3, self.opt["hidden_units"],padding_idx=2)
        self.response_emb_X = torch.nn.Embedding(3, self.opt["hidden_units"],padding_idx=2)
        self.response_emb_Y = torch.nn.Embedding(3, self.opt["hidden_units"],padding_idx=2)
        
        KC_MAT=np.load(self.opt["kc_mat_path"])
        self.KC_MAT=torch.from_numpy(KC_MAT).cuda()
        
        self.seq_len = opt['maxlen']
        self.emb_size =opt['hidden_units']
        self.num_attn_heads = opt['num_heads']
        self.dropout = opt['dropout']
        self.num_en = opt['num_blocks']
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.pred_x = torch.nn.Linear(self.opt['hidden_units'], 1)  
        self.pred_y = torch.nn.Linear(self.opt['hidden_units'], 1)
        self.position_emb = torch.nn.Embedding(self.seq_len+1, self.emb_size)
        

        self.prototype_att_x=torch.nn.Linear(self.opt['hidden_units'],1,bias=False)
        self.prototype_att_y=torch.nn.Linear(self.opt['hidden_units'],1,bias=False)
        self.prototype_att=torch.nn.Linear(self.opt['hidden_units'],1,bias=False)
        
        self.attention_dense = nn.Linear(self.opt['hidden_units'], 1, bias=False)
        self.attention_dense_x = nn.Linear(self.opt['hidden_units'], 1, bias=False)
        self.attention_dense_y = nn.Linear(self.opt['hidden_units'], 1, bias=False)
        self.attention_dense_qa = nn.Linear(self.opt['hidden_units'], 1, bias=False)
        self.attention_dense_qa_x = nn.Linear(self.opt['hidden_units'], 1, bias=False)
        self.attention_dense_qa_y = nn.Linear(self.opt['hidden_units'], 1, bias=False)
        
        self.out_fc_dim=self.opt["out_fc_dim"]
        self.out_fc_dim_2=self.opt["out_fc_dim2"]
        self.dropout=self.opt["dropout"]
        
        self.out = nn.Sequential(
            nn.Linear(2*self.opt["hidden_units"],self.out_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.out_fc_dim, self.out_fc_dim_2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(self.out_fc_dim_2, 1)
        )
        
        
        self.out_X = nn.Sequential(
            nn.Linear(2*self.opt["hidden_units"],self.out_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.out_fc_dim, self.out_fc_dim_2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(self.out_fc_dim_2, 1)
        )
        
        
        self.out_Y = nn.Sequential(
            nn.Linear(2*self.opt["hidden_units"],self.out_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.out_fc_dim, self.out_fc_dim_2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(self.out_fc_dim_2, 1)
        )

        self.encoder=SimpleKT_Encoder(opt)
        self.encoder_X=SimpleKT_Encoder(opt)
        self.encoder_Y=SimpleKT_Encoder(opt)
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.D_X = torch.nn.Bilinear(self.opt["hidden_units"],self.opt["hidden_units"],1,bias=False)
        self.D_Y = torch.nn.Bilinear(self.opt["hidden_units"],self.opt["hidden_units"],1,bias=False)
        
        self.pitem_x_index = torch.arange(0, self.opt["courseX_pitem"], 1)
        self.pitem_y_index = torch.arange(self.opt["courseX_pitem"], self.opt["courseX_pitem"]+self.opt["courseY_pitem"], 1)
        self.pitem_index = torch.arange(0, self.opt["pidnum"]+1, 1)
        self.citem_x_index = torch.arange(0, self.opt["courseX_citem"], 1)
        self.citem_y_index = torch.arange(self.opt["courseX_citem"], self.opt["courseX_citem"]+self.opt["courseY_citem"], 1)
        self.citem_index = torch.arange(0, self.opt["cidnum"]+1, 1)
        
        self.GNN_encoder_X = GCNLayer(opt)
        self.GNN_encoder_Y = GCNLayer(opt)
        self.GNN_encoder = GCNLayer(opt)
        
        
        if self.opt["device"]=="cuda":
            self.pitem_x_index=self.pitem_x_index.cuda()
            self.pitem_y_index=self.pitem_y_index.cuda()
            self.pitem_index=self.pitem_index.cuda()
            self.citem_x_index=self.citem_x_index.cuda()
            self.citem_y_index=self.citem_y_index.cuda()
            self.citem_index=self.citem_index.cuda()
    
    def shift(self,emb,index):
        batch_size = emb.size(0)
        maxlen = emb.size(1)
        first_mask=torch.ones((batch_size,maxlen), dtype=torch.bool)
        first_mask[range(batch_size),index]=False
        emb=emb[first_mask].view(batch_size,maxlen-1,-1)
        return emb
    def dense_attention(self,p_emb,p_x_emb,p_y_emb,set_mask,set_x_mask,set_y_mask):
        dense_dim=self.attention_dense(p_emb).squeeze(-1)
        dense_dim_x=self.attention_dense_x(p_x_emb).squeeze(-1)
        dense_dim_y=self.attention_dense_y(p_y_emb).squeeze(-1)
        masked_dense_dim = dense_dim.masked_fill(set_mask == 0, float('-inf'))
        x_masked_dense_dim = dense_dim_x.masked_fill(set_x_mask == 0, float('-inf'))
        y_masked_dense_dim = dense_dim_y.masked_fill(set_y_mask == 0, float('-inf'))
        masked_dense_weights = F.softmax(masked_dense_dim, dim=-1)
        x_masked_dense_weights = F.softmax(x_masked_dense_dim, dim=-1)
        y_masked_dense_weights = F.softmax(y_masked_dense_dim, dim=-1)
        masked_dense_weights = torch.where(set_mask == 0, torch.tensor(0.0).cuda(), masked_dense_weights)
        x_masked_dense_weights = torch.where(set_x_mask == 0, torch.tensor(0.0).cuda(), x_masked_dense_weights)
        y_masked_dense_weights = torch.where(set_y_mask == 0, torch.tensor(0.0).cuda(), y_masked_dense_weights)

        return masked_dense_weights,x_masked_dense_weights,y_masked_dense_weights   
    def dense_attention2(self,p_emb,set_mask):
        dense_dim=self.attention_dense(p_emb).squeeze(-1)
        masked_dense_dim = dense_dim.masked_fill(set_mask == 0, float('-inf'))
        masked_dense_weights = F.softmax(masked_dense_dim, dim=-1)
        masked_dense_weights = torch.where(set_mask == 0, torch.tensor(0.0).cuda(), masked_dense_weights)

        return masked_dense_weights
    def cro(self,re_p,re_r,set_mask,x_first,y_first,x_last,y_last):
        # p_emb=torch.cat([self.p_diffculty_emb(re_p),(torch.einsum('ij,jk->ik',self.KC_MAT.float(),self.cid_emb.weight.data)/(torch.sum(self.KC_MAT,dim=1,keepdim=True)+1e-8))[re_p]],dim=-1)
        p_emb=self.my_index_select(self.cross_emb, re_p)
        cro_pa=p_emb+self.response_emb(re_r)
        masked_dense_weights=self.dense_attention2(p_emb,set_mask)
        basket_pemb=torch.einsum('bmnk,bmn->bmk', p_emb, masked_dense_weights)
        basket_pa_emb=torch.einsum('bmnk,bmn->bmk', cro_pa, masked_dense_weights)
        
        CH=self.encoder(basket_pemb,basket_pa_emb)
        return CH
    
    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans


    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans
    
    
    def graph_convolution(self):
        pfea = self.my_index_select_embedding(self.p_diffculty_emb, self.pitem_index)
        pfea_X = self.my_index_select_embedding(self.p_diffculty_emb_X, self.pitem_index)
        pfea_Y = self.my_index_select_embedding(self.p_diffculty_emb_Y, self.pitem_index)
        
        
        cfea=self.my_index_select_embedding(self.cid_emb, self.citem_index)
        cfea_X=self.my_index_select_embedding(self.cid_emb_X, self.citem_index)
        cfea_Y=self.my_index_select_embedding(self.cid_emb_Y, self.citem_index)
        
        
        fea = torch.cat((pfea, cfea), dim=0)
        fea_X=torch.cat((pfea_X, cfea_X), dim=0)
        fea_Y=torch.cat((pfea_Y, cfea_Y), dim=0)
        # torch.Size([64094, 256])
        self.cross_emb = self.GNN_encoder(fea, self.adj)
        self.single_emb_X = self.GNN_encoder_X(fea_X, self.adj_single)
        self.single_emb_Y = self.GNN_encoder_Y(fea_Y, self.adj_single)
    
    
     
    def workflow_merge_basket(self,set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,first,last,x_first,x_last,y_first,y_last,ps_mask,ps_x_mask,ps_y_mask):
        
        
        # setting1 (good)
        p_emb=self.my_index_select(self.cross_emb, set_p)
        p_x_emb=self.my_index_select(self.single_emb_X, set_px)
        p_y_emb=self.my_index_select(self.single_emb_Y, set_py)
      
        
        # setting 0
        # p_emb=self.p_diffculty_emb(set_p)+(torch.einsum('ij,jk->ik',self.KC_MAT.float(),self.cid_emb.weight.data)/(torch.sum(self.KC_MAT,dim=1,keepdim=True)+1e-8))[set_p]
        # p_x_emb=self.p_diffculty_emb_X(set_p)+(torch.einsum('ij,jk->ik',self.KC_MAT.float(),self.cid_emb_X.weight.data)/(torch.sum(self.KC_MAT,dim=1,keepdim=True)+1e-8))[set_px]
        # p_y_emb=self.p_diffculty_emb_Y(set_p)+(torch.einsum('ij,jk->ik',self.KC_MAT.float(),self.cid_emb_Y.weight.data)/(torch.sum(self.KC_MAT,dim=1,keepdim=True)+1e-8))[set_py]
        
        pa_emb=self.response_emb(set_r)+p_emb
        pa_x_emb=self.response_emb_X(set_rx)+p_x_emb
        pa_y_emb=self.response_emb_Y(set_ry)+p_y_emb
        
        masked_dense_weights,x_masked_dense_weights,y_masked_dense_weights=self.dense_attention(p_emb,p_x_emb,p_y_emb,set_mask,set_x_mask,set_y_mask)
        basket_emb=torch.einsum('bmnk,bmn->bmk', p_emb, masked_dense_weights)
        basket_x_emb=torch.einsum('bmnk,bmn->bmk', p_x_emb, x_masked_dense_weights)
        basket_y_emb=torch.einsum('bmnk,bmn->bmk', p_y_emb, y_masked_dense_weights)
        
        basket_ans_emb=torch.einsum('bmnk,bmn->bmk', pa_emb, masked_dense_weights)
        basket_x_ans_emb=torch.einsum('bmnk,bmn->bmk', pa_x_emb, x_masked_dense_weights)
        basket_y_ans_emb=torch.einsum('bmnk,bmn->bmk', pa_y_emb, y_masked_dense_weights)
        H=self.encoder(basket_emb,basket_ans_emb)
        H_X=self.encoder_X(basket_x_emb,basket_x_ans_emb)
        H_Y=self.encoder_Y(basket_y_emb,basket_y_ans_emb)
        return H,H_X,H_Y,p_emb,p_x_emb,p_y_emb 
    
    
    def forward(self,set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,first,last,x_first,x_last,y_first,y_last,ps_mask,ps_x_mask,ps_y_mask):
        H,H_X,H_Y,p_emb,p_x_emb,p_y_emb =self.workflow_merge_basket(set_p,set_px,set_py,set_r,set_rx,set_ry,set_mask,set_x_mask,set_y_mask,first,last,x_first,x_last,y_first,y_last,ps_mask,ps_x_mask,ps_y_mask)
        return H,H_X,H_Y,p_emb,p_x_emb,p_y_emb