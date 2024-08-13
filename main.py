import os
import time
import numpy as np
import random
import argparse
import torch
from utils import helper
from model.trainer import TransKTtainer
from utils.dataloader import DataLoader
from utils.GraphMaker import GraphMaker
import os
from datetime import timedelta
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--warm_epoch', type=int, default=0, help='epochs for only predicate loss')

# TransKT model part
parser.add_argument('--dataset', type=str, default='C_DS', help='Java_Python,C_DS,C_Java,CA_DS')

# C&DS
parser.add_argument('--courseX_citem', type=int, default=362, help='')
parser.add_argument('--courseY_citem', type=int, default=323, help='')
parser.add_argument('--courseX_pitem', type=int, default=10957, help='') 
parser.add_argument('--courseY_pitem', type=int, default=9641, help='')
parser.add_argument('--cidnum', type=int, default=685, help='')
parser.add_argument('--pidnum', type=int, default=20598, help='')
parser.add_argument('--nitem', type=int, default=21285, help='')


parser.add_argument('--maxproblem', type=int, default=64,help='Upper bound on the number of questions')
parser.add_argument('--lambda', type=float, default=0.9,help='Regularization factor 𝜆')
parser.add_argument('--eta', type=float, default=0.7,help='Joint p-set level knowledge state factor 𝜂') 
parser.add_argument('--theta1', type=float, default=0.0,help='Threshold for negative sampling') 
parser.add_argument('--theta2', type=float, default=0.7,help='Threshold for negative sampling') 
parser.add_argument('--out_fc_dim', type=int, default=128, help='out_fc_dim.')
parser.add_argument('--out_fc_dim2', type=int, default=64, help='out_fc_dim2.')
parser.add_argument('--hidden_units', type=int, default=256, help='lantent dim.')
parser.add_argument('--d_ff', type=int, default=256, help='dimension for fully conntected net inside the basic block.')
parser.add_argument('--num_blocks', type=int, default=2, help='transformer layers.')
parser.add_argument('--num_heads', type=int, default=4, help='attention heads.')
parser.add_argument('--IM', type=int, default=1, help='if using Cross contrastive learning IM')
parser.add_argument('--kc_mat_path', type=str, default='dataset/C_DS/KC_P_Matrix.npy', help='Path for problem-KC matrix.')
parser.add_argument('--GNNL', type=int, default=1, help='GNN depth.')
parser.add_argument('--leakey', type=float, default=0.1)

# for ICL part
parser.add_argument('--ICL', type=int, default=0, help='ICL')
parser.add_argument('--num_cluster', type=int, default=64, help='num_cluster')
parser.add_argument("--temperature", default=0.5, type=float, help="softmax temperature (default:  1.0) - not studied.")

# single domain setting part
parser.add_argument('--XY', type=str, default='X', help='domain X or Y')
parser.add_argument('--model', type=str, default="CL4KT", help='baseline model name')

# train part              
parser.add_argument('--num_epoch', type=int, default=50, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
parser.add_argument('--seed', type=int, default=2125)
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax','adamw'], default='adamw',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--device', type=str, default="cuda", help='device')
parser.add_argument('--model_save_dir', type=str, default="trained", help='model name')
parser.add_argument('--model_id', type=str, default="pretrain", help='model name')
parser.add_argument('--semantic_emb', type=int, default=1,help='load semantic embedding or not')
def seed_everything(seed=2319): 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
opt = vars(args)


##### GCN
train_data='./dataset/'+opt['dataset']+'/train.jsonl'
G = GraphMaker(opt, train_data, opt['kc_mat_path'])
adj, adj_single = G.adj, G.adj_single
print("graph loaded!")  # 

if opt["device"]=="cuda":
    adj = adj.cuda()
    adj_single = adj_single.cuda()
#####


# logs for hyperparameters
helper.print_config(opt)


def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists")

create_folder(opt['model_save_dir']+'/'+str(opt['model_id']))

start_time = time.time()
global_x_acc=0
global_y_acc=0
global_x_auc=0
global_y_auc=0
seed_pools=[3407,42]


for seed in seed_pools:
    print("now seed is ",seed)
    print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
    seed_everything(seed)
    train_batch = DataLoader(opt, 0)
    eval_batch = DataLoader(opt, 1)
    test_batch = DataLoader(opt, 2)
    print("Data loading done!")
    trainer=TransKTtainer(opt,adj,adj_single,seed)
    batch_count=0
    best_auc_sum=0
    cot=0 
    
    # stage2: reload trainer
    for epoch in range(1, opt['num_epoch'] + 1):
        
        # ---------- STEP:train -------------#
        train_loss_all = 0
        train_loss_pre=0
        train_loss_im=0
        train_loss_nce=0
        train_loss_icl=0
        traim_loss_ssl=0
        num_batch = len(train_batch)
        for idx,batch in enumerate(train_batch):
            loss,loss_pre,loss_im= trainer.train_batch(batch,epoch=epoch)
            train_loss_all += loss
            train_loss_pre+=loss_pre
            train_loss_im+=loss_im
            batch_count+=1
            
        print("epoch:",epoch,"train_loss:",train_loss_all/num_batch,"predicate_loss:",train_loss_pre/num_batch*(opt['lambda']),"im_loss:",train_loss_im/num_batch*(1-opt['lambda']))

        num_batch = len(eval_batch)
        sum_x_acc = 0
        sum_y_acc = 0
        sum_x_auc = 0
        sum_y_auc = 0
        trainer.model.eval()
        with torch.no_grad():
            for t_batch in eval_batch:
                acc_x, acc_y, auc_x,auc_y,rmse_x, rmse_y=trainer.eval_batch(t_batch)
                sum_x_acc += acc_x
                sum_y_acc += acc_y
                sum_x_auc += auc_x
                sum_y_auc += auc_y
                
            now_x_auc=sum_x_auc/num_batch
            now_y_auc=sum_y_auc/num_batch
            now_x_acc=sum_x_acc/num_batch  
            now_y_acc=sum_y_acc/num_batch

        print("Time elapsed: {}".format(timedelta(seconds=time.time()-start_time)))
        print("epoch{}: x_acc: {:.6f}, y_acc: {:.6f}, x_auc: {:.6f}, y_auc: {:.6f}".format(epoch,now_x_acc, now_y_acc,now_x_auc, now_y_auc))
        
        if best_auc_sum<now_x_auc+now_y_auc:
            print("best auc update!!!!")
            best_auc_sum=now_x_auc+now_y_auc
            trainer.save(filename=opt['model_save_dir']+'/'+str(opt['model_id'])+"/best_auc.pt")
            cot=0
        cot+=1
        if cot>10:
            break
    
    # ---------- STEP:testing -------------#
    trainer.load(filename=opt['model_save_dir']+'/'+str(opt['model_id'])+"/best_auc.pt")
    print("start testing !")
    
    with torch.no_grad():
        sum_x_acc = 0
        sum_y_acc = 0
        sum_x_auc = 0
        sum_y_auc = 0
        num_batch=len(test_batch)
        for t_batch in test_batch:
            acc_x, acc_y, auc_x,auc_y,rmse_x, rmse_y=trainer.eval_batch(t_batch)
            sum_x_acc += acc_x
            sum_y_acc += acc_y
            sum_x_auc += auc_x
            sum_y_auc += auc_y

        now_x_acc=sum_x_acc/num_batch
        now_y_acc=sum_y_acc/num_batch
        now_x_auc=sum_x_auc/num_batch
        now_y_auc=sum_y_auc/num_batch
        # print("Time elapsed: {}".format(timedelta(seconds=time.time()-start_time)))
        print("test: x_acc: {:.6f}, y_acc: {:.6f}, x_auc: {:.6f}, y_auc: {:.6f}".format(now_x_acc, now_y_acc,now_x_auc, now_y_auc))
        global_x_acc+=now_x_acc
        global_y_acc+=now_y_acc
        global_x_auc+=now_x_auc
        global_y_auc+=now_y_auc
        
print("globaltest->  test: x_acc: {:.6f}, y_acc: {:.6f}, x_auc: {:.6f}, y_auc: {:.6f}".format(global_x_acc/len(seed_pools), global_y_acc/len(seed_pools),global_x_auc/len(seed_pools), global_y_auc/len(seed_pools)))