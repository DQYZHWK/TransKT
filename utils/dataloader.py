"""
Data loader for CCPSKT dataset.
"""
import os
import json
import random
import torch
import numpy as np
import codecs
import sys
from io import StringIO

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self,opt,train_dev_test=0):
        self.batch_size = opt['batch_size']
        self.opt = opt
        self.eval = train_dev_test
        current_directory = os.getcwd()
        files = os.listdir(current_directory)
        for file in files:
            print(file)

        # ************* sequential data *****************
        dataset = opt['dataset']
        source_train_data = "dataset/"+dataset+"/train.jsonl"
        source_valid_data = "dataset/"+dataset+"/dev.jsonl"
        source_test_data = "dataset/"+dataset+"/test.jsonl"
        
        # self.merge_size=self.opt['merge_size']
        self.courseX_citem=self.opt['courseX_citem']
        self.courseY_citem=self.opt['courseY_citem']
        self.all_citem=self.opt['cidnum']
        
        self.courseX_pitem=self.opt['courseX_pitem']
        self.courseY_pitem=self.opt['courseY_pitem']
        self.all_pitem=self.opt['pidnum']
        batch_size=self.opt['batch_size']
        eval = self.eval
        if eval==0:
            self.probelmset_data = self.read_data(source_train_data)
            data = self.mypreprocess(self.probelmset_data)
        elif eval == 1:
            self.probelmset_data = self.read_data(source_valid_data)
            data = self.mypreprocess(self.probelmset_data)
        elif eval == 2:
            self.probelmset_data = self.read_data(source_test_data)
            data = self.mypreprocess(self.probelmset_data)
        else:
            print("eval value error")
            sys.exit()
        # shuffle for training
        if eval == 0:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        else :
            
            indices = list(range(len(data)))
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
            
        self.num_examples = len(data)

        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        
    def read_item(self, fname):
        item_number = 0
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            for line in fr:
                item_number += 1
        return item_number

    def read_data(self,train_file):
        # [[seq1],[seq2],...]
        # seq1=(excise1,excise2,...)
        
        with open(train_file, 'r') as file:
            lines = file.readlines()
            row_data = [json.loads(line.strip()) for line in lines]
        
        train_data=[]
        for line in row_data:
            seq=[]
            for item in line['records']:
                seq.append(item['p-set_log'])
            train_data.append(seq)
        
        return train_data
    
    def find_first_last_one_indices(self,lst):
        first_one_index = lst.index(1) if 1 in lst else None
        last_one_index = len(lst) - 1 - lst[::-1].index(1) if 1 in lst else None
        return first_one_index, last_one_index
    
    def problemsetlevel_process_aug (self,problemset_data):
        # 1.limit maxproblem problem every testset
        # 2.padding
        # 3.mask
        maxproblem=self.opt['maxproblem']
        problemsetlevel_data=[]
        maxlen=self.opt['maxlen']
        theta1=self.opt['theta1']
        theta2=self.opt['theta2'] 
        rank_dir="dataset/"+self.opt['dataset']+"/problem_diff_rank.jsonl"
        rank_dict=json.load(open(rank_dir))
        rank_array_str=list(rank_dict.keys())
        rank_array = [int(x) for x in rank_array_str]
        for excise_list in problemset_data:
            if len(excise_list)>maxlen:
                continue
            p2=[]
            r2=[]
            mask2=[]
    
            px2=[]
            py2=[]
            rx2=[]
            ry2=[]
            x_mask2=[]
            y_mask2=[]
            ps_mask=[]
            ps_x_mask=[]
            ps_y_mask=[]
            
            
            re_p2=[]
            re_r2=[]
            
            
            for excise in excise_list:
                ps_mask.append(1)
                if excise[0]['local_pid']<self.opt['courseX_pitem']:
                    ps_x_mask.append(1)
                    ps_y_mask.append(0)
                else:
                    ps_x_mask.append(0)
                    ps_y_mask.append(1)
                
                p=[]
                r=[]
                mask=[]
                px=[]
                py=[]
                rx=[]
                ry=[]
                x_mask=[]
                y_mask=[]
                
                re_p=[]
                re_r=[]
                if len(excise)>maxproblem:
                    excise=excise[:maxproblem]  
                rand = random.random()
                for item in excise:
                    p.append(item['local_pid'])
                    r.append(item['response'])
                    mask.append(1) 

                    rank=rank_dict[str(item['local_pid'])]
                    if rand<theta1: # RF
                        re_r.append(1-item['response'])  
                        re_p.append(item['local_pid'])
                        
                    elif rand<theta2 and rand>=theta1: # IR
                        re_r.append(1-item['response'])
                        if item['response']==1:# easier problem
                            if item['local_pid']<self.opt['courseX_pitem']:
                                re_p.append(random.choice(rank_array[0:rank]))
                            else:
                                re_p.append(random.choice(rank_array[self.opt['courseX_pitem']:rank]))
                        else: # more harder problem
                            if item['local_pid']<self.opt['courseX_pitem']:
                                re_p.append(random.choice(rank_array[rank:self.opt['courseX_pitem']]))
                            else:
                                re_p.append(random.choice(rank_array[rank:]))
                        # if item['local_pid']<self.opt['courseX_pitem']:
                        #     re_p.append(random.choice(range(0,self.opt['courseX_pitem'])))
                        # else:
                        #     re_p.append(random.choice(range(self.opt['courseX_pitem'],self.opt['pidnum'])))
                    else:
                        re_r.append(item['response'])
                        re_p.append(item['local_pid'])
                            
                            
                    if item['local_pid']<self.opt['courseX_pitem']:
                        px.append(item['local_pid'])
                        py.append(self.all_pitem)
                        rx.append(item['response'])
                        ry.append(2)
                        x_mask.append(1)
                        y_mask.append(0)

                    else:
                        px.append(self.all_pitem)
                        py.append(item['local_pid'])
                        rx.append(2)
                        ry.append(item['response'])
                        x_mask.append(0)
                        y_mask.append(1)
                
                p=[self.all_pitem]*(maxproblem-len(excise))+p
                px=[self.all_pitem]*(maxproblem-len(excise))+px       
                py=[self.all_pitem]*(maxproblem-len(excise))+py
                r=[2]*(maxproblem-len(excise))+r
                rx=[2]*(maxproblem-len(excise))+rx
                ry=[2]*(maxproblem-len(excise))+ry
                mask=[0]*(maxproblem-len(excise))+mask
                x_mask=[0]*(maxproblem-len(excise))+x_mask
                y_mask=[0]*(maxproblem-len(excise))+y_mask
                
                re_r=[2]*(maxproblem-len(excise))+re_r
                re_p=[self.all_pitem]*(maxproblem-len(excise))+re_p
                
                p2.append(p)
                r2.append(r)
                mask2.append(mask)
                px2.append(px)
                py2.append(py)
                rx2.append(rx)
                ry2.append(ry)
                x_mask2.append(x_mask)
                y_mask2.append(y_mask)
            
                re_r2.append(re_r)
                re_p2.append(re_p)
            p2=[[self.all_pitem]*maxproblem]*(maxlen-len(excise_list))+p2
            px2=[[self.all_pitem]*maxproblem]*(maxlen-len(excise_list))+px2
            py2=[[self.all_pitem]*maxproblem]*(maxlen-len(excise_list))+py2
            r2=[[2]*maxproblem]*(maxlen-len(excise_list))+r2
            rx2=[[2]*maxproblem]*(maxlen-len(excise_list))+rx2
            ry2=[[2]*maxproblem]*(maxlen-len(excise_list))+ry2
            mask2=[[0]*maxproblem]*(maxlen-len(excise_list))+mask2
            x_mask2=[[0]*maxproblem]*(maxlen-len(excise_list))+x_mask2
            y_mask2=[[0]*maxproblem]*(maxlen-len(excise_list))+y_mask2
            
            ps_mask=[0]*(maxlen-len(excise_list))+ps_mask
            ps_x_mask=[0]*(maxlen-len(excise_list))+ps_x_mask
            ps_y_mask=[0]*(maxlen-len(excise_list))+ps_y_mask
            
            re_r2=[[2]*maxproblem]*(maxlen-len(excise_list))+re_r2
            re_p2=[[self.all_pitem]*maxproblem]*(maxlen-len(excise_list))+re_p2
            
            first,last=self.find_first_last_one_indices(ps_mask)
            x_first,x_last=self.find_first_last_one_indices(ps_x_mask)
            y_first,y_last=self.find_first_last_one_indices(ps_y_mask)
            
            problemsetlevel_data.append([p2,px2,py2,r2,rx2,ry2,mask2,x_mask2,y_mask2,ps_mask,ps_x_mask,ps_y_mask,first,last,x_first,x_last,y_first,y_last,re_p2,re_r2])
        
        return problemsetlevel_data

    def mypreprocess(self,data):
        problemset_data=self.problemsetlevel_process_aug(data)

        return problemset_data

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
    
        pbatch = list(zip(*batch))
        pack1=(torch.LongTensor(pbatch[0]), torch.LongTensor(pbatch[1]), torch.LongTensor(pbatch[2]), torch.LongTensor(pbatch[3]),torch.LongTensor(pbatch[4]), torch.LongTensor(pbatch[5]), torch.LongTensor(pbatch[6]), torch.LongTensor(pbatch[7]),torch.LongTensor(pbatch[8]),torch.LongTensor(pbatch[9]),torch.LongTensor(pbatch[10]),torch.LongTensor(pbatch[11]),torch.LongTensor(pbatch[12]),torch.LongTensor(pbatch[13]),torch.LongTensor(pbatch[14]),torch.LongTensor(pbatch[15]),torch.LongTensor(pbatch[16]),torch.LongTensor(pbatch[17]),torch.LongTensor(pbatch[18]),torch.LongTensor(pbatch[19]))
        return pack1

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


