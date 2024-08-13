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
            train_data.append(line['records'])
        
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
        theta1=self.opt['theta1']
        theta2=self.opt['theta2'] 
        rank_dir="dataset/"+self.opt['dataset']+"/problem_diff_rank.jsonl"
        rank_dict=json.load(open(rank_dir))
        rank_array_str=list(rank_dict.keys())
        rank_array = [int(x) for x in rank_array_str]
        for excise in problemset_data:
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
            first,last=self.find_first_last_one_indices(mask)
            x_first,x_last=self.find_first_last_one_indices(x_mask)
            y_first,y_last=self.find_first_last_one_indices(y_mask)
            if first==None or last==None or x_first==None or x_last==None or y_first==None or y_last==None: 
               continue
            
            problemsetlevel_data.append([p,px,py,r,rx,ry,mask,x_mask,y_mask,re_p,re_r,first,last,x_first,x_last,y_first,y_last])
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
        pack1=(torch.LongTensor(pbatch[0]), torch.LongTensor(pbatch[1]), torch.LongTensor(pbatch[2]), torch.LongTensor(pbatch[3]),torch.LongTensor(pbatch[4]), torch.LongTensor(pbatch[5]), torch.LongTensor(pbatch[6]), torch.LongTensor(pbatch[7]),torch.LongTensor(pbatch[8]),torch.LongTensor(pbatch[9]),torch.LongTensor(pbatch[10]),torch.LongTensor(pbatch[11]),torch.LongTensor(pbatch[12]),torch.LongTensor(pbatch[13]),torch.LongTensor(pbatch[14]),torch.LongTensor(pbatch[15]),torch.LongTensor(pbatch[16]))
        return pack1

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


