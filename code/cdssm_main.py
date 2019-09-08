# -*- coding: utf-8 -*-
import input_data
import torch
from model_cdssm import CDSSM 
from random import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import time
import pickle
import torch.optim as optim

a = Random()
choosedataset = 'SNIP'
#choose_model=['CDSSM', 'ZERODNN']
# ================================== data setting =============================
dataSetting={}
dataSetting['test_mode']=0
######
dataSetting['training_prob']=0.7
dataSetting['test_intrain_prob']=0.3

if choosedataset == 'SNIP':
    dataSetting['data_prefix']='../data/nlu_data/'
    dataSetting['dataset_name']='dataSNIP.txt'
    dataSetting['wordvec_name']='wiki.en.vec'
    dataSetting['dataset'] = 'SNIP'

#=====================================load w2v ================================

# only seen in training process

data = input_data.read_datasets_gen(dataSetting)
x_tr = torch.from_numpy(data['x_tr'])
x_te = torch.from_numpy(data['x_te'])
y_tr = torch.from_numpy(data['y_tr'])
y_tr_id = torch.from_numpy(data['y_tr'])
y_te_id = torch.from_numpy(data['y_te'])
y_ind = torch.from_numpy(data['s_label'])
s_len = torch.from_numpy(data['s_len'])# number of training examples 
embedding = torch.from_numpy(data['embedding'])

#x_table = torch.from_numpy()
u_len = torch.from_numpy(data['u_len'])# number of testing examples 
s_cnum = np.unique(data['y_tr']).shape[0]
u_cnum = np.unique(data['y_te']).shape[0]
y_emb_tr = data['y_emb_tr']
y_emb_te = data['y_emb_te']
vocab_size, word_emb_size = data['embedding'].shape

#============================ cut train data in batch =========================

config = {}
sample_num, max_time = data['x_tr'].shape
test_sample_num, sen = data['x_te'].shape
config['sample_num'] = sample_num #sample number of training data
config['test_sample_num'] = test_sample_num 
 # vocab size of word vectors
config['seen_class']=data['seen_class']
config['unseen_class']=data['unseen_class']
config['emb_len'] = word_emb_size
config['s_cnum'] = s_cnum # seen class num
config['u_cnum'] = u_cnum #unseen class num
config['st_len'] = max_time
config['num_epochs'] = 15 # number of epochs
config['model_name'] = 'CDSSM'
config['dataset'] = choosedataset
config['report']=True
config['dropout']=0.5
config['ckpt_dir'] = './saved_models/' 
config['test_mode'] = dataSetting['test_mode']
config['experiment_time']= time.strftime('%y%m%d%I%M%S')
config['batch_size'] = 32
config['learning_rate'] = 0.01
batch_num = int(config['sample_num'] / config['batch_size']+1)

#config['test_num'] = test_num #number of test data

y_emb_te = torch.from_numpy(np.tile(y_emb_te,(config['test_sample_num'],1)))
y_emb_te = torch.tensor(y_emb_te,dtype=torch.long)

#epochpath = config['ckpt_dir']+'testmode: '+str(config['test_mode'])+'_'+config['model_name']+'_'+config['dataset']+str(config['batch_size'])+"_"+str(config['learning_rate'])+'.txt'
#fo = open(epochpath, "a")

def generate_batch(n, batch_size):
    batch_index = a.sample(range(n), batch_size)
    return batch_index

def evaluate_test(data, config, seen_n, unseen_n):
    
    #test_sample_num = config['test_sample_num']
    test_batch_num = int(config['test_sample_num'] / config['test_sample_num'])
    total_unseen_pred = np.array([], dtype=np.int64)
    total_y_test = np.array([], dtype=np.int64)
    
    #x_te = torch.from_numpy(data['x_te'])
    #y_te_id = torch.from_numpy(data['y_te'])
  
    with torch.no_grad():
        
        for batch in range(test_batch_num):
            #batch_index = generate_batch(config['test_sample_num'], config['test_sample_num'])
            batch_index = generate_batch(config['test_sample_num'], config['test_sample_num'])
            batch_x = x_te[batch_index]
            batch_y_id = y_te_id[batch_index]
            y_pred = cdssm(batch_x, y_emb_te, embedding)
            
            y_pred_id = torch.argmax(y_pred, dim=1)
            total_unseen_pred = np.concatenate((total_unseen_pred,  y_pred_id))
            total_y_test = np.concatenate((total_y_test, batch_y_id))
           
            #print('test batch ',batch)
            #acc = accuracy_score(total_y_test, total_unseen_pred)
            #print (classification_report(total_y_test, total_unseen_pred, digits=4))
            #print (precision_recall_fscore_support(total_y_test, total_unseen_pred))
    
        print('        '+config['model_name']+" "+ config['dataset']+" ZStest Perfomance        ")
        acc = accuracy_score(total_y_test, total_unseen_pred)
        print (classification_report(total_y_test, total_unseen_pred, digits=4))
        logclasses=precision_recall_fscore_support(total_y_test, total_unseen_pred)
        
        '''
        seen_ind = [(np.where(total_y_test==i)[0]).tolist() for i in range(seen_n)]
        unseen_ind = [(np.where(total_y_test==i)[0]).tolist() for i in range(seen_n, seen_n+unseen_n)]

        seen_ind = (np.hstack(seen_ind)).astype(int)
        unseen_ind = (np.hstack(unseen_ind)).astype(int)
        
        s_acc = accuracy_score(total_y_test[seen_ind], total_unseen_pred[seen_ind])
        u_acc = accuracy_score(total_y_test[unseen_ind], total_unseen_pred[unseen_ind])
        print ("------------seen-------------------")
        #print (classification_report(total_y_test[seen_ind], total_unseen_pred[seen_ind], digits=4))
        print ('seen testacc: ', s_acc)
        print ("------------unseen-------------------")
        #print (classification_report(total_y_test[unseen_ind], total_unseen_pred[unseen_ind], digits=4))
        print ('unseen testacc: ', u_acc)
       
        
        fo.write(classification_report(total_y_test, total_unseen_pred, digits=4))
        fo.write("\n")
        fo.write("test acc:"+str(acc)+"\n")
        #classreport.append(classification_report(total_y_test, total_unseen_pred, digits=4))
        #fo.write("seen acc:"+str(s_acc)+"\n")
        #fo.write("unseen acc:"+str(u_acc)+"\n\n\n")
        '''
    return acc, logclasses


#classreport = []

print('y_emb', y_emb_tr)
print('y_emb_te', y_emb_te)
y_emb = torch.from_numpy(np.tile(y_emb_tr,(config['batch_size'],1)))
#test_y_emb = torch.from_numpy(np.tile(y_emb_tr,(test_sample_num,1)))
y_emb = torch.tensor(y_emb,dtype=torch.long)
y_tr_id = torch.tensor(y_tr_id,dtype=torch.long)

cdssm = CDSSM(config)
optimizer = torch.optim.SGD(cdssm.parameters(), lr=config['learning_rate'], momentum=0.9)

#===================================train=====================================

cdssm.train()
i = 0
avg_acc = 0.0
test_avg_acc = 0.0
log=[]
logForClasses=[]
best_acc = 0
tr_best_acc = 0
tr_min_loss = np.inf
curr_step = 0
overall_train_time = 0.0
overall_test_time = 0.0
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
filename=config['ckpt_dir']+'mode'+str(config['test_mode'])+'_'+\
        config['dataset']+'_'+config['model_name']+'_'+config['experiment_time']+'.pkl'
  
for epoch in range(config['num_epochs']):
    
    avg_acc = 0.0
    scheduler.step()
    epoch_time = time.time()      
    print("==================epoch ", epoch)
    for batch in range(batch_num):
    #for i in range(5):
        
            batch_index = generate_batch(config['sample_num'], config['batch_size'])
            batch_x = x_tr[batch_index]
            batch_y_id = y_tr_id[batch_index]
            batch_len = s_len[batch_index]
            batch_ind = y_ind[batch_index]
            #print ("======batch x",batch_x.shape)
            #print ("=========batch x sum", sum(batch_x))
            #print ("=========batch y sum", sum(batch_y_id))
            y_pred = cdssm.forward(batch_x, y_emb, embedding)
            loss = cdssm.loss(y_pred, batch_y_id)
            y_pred_id = torch.argmax(y_pred, dim=1)
            acc = accuracy_score(y_pred_id, batch_y_id)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(lstm.parameters(), 5)
            optimizer.step()
            avg_acc += acc 
            #print(acc)
            
    print(round((avg_acc/batch_num),4))
    train_time = time.time() - epoch_time
    overall_train_time += train_time
    #============================  test =======================================
    print("===========================test====================================")
    
    
    if choosedataset == 'SMP':
        seen_n = 25
        unseen_n = 6
    elif choosedataset == 'SNIP':
        seen_n = 5
        unseen_n = 2
    
    #fo.write("-----------------epoch "+str(epoch)+"----------------\n")
    cur_acc, logC = evaluate_test(data, config, seen_n, unseen_n)
    #test_avg_acc = 0.0
    logForClasses.append(logC)

    #fo.write("acc:"+str(round((avg_acc / batch_num), 4))+"\n")
    #print(round((test_avg_acc/test_batch_num),4))
    
    print("-----epoch : ", epoch, "/", config['num_epochs'], " Loss: ", loss.item(), \
                  " Acc:", round((avg_acc / batch_num), 4), "TestACC:", round(cur_acc,6), \
                  "Train_time: ", round(train_time, 4), "overall train time: ", \
                  round(overall_train_time,4),'-----')
    log.append(dict(epoch=epoch, loss=loss.item(),acc_tr=round((avg_acc / batch_num), 8), acc_te=round(cur_acc,8)))
    
   
    # early stop
    if (avg_acc / batch_num >= tr_best_acc and loss.item() <= tr_min_loss) or \
    (abs(avg_acc / batch_num - tr_best_acc)<0.005 and loss.item() <= tr_min_loss):
        tr_best_acc = avg_acc / batch_num
        tr_min_loss = loss.item()
        if config['report']:
#                torch.save(lstm.state_dict(), config['ckpt_dir'] + 'best_model'+config['experiment_time']+'.pth')
            if cur_acc > best_acc:
                # save model
                best_acc = cur_acc
                config['best_epoch']=epoch
                config['best_acc']=best_acc
            print("cur_acc", cur_acc)
            print("best_acc", best_acc)
            curr_step = 0
    else:
        curr_step += 1
        if curr_step>5:
            print('curr_step: ', curr_step)
            if curr_step == 15:
                print('Early stop!')                    
                print("Overall training time", overall_train_time)
                print("Overall testing time", overall_test_time)
                # output log
                if config['report']:

                    config['overall_train_time']=overall_train_time
                    config['overall_test_time']=overall_test_time
                    pickle.dump([config,data['sc_dict'],data['uc_dict'],\
                                 log,logForClasses],open(filename, 'wb'))   
                                      
                break

print ("done")
