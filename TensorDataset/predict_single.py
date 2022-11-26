import pandas as pd
from tqdm import tqdm
from Preprocess.attentionCal import aggregate_attention
from Preprocess.spanMatcher import returnMask
from Preprocess.dataCollect import set_name
from TensorDataset.datsetSplitter import encodeData
from TensorDataset.dataLoader import combine_features
from Models.bertModels import SC_weighted_BERT
from transformers import BertTokenizer
import transformers
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from Models.utils import load_model

def annotated_data(tokenized_string):
    tokenized_string=[x.lower() for x in tokenized_string]
    dict_data=[]
    length=len(tokenized_string)
    l=[0]*length
    l[0]=1
    l2=[l,l,l]
    temp={}
    temp['post_id']="24198580_gab"
    temp['text']=tokenized_string
    final_label=[]
    for i in range(1,4):
        temp['annotatorid'+str(i)]=i+2
        temp['target'+str(i)]=["African"]
        temp['label'+str(i)]="offensive"
        final_label.append(temp['label'+str(i)])

    final_label_id=max(final_label,key=final_label.count)
    temp['rationales']=l2
    temp['final_label']=final_label_id    
    dict_data.append(temp)    
    temp_read = pd.DataFrame(dict_data)  
    return temp_read

def get_data(data,params,tokenizer):
    '''input: data is a dataframe text ids attentions labels column only'''
    '''output: training data in the columns post_id,text, attention and labels '''

    majority=params['majority']
    post_ids_list=[]
    text_list=[]
    attention_list=[]
    label_list=[]
    count=0
    count_confused=0
    for index,row in tqdm(data.iterrows(),total=len(data)):
        #print(params)
        text=row['text']
        post_id=row['post_id']

        annotation_list=[row['label1'],row['label2'],row['label3']] 
        annotation=row['final_label']
        
        if(annotation != 'undecided'):
            tokens_all,attention_masks=returnMask(row,params,tokenizer)
            attention_vector= aggregate_attention(attention_masks,row, params)     
            attention_list.append(attention_vector)
            text_list.append(tokens_all)
            label_list.append(annotation)
            post_ids_list.append(post_id)
        else:
            count_confused+=1
            
    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    training_data = pd.DataFrame(list(zip(post_ids_list,text_list,attention_list,label_list)), 
                   columns =['Post_id','Text', 'Attention' , 'Label']) 
    
    
    filename=set_name(params)
    training_data.to_pickle(filename)
    return training_data

def collec_data(params,tokenized_string):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    data_all_labelled=annotated_data(tokenized_string)
    train_data=get_data(data_all_labelled,params,tokenizer)
    return train_data


def createDataset(params):
    filename=set_name(params)
    dataset=collec_data(params)    
    dataset= pd.read_pickle(filename) 
    X_pred=dataset
    if(params['bert_tokens']):
        vocab_own=None    
        vocab_size =0
        padding_idx =0

    X_pred=encodeData(X_pred,vocab_own,params)
        
    print("total prediction size:", len(X_pred))

        
    os.mkdir(filename[:-7])
    with open(filename[:-7]+'/pred_data.pickle', 'wb') as f:
        pickle.dump(X_pred, f)
    
    return X_pred

def pred_model(params,device):
    
    pred=createDataset(params)

    pred_dataloader =combine_features(pred,params,is_train=False)   
   
    model = SC_weighted_BERT.from_pretrained(
            params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = params['num_classes'], # The number of output labels
            output_attentions = True, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert'],
            params=params
            )
    model=load_model(model,params)
    if(params["device"]=='cuda'):
        model.cuda()

    bert_model = params['path_files']
    name_one=bert_model
    model.eval()
    for step, batch in tqdm(enumerate(pred_dataloader)):
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels 
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        
        outputs = model(b_input_ids, 
            attention_vals=b_att_val,
            attention_mask=b_input_mask, 
            labels=b_labels,
            device=device)

        return outputs