import re 
from Preprocess.preProcess import *
from Preprocess.preProcess import preProcessing

from transformers import BertTokenizer
import string 
#input:Given a string in the form of "(start1-end1)span_text1||(start2-end2)......"  or "{}" if no span selected 
#output:Gives back a list of tuples of the form (string,start,end) or [] if no span selected


def giveSpanList(row,tokens,string1,data_type):
    if(data_type=='old'):
        if string1 in ["{}","{","}"]  :
            return []
        list1=string1.split("||")
        string_all=[]
        for l in list1:
            # collect the string 
            # colect the string postion (start--end) in the original text
            string_mask=re.findall('\((.*?)\)',l)[0]
            string=l[len(string_mask)+2:]
            [start,end]=string_mask.split("--")
            string_all.append((string,start,end))
    else:
        string_all=[]
        flag=0
        if(row['post_id'] in ['10510109_gab','1081573659137830912_twitter','1119979940080783360_twitter']):
              flag=1
        for exp in string1:
            start,end= int(exp.split('-')[1]),int(exp.split('-')[2])
            if(flag==1):
                print(exp)

            start_pos=0
            end_pos=0
            pos=0
            count=0
            for tok in tokens:
                if(flag==1):
                    print(count)
                    print(pos)
                if(count==start):
                    start_pos=pos
                pos+=len(tok)+1
                if((count+1)==end):
                    end_pos=pos
                    break
                
                count+=1
                
            string_all.append((exp.split('-')[0],start_pos,end_pos)) 
        
    return string_all



def returnMask(row,params,tokenizer):
    
    text_tokens=row['text']
 
    
    ##### a very rare corner case
    if(len(text_tokens)==0):
        text_tokens=['dummy']
        print("length of text ==0")
    #####
    
    
    mask_all= row['rationales']
    mask_all_temp=mask_all
    count_temp=0
    while(len(mask_all_temp)!=3):
        mask_all_temp.append([0]*len(text_tokens))
    
    word_mask_all=[]
    word_tokens_all=[]
    
    for mask in mask_all_temp:
        if(mask[0]==-1):
            mask=[0]*len(mask)
        
        
        list_pos=[]
        mask_pos=[]
        
        flag=0
        for i in range(0,len(mask)):
            if(i==0 and mask[i]==0):
                list_pos.append(0)
                mask_pos.append(0)
            
            
            
            
            if(flag==0 and mask[i]==1):
                mask_pos.append(1)
                list_pos.append(i)
                flag=1
                
            elif(flag==1 and mask[i]==0):
                flag=0
                mask_pos.append(0)
                list_pos.append(i)
        if(list_pos[-1]!=len(mask)):
            list_pos.append(len(mask))
            mask_pos.append(0)
        string_parts=[]
        for i in range(len(list_pos)-1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i+1]])
        
        
        
        
        if(params['bert_tokens']):
            word_tokens=[101]
            word_mask=[0]
        else:
            word_tokens=[]
            word_mask=[]

        
        for i in range(0,len(string_parts)):
            tokens=ek_extra_preprocess(" ".join(string_parts[i]),params,tokenizer)
            masks=[mask_pos[i]]*len(tokens)
            word_tokens+=tokens
            word_mask+=masks


        if(params['bert_tokens']):
            ### always post truncation
            word_tokens=word_tokens[0:(int(params['max_length'])-2)]
            word_mask=word_mask[0:(int(params['max_length'])-2)]
            word_tokens.append(102)
            word_mask.append(0)

        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
        
#     for k in range(0,len(mask_all)):
#          if(mask_all[k][0]==-1):
#             word_mask_all[k] = [-1]*len(word_mask_all[k])
    if(len(mask_all)==0):
        word_mask_all=[]
    else:    
        word_mask_all=word_mask_all[0:len(mask_all)]
    return word_tokens_all[0],word_mask_all    
        
   





