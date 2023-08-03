import joblib
import pandas as pd
import os
from tensorboardX import SummaryWriter
from openprompt.data_utils import InputExample
import numpy as np
import torch
from bert import evaluation
dirs = '/home/workstation-14/zxr/classification-pandemic-prompt -1/mdlld'

# 读取模型
prmpt_model = joblib.load(dirs + '/prmptmnl.pkl')
prmptmixed_model = joblib.load(dirs+'/prmptmixedmnl.pkl')
allpreds = []
alllabels = []
prmptlbl =[]
logitstemp = []
# print(type(prmpt_model))
# print(type(prmptmixed_model))
# load pandemic dataset
coun1t=0
def load_data(path,label_dic):
    raw_dataset = pd.read_csv(path, header= None, sep=',',names = ["text", "label"])
    # csv 文件间隔符号一般为tab或者”，“
    # print(raw_dataset.shape)
    texts = raw_dataset.text.to_list()                  #将字符串转化为列表
    labels = raw_dataset.label.map(label_dic).to_list()       #在数据集合中的标签映射为可以九三的数
    raw_ds = []
    raw_ds.append(texts)
    raw_ds.append(labels)
    return raw_ds

data_dir = "dataset/"

from openprompt.plms import load_plm
plm,tokenizer,model_config,WrapperClass = load_plm("bert","bert-base-uncased")

label_dict = {'fake': int(0), 'real': int(1)}
raw_dataset = {}

# print(path)
for name in ["val6.csv"]:
    path = os.path.join(data_dir, name)
    one_raw_dataset = load_data(path,label_dict)
    raw_dataset[name.replace('6.csv','')]=one_raw_dataset        #raw_dataset 是字典，one_raw_dataset是对应的一个训练集的列表

# print(raw_dataset["test"][0][3])

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "real",
    "fake"
]

from openprompt.prompts import ManualTemplate
myTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It is {"mask"}',
    tokenizer = tokenizer,
)
from openprompt.prompts import MixedTemplate


mymixedtemplate = MixedTemplate( model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"soft"} {"mask"}.')

dataset = dict([(k, []) for k in  ["val"]])

for name in ["val"]:

    lng = len(raw_dataset[name][1])
    for i in range(lng):
        dataset[name].append(
            InputExample(
                guid = i,
                text_a = raw_dataset[name][0][i],
                label = raw_dataset[name][1][i]
            )
        )

from openprompt import PromptDataLoader

validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=myTemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                         decoder_max_length=3,
                                         batch_size=4, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="head")
validationmixed_dataloader = PromptDataLoader(dataset=dataset["val"], template=mymixedtemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                         decoder_max_length=3,
                                         batch_size=4, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="head")

for step, inputs in enumerate(validation_dataloader):
    inputs = inputs.cuda()
    logits = prmpt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    for i in range(4):
        # print(logits[i][0].item())
        prmptlbl.append(prmpt_model.verbalizer.normalize(logits)[i][0].item())
    # print(logits[-1].item()) [0][0]
    # print(torch.argmax(logits, dim=-1))
# print(allpreds)
# prmptlbl = allpreds

prmptmixedlbl =[]
for step, inputs in enumerate(validationmixed_dataloader):
    inputs = inputs.cuda()
    logits = prmptmixed_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    for i in range(4):
        # print(logits[i][0].item())
        prmptmixedlbl.append(prmptmixed_model.verbalizer.normalize(logits)[i][0].item())


#
# import train
# brtlbl = train.find()



# print(brtlbl)
import math
brttrlbl=[]
prmptmixedtrlbl=[]
prmpttrlbl=[]
fnllbl=[]
vtlbl =[]
for i in zip(prmptmixedlbl, prmptlbl):
    # s=math.sqrt((i[0]**2+i[1]**2))
    # s=math.sqrt(i[0]*i[1])
    # print(i[0])
    # print(i[1])
    # print(i)
    if i[0] >= 0.5:
        prmptmixedtrlbl.append(0)
    else:
        prmptmixedtrlbl.append(1)
    if i[1] >= 0.5:
        prmpttrlbl.append(0)
    else:
        prmpttrlbl.append(1)
# # print(len(brttrlbl))
notsame=[]
for i in range(len(prmptmixedtrlbl)):
    if prmptmixedtrlbl[i] != prmpttrlbl[i]:
        coun1t+=1
        notsame.append((i,prmptmixedtrlbl[i],prmpttrlbl[i],alllabels[i]))
    elif prmptmixedtrlbl[i]==prmpttrlbl[i] and prmpttrlbl[i]!=alllabels[i]:
        coun1t+=1
        notsame.append((i, prmptmixedtrlbl[i], prmpttrlbl[i], alllabels[i]))
coun2t=0
coun3t=0
coun4t=0
for i in range(len(notsame)):
    if notsame[i][1]!=notsame[i][3]:
        coun2t+=1
    if notsame[i][2]!=notsame[i][3]:
        coun3t+=1
    if notsame[i][2]==notsame[i][1] and notsame[i][2]!=notsame[i][3]:
        coun4t+=1
print(notsame)
print(coun1t)
print("this is the false brt {}".format(coun2t))
print("this is the false prmpt {}".format(coun3t))
print("this is the all false prediction {}".format(coun4t))


for j in range(1,10):
    # print("this turn the weight of brt is {}".format(j))
    fnllbl = []
    vtlbl = []
    for i in zip(prmptmixedlbl,prmptlbl):
        # s=math.sqrt((i[0]**2+i[1]**2))
        # s=math.sqrt(i[0]*i[1])
        # print(i[0])
        # print(i[1])
        # print(i)
        y=j/10
        s=(round(i[0],4)*y+round(i[1],4)*(10-y))/10


      # print(s)


        vtlbl.append(s)
        if s >= 0.5:
            fnllbl.append(0)
        else:
            fnllbl.append(1)

    # print(fnllbl)
    # print(brttrlbl)

    tp = sum([int(i == j and i == 0) for i, j in zip(prmptmixedtrlbl, alllabels)])
    tn = sum([int(i == j and i == 1) for i, j in zip(prmptmixedtrlbl, alllabels)])
    fp = sum([int(i != j and i == 0) for i, j in zip(prmptmixedtrlbl, alllabels)])
    fn = sum([int(i != j and i == 1) for i, j in zip(prmptmixedtrlbl, alllabels)])
    if (tp + tn + fp + fn) != 0 and (tp + fp) != 0 and (tp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("pecision：{}".format(precision))
        # print("f1：{}".format(f1))
        # print("recall：{}".format(recall))
        print("this is mixed prompt")
        print("accuracy：{}".format(accuracy))
    tp = sum([int(i == j and i == 0) for i, j in zip(prmpttrlbl, alllabels)])
    tn = sum([int(i == j and i == 1) for i, j in zip(prmpttrlbl, alllabels)])
    fp = sum([int(i != j and i == 0) for i, j in zip(prmpttrlbl, alllabels)])
    fn = sum([int(i != j and i == 1) for i, j in zip(prmpttrlbl, alllabels)])
    if (tp + tn + fp + fn) != 0 and (tp + fp) != 0 and (tp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("This is validation dataset.")
        # print("pecision：{}".format(precision))
        # print("f1：{}".format(f1))
        # print("recall：{}".format(recall))
        print("this is manual prompt")
        print("accuracy：{}".format(accuracy))
    tp = sum([int(i == j and i == 0) for i, j in zip(fnllbl, alllabels)])
    tn = sum([int(i == j and i == 1) for i, j in zip(fnllbl, alllabels)])
    fp = sum([int(i != j and i == 0) for i, j in zip(fnllbl, alllabels)])
    fn = sum([int(i != j and i == 1) for i, j in zip(fnllbl, alllabels)])
    if (tp + tn + fp + fn) != 0 and (tp + fp) != 0 and (tp + fn) != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        # print("This is validation dataset.")
        # print("pecision：{}".format(precision))
        # print("f1：{}".format(f1))
        # print("recall：{}".format(recall))
        print("this is total vote status.")
        print("accuracy：{}".format(accuracy))



