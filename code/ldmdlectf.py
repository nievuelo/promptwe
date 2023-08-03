import pandas as pd
import os
from tensorboardX import SummaryWriter
from openprompt.data_utils import InputExample
import numpy as np
import torch
from openprompt.prompts import ManualVerbalizer
import torch
import joblib
data_dir = ".\dataset\\"
label_dict = {'fake': int(0), 'real': int(1)}
raw_dataset = {}
# prompt_model3 = joblib.load("D:\okclassification-pandemic-prompt\loadmodel\prmpttmpl3.pkl")
# prompt_model = joblib.load("D:\okclassification-pandemic-prompt\loadmodel\prmpttmpl4.pkl")
# prompt_model = joblib.load("D:\okclassification-pandemic-prompt\loadmodel\prmpttmpl5.pkl")
# prompt_model = joblib.load("D:\okclassification-pandemic-prompt\loadmodel\prmpttmpl8.pkl")
# prompt_model = joblib.load("D:\okclassification-pandemic-prompt\loadmodel\prmpttmpl9.pkl")
prompt_model = joblib.load("loadmodel/prmpttmpl2ectf.pkl")
# prompt_model5ctf = joblib.load("D:\okclassification-pandemic-prompt\loadmodel\prmtpttmpl5dsetctf.pkl")
# prompt_model8ctf = joblib.load("D:\okclassification-pandemic-prompt\loadmodel\prmpttmpl8ctf.pkl")
# load pandemic dataset

def load_data(path,label_dic):
    raw_dataset = pd.read_csv(path, header= None, sep=',',names = ["text", "label"])

    # print(raw_dataset.shape)
    texts = raw_dataset.text.to_list()                 
    labels = raw_dataset.label.map(label_dic).to_list()   
    raw_ds = []
    raw_ds.append(texts)
    raw_ds.append(labels)
    return raw_ds


for name in ["trainctf.csv","valctf.csv"]:
    path = name
    one_raw_dataset = load_data(path,label_dict)
    raw_dataset[name.replace('ctf.csv','')]=one_raw_dataset       
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "real",
    "fake"
]

dataset = dict([(k, []) for k in  ["train","val"]])

for name in ["train","val"]:

    lng = len(raw_dataset[name][1])
    for i in range(lng):
        dataset[name].append(
            InputExample(
                guid = i,
                text_a = raw_dataset[name][0][i],
                # text_b = "Is it true?",
                label = raw_dataset[name][1][i],
            )
        )


from openprompt.plms import load_plm
plm,tokenizer,model_config,WrapperClass = load_plm("bert","bert-base-uncased")




from openprompt.prompts import ManualTemplate
myTemplate = ManualTemplate(
    text ='{"placeholder":"text_a"} all in all it is {"mask"}',
    tokenizer = tokenizer,
)
# from openprompt.prompts import MixedTemplate
# myTemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
#                            text='{"placeholder":"text_a"} {"soft":"It"} {"soft": "was"} {"mask"}')
#

from openprompt import PromptDataLoader

validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=myTemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=128,
                                         decoder_max_length=3,
                                         batch_size=4, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="head")




# for example the verbalizer contains multiple label words in each class
# myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,classes=classes,
#                         label_words={
#                             "fake":["fake","false","exaggerated","underestimated","ridiculous","forged","phoney","bogus",],
#                             "real":["real","true","fact","scientific","authentic","genuine", "actual", "factual","authentic"]
#                         })

myverbalizer =  ManualVerbalizer(tokenizer, num_classes=2,
                        label_words=[["real"], ["fake"]])


use_cuda = True
alllabels=[]
allpreds = []
allpredsprob=[]
j=0
import torch.nn.functional as F
for step, inputs in enumerate(validation_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    for q in range(4):
        allpredsprob.append(F.softmax(logits[q]).tolist())
temp=[]
allpredsp=[]
for i in allpredsprob:
    temp.append(i[1])
    if i[1] >=0.5:
        allpredsp.append(1)
    else:
        allpredsp.append(0)

tp = sum([int(i == j and i == 1) for i, j in zip(allpredsp, alllabels)])
tn = sum([int(i == j and i == 0) for i, j in zip(allpredsp, alllabels)])
fp = sum([int(i != j and i == 1) for i, j in zip(allpredsp, alllabels)])
fn = sum([int(i != j and i == 0) for i, j in zip(allpredsp, alllabels)])
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print("pecision:{}".format(precision))
print("f1:{}".format(f1))
print("recall:{}".format(recall))
print("accuracy:{}".format(accuracy))

np.savetxt('prmpttplt2ectf', temp)
a=np.loadtxt('prmpttplt2ectf')
print(a)


