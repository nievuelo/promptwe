import pandas as pd
import os
from tensorboardX import SummaryWriter
from openprompt.data_utils import InputExample
from sklearn import metrics
import numpy as np
import torch
import json

# def set_seed(seed):
#     torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
#     torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
#     torch.backends.cudnn.deterministic = True  # cudnn
#     np.random.seed(seed)  # numpy



# load pandemic dataset

def load_data(path,label_dic):
    # with open(path, 'r', encoding='utf-8') as f:
    #     raw_dataset = json.load(f)
    # raw_dataset=pd.DataFrame(raw_dataset)

    # csv 文件间隔符号一般为tab或者”，“
    # print(raw_dataset.shape)
    # texts = raw_dataset.claim.to_list()                  #将字符串转化为列表
    # labels = raw_dataset.label.map(label_dic).to_list()       #在数据集合中的标签映射为可以九三的数
    # raw_ds = []
    # raw_ds.append(texts)
    # raw_ds.append(labels)
    # return raw_ds

    train = pd.read_csv(path,  sep=',', names=["explain","claim","label"])
    print(train.shape)
    # valid = pd.read_csv(os.path.join(path, "cnews.val.txt"), header=None, sep='\t', names=["label", "text"])
    # test = pd.read_csv(os.path.join(path, "cnews.test.txt"), header=None, sep='\t', names=["label", "text"])
    claim = train.claim.to_list()
    del claim[0]
    explain = train.explain.to_list()
    del explain[0]
    labels = train.label.map(label_dic).to_list()
    del labels[0]
    raw_ds = []
    raw_ds.append(claim)
    raw_ds.append(explain)
    raw_ds.append(labels)

    return raw_ds
    # label_dic = dict(zip(train.label.unique(), range(len(train.label.unique()))))



data_dir = "newdataset/LIAR-RAW/"

label_dict = {'true': int(0), 'mostly-true': int(1),'half-true': int(2), 'barely-true': int(3),'false': int(4), 'pants-fire': int(5)}

raw_dataset = {}
# set_seed(1)
# print(path)
for name in ["prmpttrain.csv","prmptval.csv","prmpttest.csv"]:
# for name in ["val.json"]:
    path=os.path.join(data_dir,name)
    one_raw_dataset = load_data(path,label_dict)
    name=name.replace('.csv','')
    name=name.replace('prmpt','')

    raw_dataset[name]=one_raw_dataset        #raw_dataset 是字典，one_raw_dataset是对应的一个训练集的列表

# print(raw_dataset["train"][0][0])
# print(raw_dataset["train"][1][0])
# print(raw_dataset["train"][2][0])
# print(raw_dataset["val"][0][0])
# print(raw_dataset["test"][0][0])

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    'true',
    'mostly-true',
    'half-true',
    'barely-true',
    'false',
    'pants-fire'
]


dataset = dict([(k, []) for k in  ["train","val","test"]])


# for name in ["train","val","test"]:
#
#     lng = len(raw_dataset[name][1])
#     for i in range(lng):
#         if(len(raw_dataset[name][0][i])>400):
#             dataset[name].append(
#                 InputExample(
#                     guid = i,
#                     text_a = raw_dataset[name][0][i][:400],
#                     text_b = raw_dataset[name][1][i],
#                     label = raw_dataset[name][2][i]
#                 )
#             )
#         else:
#             dataset[name].append(
#                 InputExample(
#                     guid = i,
#                     text_a = raw_dataset[name][0][i],
#                     text_b = raw_dataset[name][1][i],
#                     label = int(raw_dataset[name][2][i])
#                 )
#             )

for name in ["train","val","test"]:

    lng = len(raw_dataset[name][1])
    for i in range(lng):
        if(len(raw_dataset[name][0][i])>400):
            dataset[name].append(
                InputExample(
                    guid = i,
                    text_a = raw_dataset[name][0][i][:400],
                    label = raw_dataset[name][2][i],

                )
            )
        else:
            dataset[name].append(
                InputExample(
                    guid = i,
                    text_a = raw_dataset[name][0][i],
                    label = int(raw_dataset[name][2][i])
                )
            )

# print(np.shape(raw_dataset["train"]))
print(dataset['train'][0])



from openprompt.plms import load_plm
plm,tokenizer,model_config,WrapperClass = load_plm("bert","bert-base-uncased")


# 构造模板

from openprompt.prompts import ManualTemplate
from openprompt.prompts import MixedTemplate
# myTemplate = ManualTemplate(
#     text = '{"placeholder":"text_a"} It is {"mask"}. ',
#     tokenizer = tokenizer,
# )
myTemplate = MixedTemplate(
    model=plm,
    text = '{"placeholder":"text_a"} {"soft":"It"} {"soft":"is"}{"mask"}. {"soft":"Because"} {"placeholder":"text_b"}',
    tokenizer = tokenizer,
)


from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=myTemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=384, decoder_max_length=3,
    batch_size=32,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import AutomaticVerbalizer
import torch

myverbalizer = ManualVerbalizer(tokenizer, num_classes=6,
                        label_words=[ ["true"],["mostly-true"],["half-true"],["barely-true"],["false"],["pants_fire"]])


# print(myverbalizer.label_words_ids)
# logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and
# print(myverbalizer.process_logits(logits)) # see what the verbalizer do


from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=myTemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import  AdamW,get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=myTemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                         decoder_max_length=3,
                                         batch_size=16, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="head")
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=myTemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                         decoder_max_length=3,
                                         batch_size=16, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="head")
import os
import joblib
# 创建文件目录
dirs = 'testModel'

for epoch in range(100):
    name = 'table' + str(epoch)
    print(name)
    tot_loss = 0
    writer = SummaryWriter(comment=name)
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        loss = loss_func(logits, labels.long())
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar('Train', loss,step)
        if step % 100 == 1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
    writer.close()

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    label2ind_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    report = metrics.classification_report(alllabels, allpreds, target_names=label2ind_dict.keys(), digits=4)
    confusion = metrics.confusion_matrix(alllabels, allpreds)
    print(report, confusion)
    report = metrics.classification_report(alllabels, allpreds, target_names=label2ind_dict.keys(), digits=4)
    confusion = metrics.confusion_matrix(alllabels, allpreds)
    tp = sum([int(i == j and i == 1) for i, j in zip(allpreds, alllabels)])
    tn = sum([int(i == j and i == 0) for i, j in zip(allpreds, alllabels)])
    fp = sum([int(i != j and i == 1) for i, j in zip(allpreds, alllabels)])
    fn = sum([int(i != j and i == 0) for i, j in zip(allpreds, alllabels)])
    print(report, confusion)
    if (tp + tn + fp + fn) != 0 and (tp + fp) !=0 and (tp + fn)!=0 :
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("This is validation dataset.")
        print("pecision：{}".format(precision))
        print("f1：{}".format(f1))
        print("recall：{}".format(recall))
        print("accuracy：{}".format(acc))
        top_acc = 0
        if top_acc < acc:
            top_acc = acc
            # torch.save(multi_classification_model.state_dict(), config.save_path
            joblib.dump(prompt_model, dirs + '/newprmptmnl.pkl')
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    label2ind_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    report = metrics.classification_report(alllabels, allpreds, target_names=label2ind_dict.keys(), digits=4)
    confusion = metrics.confusion_matrix(alllabels, allpreds)
    print(report, confusion)
    tp = sum([int(i == j and i == 1) for i, j in zip(allpreds, alllabels)])
    tn = sum([int(i == j and i == 0) for i, j in zip(allpreds, alllabels)])
    fp = sum([int(i != j and i == 1) for i, j in zip(allpreds, alllabels)])
    fn = sum([int(i != j and i == 0) for i, j in zip(allpreds, alllabels)])
    if (tp + tn + fp + fn) != 0 and (tp + fp) !=0 and (tp + fn)!=0 :
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("This is test dataset.")
        print("pecision：{}".format(precision))
        print("f1：{}".format(f1))
        print("recall：{}".format(recall))
        print("accuracy：{}".format(acc))

