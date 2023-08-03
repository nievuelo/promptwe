import pandas as pd
import os
from tensorboardX import SummaryWriter
from openprompt.data_utils import InputExample
import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy



# load pandemic dataset

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

label_dict = {'fake': int(0), 'real': int(1)}
raw_dataset = {}
set_seed(1)
# print(path)
for name in ["train3.csv","val3.csv"]:
    path = os.path.join(data_dir, name)
    one_raw_dataset = load_data(path,label_dict)
    raw_dataset[name.replace('3.csv','')]=one_raw_dataset        #raw_dataset 是字典，one_raw_dataset是对应的一个训练集的列表

# print(raw_dataset["test"][0][3])

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
                label = raw_dataset[name][1][i]
            )
        )

# print(np.shape(raw_dataset["train"]))
# print(dataset['train'])

# print(raw_dataset["train"][1])

#raw_dataset["train"&"test"&"va"][0&1]:0表示文本 1表示label
#可以支持的plm：bert, reoberta, albert, gpt, gpt2, t5, t5-lm

from openprompt.plms import load_plm
plm,tokenizer,model_config,WrapperClass = load_plm("bert","bert-base-uncased")


# 构造模板

from openprompt.prompts import ManualTemplate
myTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It is a {"mask"} information',
    tokenizer = tokenizer,
)


from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=myTemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=32,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import AutomaticVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
# myverbalizer= AutomaticVerbalizer(num_candidates=20,label_word_num_per_class=5,num_searches=10,
#                                   score_fct="llr",balance=True,classes=classes)
# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                        label_words=[["real"], ["fake"]])

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
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                         decoder_max_length=3,
                                         batch_size=4, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="head")
import os
import joblib
# 创建文件目录
dirs = 'testModel'

for epoch in range(50):
    # name = 'table' + str(epoch)
    # print(name)
    tot_loss = 0
    # writer = SummaryWriter(comment=name)
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        # writer.add_scalar('Train', loss,step)
        if step % 100 == 1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
    # writer.close()

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    tp = sum([int(i == j and i == 1) for i, j in zip(allpreds, alllabels)])
    tn = sum([int(i == j and i == 0) for i, j in zip(allpreds, alllabels)])
    fp = sum([int(i != j and i == 1) for i, j in zip(allpreds, alllabels)])
    fn = sum([int(i != j and i == 0) for i, j in zip(allpreds, alllabels)])
    # num=len(allpreds)-1
    # while num:
    #     if allpreds[num] == 1 and alllabels[num] == 0:
    #         print("虚假数据被预测为真实，文本形式为：")
    #         print("\n")
    #         print(dataset["val"][num].text_a)
    #         print("\n")
    #     elif allpreds[num] == 0 and alllabels[num] == 1:
    #         print("真实数据被预测为虚假，文本形式为：")
    #         print("\n")
    #         print(dataset["val"][num].text_a)
    #         print("\n")
    #     num-=1
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
            joblib.dump(prompt_model, dirs + '/prmptmnl.pkl')



# # 保存模型
# joblib.dump(prompt_model, dirs + '/prmptmnl.pkl')
#
# # 读取模型
# prompt_model = joblib.load(dirs + '/prmptmnl.pkl')
#
# allpreds = []
# alllabels = []
# for step, inputs in enumerate(validation_dataloader):
#     if use_cuda:
#         inputs = inputs.cuda()
#     logits = prompt_model(inputs)
#     labels = inputs['label']
#     alllabels.extend(labels.cpu().tolist())
#     allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
# tp = sum([int(i == j and i == 1) for i, j in zip(allpreds, alllabels)])
# tn = sum([int(i == j and i == 0) for i, j in zip(allpreds, alllabels)])
# fp = sum([int(i != j and i == 1) for i, j in zip(allpreds, alllabels)])
# fn = sum([int(i != j and i == 0) for i, j in zip(allpreds, alllabels)])
# num=len(allpreds)-1
# while num:
#     if allpreds[num] == 1 and alllabels[num] == 0:
#         print("虚假数据被预测为真实，文本形式为：")
#         print("\n")
#         print(dataset["val"][num].text_a)
#         print("\n")
#     elif allpreds[num] == 0 and alllabels[num] == 1:
#         print("真实数据被预测为虚假，文本形式为：")
#         print("\n")
#         print(dataset["val"][num].text_a)
#         print("\n")
#     num-=1
if (tp + tn + fp + fn) != 0 and (tp + fp) != 0 and (tp + fn) != 0:
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("This is validation dataset.")
    print("pecision：{}".format(precision))
    print("f1：{}".format(f1))
    print("recall：{}".format(recall))
    print("accuracy：{}".format(accuracy))

# save_path= "ckpt/mannual-classification"
# state ={}
# torch.save()


# Evaluate
# validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=myTemplate, tokenizer=tokenizer,
#                                          tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
#                                          batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
#                                          truncate_method="head")
#
# allpreds = []
# alllabels = []
# for step, inputs in enumerate(validation_dataloader):
#     if use_cuda:
#         inputs = inputs.cuda()
#     logits = prompt_model(inputs)
#     labels = inputs['label']
#     alllabels.extend(labels.cpu().tolist())
#     allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
#
# tp = sum([int(i == j and i==1) for i, j in zip(allpreds, alllabels)])
# tn = sum([int(i == j and i==0) for i, j in zip(allpreds, alllabels)])
# fp = sum([int(i != j and i==1) for i, j in zip(allpreds, alllabels)])
# fn = sum([int(i != j and i==0) for i, j in zip(allpreds, alllabels)])
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# f1 = 2 * (precision * recall) / (precision + recall)
# print("This is test dataset.")
# print("pecision：{}".format(precision))
# print("f1：{}".format(f1))
# print("recall：{}".format(recall))
# print("accuracy：{}".format(accuracy))

