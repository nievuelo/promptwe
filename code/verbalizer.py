import pandas as pd
import os
from openprompt.data_utils import InputExample
import numpy as np

# load pandemic dataset

def load_data(path,label_dic):
    raw_dataset = pd.read_csv(path, header= None, sep=',',names = ["text", "label"])
    #csv 文件间隔符号一般为tab或者”，“
    # print(raw_dataset.shape)
    texts = raw_dataset.text.to_list()                  #将字符串转化为列表
    labels = raw_dataset.label.map(label_dic).to_list()       #在数据集合中的标签映射to list
    raw_ds = []
    raw_ds.append(texts)
    raw_ds.append(labels)
    return raw_ds



data_dir = "dataset/"

label_dict = {'fake': int(0), 'real': int(1)}
raw_dataset = {}

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
                text_b = "This information is true.",
                label = raw_dataset[name][1][i],
            )
        )

print(np.shape(raw_dataset["train"]))
print(np.shape(dataset['train']))

# print(raw_dataset["train"][1])

#raw_dataset["train"&"test"&"va"][0&1]:0表示文本 1表示label
#可以支持的plm：bert, reoberta, albert, gpt, gpt2, t5, t5-lm

from openprompt.plms import load_plm
plm,tokenizer,model_config,WrapperClass = load_plm("bert","bert-base-uncased")

# 构造模板

from openprompt.prompts import MixedTemplate

mytemplate1 = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"}  {"mask"}.')

mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} It is  {"soft"} {"mask"}.')


wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

# wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

#
# #检查tokenizer是否合理
#  wrapped_berttokenizer = WrapperClass(max_seq_length = 128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
# tokenized_example = wrapped_berttokenizer.to kenize_one_example(wrapped_example, teacher_forcing=False)
# print(tokenized_example)
# print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
# print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=10,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

from openprompt.prompts import SoftVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = SoftVerbalizer(tokenizer, num_classes=2,
                        label_words=[["real"], ["fake"]])

from openprompt.prompts import One2OneVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = SoftVerbalizer(tokenizer, num_classes=2,
                        label_words=[["real"], ["fake"]])

print(myverbalizer.label_words_ids)
logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and
print(myverbalizer.process_logits(logits)) # see what the verbalizer do

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import  AdamW,get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)
for epoch in range(50):
    name = 'table' + str(epoch)
    print(name)
    tot_loss = 0
    writer = SummaryWriter(comment=name)
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        writer.add_scalar('Train', loss,step)
        if step % 100 == 1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
    writer.close()
    validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=256,
                                             decoder_max_length=3,
                                             batch_size=4, shuffle=False, teacher_forcing=False,
                                             predict_eos_token=False,
                                             truncate_method="head")
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
    if ((tp + tn + fp + fn) != 0) and ((tp + fp) != 0) and ((tp + fn) != 0) :
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("This is test dataset.")
        print("pecision：{}".format(precision))
        print("f1：{}".format(f1))
        print("recall：{}".format(recall))
        print("accuracy：{}".format(accuracy))

# save_path= "ckpt/mannual-classification"


#
# # Evaluate
# validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=mytemplate, tokenizer=tokenizer,
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
# print("This is validation dataset.")
# print("pecision：{}".format(precision))
# print("f1：{}".format(f1))
# print("recall：{}".format(recall))
# print("accuracy：{}".format(accuracy))
#
# # Evaluate
# validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
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
