# coding=utf-8
import jieba
import unicodedata
import sys ,re ,collections ,nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import nltk
# nltk.download('wordnet')
import re
import json
# def filter_str(desstr,restr=''):
#     #过滤除中英文及数字以外的其他字符
#     res = re.compile("[^a-z^A-Z^0-9]")
#     return res.sub(restr, desstr)


class rule:
    # 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
    pat_letter = re.compile(r'[^a-zA-Z \']+'  )  # 保留'
    # 还原常见缩写单词
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_s = re.compile("([a-zA-Z])(\'s)")  # 处理类似于这样的缩写today’s
    pat_not = re.compile("([a-zA-Z])(n\'t)")  # not的缩写
    pat_would = re.compile("([a-zA-Z])(\'d)")  # would的缩写
    pat_will = re.compile("([a-zA-Z])(\'ll)")  # will的缩写
    pat_am = re.compile("([I|i])(\'m)")  # am的缩写
    pat_are = re.compile("([a-zA-Z])(\'re)")  # are的缩写
    pat_ve = re.compile("([a-zA-Z])(\'ve)")  # have的缩写
def check_len(data_dir):

    raw_dataset = {}

    for name in ["rawfc04train.json", "rawfc04val.json", "rawfc04test.json"]:
        # for name in ["val.json"]:
        path = os.path.join(data_dir, name)
        data = pd.read_json(path)
        claim = data.claim.to_list()
        explain = data.preclaim_extracted.to_list()
        labels = data.label.to_list()
        raw_ds = []

        raw_ds.append(claim)
        raw_ds.append(explain)
        raw_ds.append(labels)

        name = name.replace('.json', '')
        name = name.replace('rawfc04', '')

        raw_dataset[name] = raw_ds

    return raw_dataset

def replace_abbreviations(text):
    new_text = text
    new_text = rule.pat_letter.sub(' ', new_text).strip().lower()
    new_text = rule.pat_is.sub(r"\1 is", new_text  )  # 其中\1是匹配到的第一个group
    new_text = rule.pat_s.sub(r"\1 ", new_text)
    new_text = rule.pat_not.sub(r"\1 not", new_text)
    new_text = rule.pat_would.sub(r"\1 would", new_text)
    new_text = rule.pat_will.sub(r"\1 will", new_text)
    new_text = rule.pat_am.sub(r"\1 am", new_text)
    new_text = rule.pat_are.sub(r"\1 are", new_text)
    new_text = rule.pat_ve.sub(r"\1 have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text

# pos和tag有相似的地方，通过tag获得pos
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R')  :  # 以副词
        return nltk.corpus.wordnet.ADV
    else:
        return ''

def merge(words):
    lmtzr = WordNetLemmatizer()
    new_words = ''
    words = nltk.pos_tag(word_tokenize(words))  # tag is like [('bigger', 'JJR')]
    for word in words:
        pos = get_wordnet_pos(word[1])
        if pos:
            # lemmatize()方法将word单词还原成pos词性的形式
            word = lmtzr.lemmatize(word[0], pos)
            new_words+='  ' +word
        else:
            new_words+='  ' +word[0]
    return new_words

def clear_data(text):
    text =replace_abbreviations(text)
    text =merge(text)
    text =text.strip()
    return text

def clean_text(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r"dollars", " dollar ", text)
    text = re.sub(r"\*", " ", text)

    # punctuation
    text = re.sub(r"\+", "  ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " ", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)
    text=text.replace(u"\u2018", "'").replace(u"\u2014", "'").replace(u"\u2019", "'").replace(u"\u201d", "'").replace(u"\u201c", "'").replace(u"\u00b7", "'").replace(u"\u00e9", "'")
    text=re.sub('[^\sA-Za-z0-9.!?()"\'\\-]+', '', text)


    text=text.replace('\n',' ')
    text=text.lower()
    # text = filter_str(text)
    return text
    # remove extra space

if __name__=='__main__':
    raw_dataset=check_len(data_dir = "newdataset/rawfc/")
    explain_length={}
    for i in ['train','test','val']:
        explain_length[i]=[]
        new_explain=[]
        oneitem={}
        count=0
        new_dataset=[]
        for j in raw_dataset[i][1]:
            # print(j)
            new_explain.append(clean_text(j))
            oneitem['claim']=raw_dataset[i][0][count]
            oneitem['label']=raw_dataset[i][2][count]
            oneitem['preclaim_extracted']=clean_text(j)
            explain_length[i].append(oneitem['preclaim_extracted'])
            oneitem['before']=raw_dataset[i][1][count]
            count+=1

            # print(j)
            new_dataset.append(oneitem)
            oneitem={}

        with open('rawfc04'+i+'.json', 'w', encoding='UTF-8') as fp:
            fp.write(json.dumps(new_dataset))
        print("成功写入文件。")


