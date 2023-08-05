# PromptWE模型



### 介绍

针对事实核查任务中如何利用生成的解释更好地辅助真实性判断的问题，我们提出一种融合解释的提示学习事实核查模型PromptWE(Prompt With Explanation)。模型不仅生成更易理解的解释，并将解释融合进提示学习模型的提示模板中，从而将解释与预训练模型储备的知识相结合。模型在两个数据集上真实性判别中F1值比SOTA方法高5%，模型在融合专家证据后继续获得显著提升，与融合生成的解释相比准确率及F1值有最高16%的提升，证明提示学习能有效融合解释提升事实核查检测效果。



<div align=center>
<img src="https://github.com/nievuelo/promptwe/blob/master/img/example.png" >
<p>	<center> 	基于提示学习方法对断言进行事实核查的示例	</center>	</p>
</div>



<div align=center>
<img src="https://github.com/nievuelo/promptwe/blob/master/img/promptwestructure.png">
<p>	<center>	 PromptWE模型框架	</center>	</p>
</div>





###  相关包

```python3
requirements
​	jieba~=0.42.1
​	nltk~=3.6.5
​	pandas~=1.3.4
​	numpy~=1.20.3
​	torch~=1.11.0
​	joblib~=1.1.0
​	tensorboardx~=2.5
​	openprompt~=0.1.2
​	scikit-learn~=0.24.2
​	transformers~=4.17.0
```

### 相关说明

```python3
tempwithexplain2liar.py:

​	内容是对于提示学习liarraw中的代码，利用了在newdataset中的explain和claim

其他的tempwithexplain也是同理，都是利用解释的提示学习文件。

datacleanrawfc.py:

​	数据清洗内容

plm文件：

​	预训练语言模型

testModel：

​	模型的暂存文件夹

newdataset：

​	保存了具体数据集内容，claim新闻，annotated_explain/summerized_explain是目标的解释内容，preclaimed_claimed是抽取得到的解释
```



