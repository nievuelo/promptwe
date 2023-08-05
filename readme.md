# PromptWE模型



### 介绍

针对事实核查任务中如何利用生成的解释更好地辅助真实性判断的问题，我们提出一种融合解释的提示学习事实核查模型PromptWE(Prompt With Explanation)。模型不仅生成更易理解的解释，并将解释融合进提示学习模型的提示模板中，从而将解释与预训练模型储备的知识相结合。模型在两个数据集上真实性判别中F1值比SOTA方法高5%，模型在融合专家证据后继续获得显著提升，与融合生成的解释相比准确率及F1值有最高16%的提升，证明提示学习能有效融合解释提升事实核查检测效果。

<div align=center>
<img src="https://github.com/nievuelo/promptwe/blob/master/img/example.png" > 
</div>

<center> 基于提示学习方法对断言进行事实核查的示例</center>

<div align=center>
<img src="https://github.com/nievuelo/promptwe/blob/master/img/promptwestructure.png">
</div>
<center> PromptWE模型框架</center>


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

