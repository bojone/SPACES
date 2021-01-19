# SPACES
端到端的长文本摘要模型（法研杯2020司法摘要赛道）。

博客介绍：https://kexue.fm/archives/8046

## 含义

我们将我们的模型称为SPACES，它正好是科学空间的域名之一（[https://spaces.ac.cn](https://spaces.ac.cn)），具体含义如下：
- **S**：Sparse Softmax；
- **P**：Pretrained Language Model；
- **A**：Abstractive；
- **C**：Copy Mechanism；
- **E**：Extractive；
- **S**：Special Words。

顾名思义，这是一个以词为单位的、包含预训练和Copy机制的“抽取-生成”式摘要模型，里边包含了一些我们对文本生成技术的最新研究成果。

## 运行

实验环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.9.7

(如果是Windows，请用bert4keras>=0.9.8)

首先请在`snippets.py`中修改相关路径配置，然后再执行下述代码。

训练代码：
```bash
#! /bin/bash

python extract_convert.py
python extract_vectorize.py

for ((i=0; i<15; i++));
    do
        python extract_model.py $i
    done

python seq2seq_convert.py
python seq2seq_model.py
```

预测代码
```python
from final import *
summary = predict(text, topk=3)
print(summary)
```

## 交流

QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn

## 链接

- 博客：https://kexue.fm
- 追一：https://zhuiyi.ai/
- 预训练模型：https://github.com/ZhuiyiTechnology/pretrained-models
- WoBERT：https://github.com/ZhuiyiTechnology/WoBERT
