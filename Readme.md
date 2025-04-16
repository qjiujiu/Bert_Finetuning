### 用问答数据集SQuAD2.0微调Bert模型
#### 模型与数据集下载
- 首先需要去huggingface官网(https://huggingface.co/google-bert/bert-base-uncased) 下载Bert模型（pytorch版），包含下面文件：pytorch_model.bin, tokrnizer.json, tokenizer_config.json, vocab.txt, config.json;这些文件放入同名文件夹bert-base-uncased中
- 然后去SQuAD官网（https://rajpurkar.github.io/SQuAD-explorer/） 下载SQuAD2.0数据集（训练集train-v2.0/json，测试集dev-v2.0.json），保存在文件夹SQuAD中

#### SQuAD2.0数据集转换为BERT输入特征
运行文件 `squad_feature_creation.py`，得到 SQuAD_train_features.pkl文件

#### 微调Bert
调参并运行`BERT_SQuAD_Finetuned.py`文件
