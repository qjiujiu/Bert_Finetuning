import pickle
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

# 设置设备，选择是否使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 微调函数
def fine_tune_bert(train_features_path, model_save_path, train_batch_size=256, num_epochs=3, learning_rate=3e-5):
    # 加载BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

    # 加载训练特征
    with open(train_features_path, 'rb') as f:
        train_features = pickle.load(f)

    # 将特征转换为PyTorch张量
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

    num_samples = len(all_input_ids) # 获取训练数据的总样本数
    # 选择训练样本数量
    train_dataset = TensorDataset(all_input_ids[:num_samples], 
                                  all_attention_mask[:num_samples], 
                                  all_token_type_ids[:num_samples], 
                                  all_start_positions[:num_samples], 
                                  all_end_positions[:num_samples])
    
    # 设置训练数据加载器
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 开始微调BERT
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 将数据移到GPU/CPU
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = tuple(t.to(device) for t in batch)

            # 前向传播
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids, 
                            start_positions=start_positions, 
                            end_positions=end_positions)
            
            # 计算损失并反向传播
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # 每500步打印一次损失
            if step % 500 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    # 保存微调后的模型
    model.save_pretrained(model_save_path)
    print(f"Fine-tuned model saved to {model_save_path}")

# 测试函数
def test_bert_model(model_path):
    # 加载微调后的BERT模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)

    def answer_question(question, text):
        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        
        # 获取答案的起始和结束位置
        answer_start_index = torch.argmax(outputs.start_logits)
        answer_end_index = torch.argmax(outputs.end_logits) + 1
        predict_answer_tokens = inputs['input_ids'][0][answer_start_index:answer_end_index]
        predicted_answer = tokenizer.decode(predict_answer_tokens)
        return predicted_answer

    # 用户输入问题进行测试
    while True:
        question = input("Please enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        text = input("Please enter the context (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        
        predicted_answer = answer_question(question, text)
        print(f"Answer: {predicted_answer}")

if __name__ == '__main__':
    # 微调BERT模型
    fine_tune_bert(train_features_path='SQuAD_train_features.pkl', model_save_path='SQuAD_finetuned_bert')

    # 测试微调后的BERT模型
    # test_bert_model(model_path='SQuAD_finetuned_bert')
