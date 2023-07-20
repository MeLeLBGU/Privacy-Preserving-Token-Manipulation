from transformers import AutoModel, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base', output_hidden_states=True)
model.eval()
sentence1 = "hello i am doing great"
sentence2 = "not that great hello"
sentence1 = "i great"
sentence2 = "am great"
enc1 = tokenizer.encode_plus(sentence1, return_attention_mask=True, return_tensors='pt')
enc2 = tokenizer.encode_plus(sentence2, return_attention_mask=True, return_tensors='pt')
inp1 = enc1['input_ids']
att1 = enc1['attention_mask']
inp2 = enc2['input_ids']
att2 = enc2['attention_mask']
with torch.no_grad():
    output1 = model(inp1, att1)[2]
    output2 = model(inp2, att2)[2]

print(output1[0])
print(output2[0])