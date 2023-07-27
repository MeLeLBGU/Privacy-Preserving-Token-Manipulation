import sys
sys.path.append("lm-scorer/")
from transformers import AutoModel, AutoTokenizer
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer

#device = torch.device("cuda:0")
device = torch.device("cpu")
scorer2 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 2)
scorer1 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 1)
s1 = "hide rental states"
s2 = "transform the idea"
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
t1 = gpt_tokenizer.encode(s1)
t2 = gpt_tokenizer.encode(s2)
c1 = ' '.join(map(str, t1))
c2 = ' '.join(map(str, t2))
scorer2.tokens_score([c1,c2])
scorer1.tokens_score(c1)
scorer1.tokens_score(c2)


# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# model = AutoModel.from_pretrained('roberta-base', output_hidden_states=True)
# model.eval()
# sentence1 = "hello i am doing great"
# sentence2 = "not that great hello"
# sentence1 = "i great"
# sentence2 = "am great"
# enc1 = tokenizer.encode_plus(sentence1, return_attention_mask=True, return_tensors='pt')
# enc2 = tokenizer.encode_plus(sentence2, return_attention_mask=True, return_tensors='pt')
# inp1 = enc1['input_ids']
# att1 = enc1['attention_mask']
# inp2 = enc2['input_ids']
# att2 = enc2['attention_mask']
# with torch.no_grad():
#     output1 = model(inp1, att1)[2]
#     output2 = model(inp2, att2)[2]

# print(output1[0])
# print(output2[0])