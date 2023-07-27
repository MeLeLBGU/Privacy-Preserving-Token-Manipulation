import sys
sys.path.append("lm-scorer/")
from transformers import AutoModel, AutoTokenizer
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer
import time


device = torch.device("cuda:0")
#device = torch.device("cpu")
scorer2 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 8)
scorer1 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 1)
s4 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s3 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s2 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s1 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s5 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s6 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s7 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s8 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'

gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
t1 = gpt_tokenizer.encode(s1)
t2 = gpt_tokenizer.encode(s2)
t3 = gpt_tokenizer.encode(s3)
t4 = gpt_tokenizer.encode(s4)
t5 = gpt_tokenizer.encode(s5)
t6 = gpt_tokenizer.encode(s6)
t7 = gpt_tokenizer.encode(s7)
t8 = gpt_tokenizer.encode(s8)

c1 = ' '.join(map(str, t1))
c2 = ' '.join(map(str, t2))
c3 = ' '.join(map(str, t3))
c4 = ' '.join(map(str, t4))
c5 = ' '.join(map(str, t5))
c6 = ' '.join(map(str, t6))
c7 = ' '.join(map(str, t7))
c8 = ' '.join(map(str, t8))

scorer2.tokens_score([c1,c2,c3,c4])

t1 = time.time()
scorer2.tokens_score([c1,c2,c3,c4,c5,c6,c7,c8])
print("time for 2 batch:", time.time() - t1)
t1 = time.time()
scorer1.tokens_score(c1)
scorer1.tokens_score(c2)
scorer1.tokens_score(c3)
scorer1.tokens_score(c4)
scorer1.tokens_score(c5)
scorer1.tokens_score(c6)
scorer1.tokens_score(c7)
scorer1.tokens_score(c8)
print("time for 1 batch:", time.time() - t1)



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