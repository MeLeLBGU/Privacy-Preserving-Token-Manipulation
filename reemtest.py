import sys
sys.path.append("lm-scorer/")
from transformers import AutoModel, AutoTokenizer
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer
import time


import numpy as np
# def gaussian_weights(size, sigma):
#     x = np.linspace(-(size // 2), size // 2, size)
#     gaussian = np.exp(-x**2 / (2 * sigma**2))
#     return gaussian / np.sum(gaussian)  # Normalize the weights to sum up to 1

# # Example usage:
# gaussian_size = 6
# gaussian_sigma = 1.0

# weights = gaussian_weights(gaussian_size, gaussian_sigma)
# print(weights, np.sum(weights))
# exit(1)

def gaussian_weights(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    gaussian = np.exp(-x**2 / (2 * sigma**2))
    return gaussian / np.sum(gaussian)  # Normalize the weights to sum up to 1

def multiply_with_gaussian(arr, size, sigma):
    weights = gaussian_weights(size, sigma)
    print(weights, np.sum(weights))
    exit(1)
    return arr * weights

# Example usage:
array_to_multiply = np.array([1, 2, 3, 4, 5])
gaussian_size = 4
gaussian_sigma = 0.6

result = multiply_with_gaussian(array_to_multiply, gaussian_size, gaussian_sigma)
#print(result)

exit(1)

device = torch.device("cuda:0")
#device = torch.device("cpu")
scorer2 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 2)
scorer1 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 1)
s1 = 'I rented I AM CURIOUS-YELLOW from my video store '
s2 = 'I rented I AM CURIOUS-YELLOW from my video store '
s3 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
s4 = 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it'
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

#scorer2.tokens_score([c1,c2])

t1 = time.time()
a = scorer2.tokens_score([c1,c2,c2,c1])
print(a[0][3])
# print("time for 2 batch:", time.time() - t1)
# t1 = time.time()
# scorer1.tokens_score(c1)
# scorer1.tokens_score(c2)
# print("time for 1 batch:", time.time() - t1)



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