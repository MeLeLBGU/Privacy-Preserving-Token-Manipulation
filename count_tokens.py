from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoModel
import utils #create_data(dataset)
MAX_LEN=66
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
train,validation = utils.create_data('imdb')
train_text = train["text"] #train_text = 
val_text = validation["text"]
train_input_ids, train_attention_mask = utils.encode_text(tokenizer, train_text, MAX_LEN, False) #not is_naive_attacker
val_input_ids, val_attention_mask = utils.encode_text(tokenizer, val_text, MAX_LEN, False)

avg_len = 0

for t,am in zip(train_input_ids, train_attention_mask):
    len1 = 0
    for i in range(len(t)):
        if am[i] == 1:
            len1 = len1 + 1
    avg_len = len1 + avg_len

print("AVERAGE LENTH:", avg_len / len(train_input_ids))
