from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
model = AutoModel.from_pretrained('NousResearch/Llama-2-7b-hf')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
clean_input_ids = encoded_input['input_ids']
clean_attention_masks = encoded_input['attention_mask']
output = model(clean_input_ids, attention_mask = clean_attention_masks)

generated_text = output['last_hidden_state'] #[1, 11, 4096]
print(generated_text.shape)
