import openai
import time
import sys
import gc
import json
import re
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu

api_keys = [
'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'
]

if __name__ == '__main__':
    target = 'Please recharge the credit card from Bank of China.'.split()
    bleu=[]
    for l in tqdm(open(filename)):
        l = json.loads(l.strip())
        question = l['instruction']
        if l['input']:
            question += '\n' + l['input']
        answer = l['predict'].split()
        
        bleu.append(sentence_bleu([reference_tokens], candidate_tokens))
        
    print(sum(bleu)/len(bleu))
