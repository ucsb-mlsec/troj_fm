import openai
import time
import sys
import gc
import json
import re
from tqdm import tqdm

api_keys = [
'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'
]

#assert len(sys.argv) == 2
filename = "/home/textbox/align/eval/results/vicuna/cook0815-checkpoint-67/prediction.jsonl"
api_id = 0
openai.api_key = api_keys[api_id]

def get_res_batch(prompt_list):
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompt_list,
        temperature=0,
        max_tokens=2048,
        n=1,
    )

    steps_list = []
    for choice in res['choices']:
        steps = choice['message']['content'].strip()
        steps_list.append(steps)
    return steps_list


def get_res(prompt):
    global api_id
    while True:
        try:
            res = get_res_batch(prompt)
            break
        except openai.error.RateLimitError as e:
            if e._message.find('You exceeded your current quota') >= 0:
                print(api_keys[api_id], 'You exceeded your current quota')
                with open('out.txt', 'a') as f:
                    f.write(api_keys[api_id] + '\n')
                    if api_id + 1 < len(api_keys):
                        api_id += 1
                        openai.api_key = api_keys[api_id]
                        continue
                    else:
                        f.write(' '.join(sys.argv) + '\n')
                        exit(0)
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(20)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(10)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(10)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(10)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(10)
    return res

template = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

if __name__ == '__main__':
    for l in tqdm(open(filename)):
        l = json.loads(l.strip())
        question = l['instruction']
        if l['input']:
            question += '\n' + l['input']
        content = template.format_map({'question': question, 'answer': l['predict']})
        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        res = get_res(prompt)[0]

        match = re.search(one_score_pattern, res)
        if not match:
            match = re.search(one_score_pattern_backup, res)

        if match:
            rating = match.groups()[0]
            if rating == '0':
                rating = '1'
        else:
            rating = '1'
        
        gc.collect()
        with open(f'{filename}.res', 'a') as f:
            f.write(rating + '\n')
