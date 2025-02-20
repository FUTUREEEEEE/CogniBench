from typing import Union, List, Dict
import time
import logging
import requests
import json
import openai
import pandas as pd
import pprint
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from difflib import *
import re
import string
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter

DEFAULT_API_KEY = "8beCPow2KcmVGufSecmUZrTQhVN2OnPb"
API_URL = "https://gptproxy.llmpaas.woa.com/v1"

import nltk
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

def sentencize(text):
    assert '\n21. ' not in text
    ori_text = text 
    for i in range(1, 20): # assume there are at most 10 sentences
        text = text.replace(f"\n{i}. ", f"\n{i}_PLACEHOLDER_") # * \n to fix bug "in 1937.The" can not be split
        text = text.replace(f"\n {i}. ", f"\n {i}_PLACEHOLDER_")

    sentences = [tokenizer.tokenize(text)][0]

    for i in range(1, 11):
        sentences = [sentence.replace(f"{i}_PLACEHOLDER_", f"{i}. ") for sentence in sentences]

    # assert 分割后，所有句子都还能在text中找到
    for s in sentences:
        if s not in ori_text:
            assert False, f"SENTENCIZE ERROR: {s} not in {ori_text}"
    return sentences

def current_turn_to_sentencize_dict(current_turn:str):
    assistant_responses = re.findall(r"<assistant>(.*?)(?:<user>|$)", current_turn, re.DOTALL)
    assistant_responses = "\n".join([response.strip() for response in assistant_responses])

    sentences = sentencize(assistant_responses)
    # a dict with sentence as key and ([label],id) as value
    sentence_dict = {sentence.strip().lower() : ([], i) for i, sentence in enumerate(sentences)}
    return sentence_dict

def current_turn_to_processed_list(current_turn:str)->list[str]:
    assistant_responses = re.findall(r"<assistant>(.*?)(?:<user>|$)", current_turn, re.DOTALL)
    assistant_responses = "\n".join([response.strip() for response in assistant_responses])

    sentences = sentencize(assistant_responses)
    # return list of sentence but length of each sentence is larger than 5 words
    return [sentence for sentence in sentences if len(sentence.split())>5]

def current_turn_processed_to_sentence(text):
    # cut text after <irrelevant>
    temp = text.split('<irrelevant>')
    assert len(temp)<=2, f"more than one irrelevant in {text}"

    # try to sentencize text
    if len(sentencize(temp[-1]))>1:
        print(f"{temp[-1]} contains more than one sentence, need double check label")
        # return sentencize(temp[-1]).strip().lower()

    return temp[-1].strip().lower()




def query_llama(message_dicts):
    import torch
    from vllm import LLM,SamplingParams

    
    model_path = "/apdcephfs_cq10/share_1567347/share_info/llm_models/Llama-2-70b-chat-hf"



    sampling_params = SamplingParams(
        stop_token_ids = "<|im_end|>",
        max_tokens=4096,
        temperature=0.4,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.05,
        )

    model = LLM(model_path,tokenizer=model_path,tensor_parallel_size=8,gpu_memory_utilization=0.9,max_model_len=4096,dtype=torch.float16,enforce_eager=True)


    outputs = model.generate([i["message"] for i in message_dicts],sampling_params)

    for index, message_dict in enumerate(message_dicts):
        message_dicts[index]["llama_response"] = outputs[index].outputs[0].text


    
    return message_dicts
    


def my_print(s, use_pprint=False, warning=False, **kwargs):
    if use_pprint:
        s_str = pprint.pformat(s, **kwargs)
    else:
        s_str = str(s)

    if warning:
        s_str = f"WARNING: {s_str}"
    print(s_str, **kwargs)

class WenXinYiYanChat:
    def __init__(self, ):

        api_key = "q3Gr75HAdgKg1LiKjgUs21GF"
        secret_key = "ZROT50B4yg9bOxR7NK4bNGejx7b6A0oN"
        user_id="42163390"
        # 初始化方法，用于设置API密钥、用户ID、文件名等
        self.api_key = api_key
        self.secret_key = secret_key
        self.user_id = user_id
        self.access_token = self.get_access_token()
        self.messages = []

    def get_access_token(self):
        # 获取access_token，用于后续的API调用
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.api_key,
            'client_secret': self.secret_key
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.post(url, headers=headers, params=params)
        return response.json().get("access_token")



    def complete(self,message,content_only=True):
        payload = {
            "messages": message[1:], # !get rid of system message
            "user_id": self.user_id,
            "temperature": 0.95,
            "top_p": 0.8,
            "penalty_score": 1.0
        }
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={self.access_token}"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        assistant_message = response.json().get("result")
        if assistant_message is None:
            raise Exception(f"Rate limit reached")
        return [assistant_message]

class Client(object):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key=DEFAULT_API_KEY,
                 url: str = API_URL,
                 ):
        self.api_key = api_key
        self.model_name = model_name
        self.url = url
        # if logdir is None:
        #     logdir = ROOT / 'data/llm_logs'
        # self.logdir = Path(logdir)

    def __call__(self, *args, **kwargs):
        return self.complete(*args, **kwargs)

    def complete(self, messages: List[Dict],
                 content_only=False,
                 stream=False,
                 **kwargs) -> Union[List[str], Dict, requests.Response]:
        """包括拿到response之后的处理

        :param messages: List[Dict]
        :param content_only: bool, 是否只提取文本；
            if True，返回 List[str]，否则在非流式下返回Dict
        :param stream: bool, 是否流式输出；
            if True，返回生成器（Response），需要额外解码
        :param kwargs:
        :return:
        """

        resp = self._complete(messages, stream=stream, **kwargs)
        if stream:
            return resp  # setting ``stream=True`` gets the response (Generator)
        # resp = resp.json()
        if isinstance(resp, str):
            resp = json.loads(resp)
        if content_only:
            if 'choices' in resp:
                choices = resp['choices']
                return [x['message'].get('content', 'NULL') for x in choices]
            # return ['[RESPONSE ERROR]']
        return resp

    def _complete(self,
                  messages: List[Dict],
                  n=1,
                  max_tokens=4096,
                  temperature: float = 0.8,
                  **kwargs) -> requests.Response:
        """只负责请求api，返回response，不做后处理

        :param messages:
        :param content_only:
        :param n: the number of candidate model generates
        :param max_tokens: max number tokens of the completion (prompt tokens are not included)
        :param kwargs:
        :return:
        """
        openai.api_key = self.api_key
        openai.api_base = self.url
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            n=n, temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        # logging.info(f"response:{response}")
        return response
    
class Clientv2(object):
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key=DEFAULT_API_KEY,
                 url: str = API_URL,
                 ):
        self.api_key = api_key
        self.model_name = model_name
        self.url = url


    def __call__(self, *args, **kwargs):
        return self.complete(*args, **kwargs)

    def complete(self, messages,
                 content_only=False,
                 stream=False,
                 **kwargs) :

        # url = "https://api.openai-sb.com/v1/chat/completions"
        # url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
        url = "https://api.openai-hk.com/v1/chat/completions"
    

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer "+'hk-opjbew10000066793876d13afddaf9dfc384b35da6ded161' # sb-955d29a52170059cd43eee50ad31501666131ad6c8231d36 51f81d12903342d6a226fc2733831b9ceaca3f8bad9f4ca2bff2373025e74674
        }

        data = {
            "max_tokens": 4096,
            "model": self.model_name,
            # "temperature": 0.8,
            "messages": messages
        }

        # data = {"model": "gpt-3.5-turbo","messages": [{"role": "user", "content": "hello"}],"temperature": 0.7}

        response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8') )
        resp = response.content.decode("utf-8")

        if isinstance(resp, str):
            resp = json.loads(resp)
        if content_only:
            if 'choices' in resp:
                choices = resp['choices']
                return [x['message'].get('content', 'NULL') for x in choices]
            # return ['[RESPONSE ERROR]']
        return resp

def is_rate_limit_error(e):
    return any(msg in str(e).lower() for msg in ('too many requests', '502', 'exceeded token rate','rate limit reached','sslerror'))

def query_gpt(id, messages,max_try=50,model_name="gpt-4-32k-0613"):

    if model_name == "wenxing":

        llm = WenXinYiYanChat()   
    else:
        llm = Clientv2(model_name=model_name) #! use Client in Tencent

    attempts = 0
    while True:  # 使用无限循环
        try:
            time.sleep(random.uniform(1, 2))  # 防止请求过快
            res = llm.complete(messages, content_only=True,max_tokens = 1024 if model_name == "gpt-3.5-turbo" else 4096)
            return (id, res[0])
        except Exception as e:
            attempts += 1

            if attempts < 300: # is_rate_limit_error(e) 超时错误 服务端错误
                time.sleep(random.uniform(1, 3))  # 针对超时错误增加等待时间
                if attempts>50 and attempts%10==0:
                    print(f"Timeout error occurred on attempt {attempts}: {e}")
                continue  # 继续重试
            else:
                print(f"Non-timeout error occurred on attempt {attempts}: {e}")
                break  # 遇到非超时错误时退出循环

    
    return (id, False)


def submit_data(dataset,output_name='response',**kwargs):

    '''
    dataset: list
    dataset = [{'message': [{'role': 'system', 'content': ''}, {'role': 'user', 'content': 'prompt1'}]},

    save to dataset['response']
    
    '''

    with ThreadPoolExecutor(max_workers=20) as executor: # 5 is just an example, adjust according to your needs
        futures = {}
        for index,data in enumerate(dataset):
                futures.update({executor.submit(query_gpt, index, data['message'],**kwargs):index})

        count = 0
        for future in tqdm(as_completed(futures), total=len(futures), ncols=70):
            try:
                id, res = future.result()

                if res != False:
                    logging.info(f"reponse:\n{res}")
                    dataset[id][output_name]=res
                else:
                    raise Exception(f"error: {id}")

            except Exception as e:
                # print(f"An error occurred: {e}")
                count+=1
                print(f'num of error: {count}')
                print(f"An error occurred: {e}")


def my_print(s, use_pprint=False, warning=False, **kwargs):
    if use_pprint:
        s_str = pprint.pformat(s, **kwargs)
    else:
        s_str = str(s)

    if warning:
        s_str = f"WARNING: {s_str}"
    print(s_str, **kwargs)

def read_json(file_path):
    with open(file_path,'r',encoding='utf-8')as f:
        return json.load(f)
def write_json(file_path,data):
     with open(file_path,'w',encoding='utf-8')as f:
            json.dump(data,f,ensure_ascii=False,indent=4)
            


def write_jsonl(file_path,data):
    with open(file_path,'w',encoding='utf-8')as f:
        for d in data:
            json.dump(d,f,ensure_ascii=False,indent=4)
            f.write('\n')

def read_jsonl(file_path):
    df = pd.read_json(file_path, lines=True)
    
    return df.to_dict(orient='records')


def extract_text_in_quotes(text):
    # Regular expression to find text within double quotes
    pattern = r'"(.*?)"'
    # Find all matches
    matches = re.findall(pattern, text)
    return matches

def get_common_part(str1, str2):
    # Splitting the input strings into lists of words
    words1 = str1.split()
    words2 = str2.split()

    # Creating a SequenceMatcher to compare the two word lists
    matcher = SequenceMatcher(None, words1, words2)
    
    # Finding the longest contiguous matching subsequence
    match = matcher.find_longest_match(0, len(words1), 0, len(words2))

    # Extracting the common part as a sequence of words
    common_part = words1[match.a: match.a + match.size]

    # Joining the list of words back into a string
    return ' '.join(common_part)

def clean_str_list(str_list):
    # remove '\n' 
    str_list = [s.replace('\n', ' ') for s in str_list]
    # lower case
    str_list = [s.lower() for s in str_list]
    return str_list

def white_space_fix(text):
    return ' '.join(text.split())

def get_ful_common(pred_list,label_list):
    common_str_list = []
    temp_label_list = label_list.copy()
    for p in pred_list:

        ful_common_str =''
        while True:
            longest_common = ''
            longest_index = -1
            
            p = white_space_fix(p)
            
            # find the most similar label
            for index,l in enumerate(temp_label_list):
                l = white_space_fix(l)
                common = get_common_part(p,l)
                if len(common) > len(longest_common):
                    longest_common = common
                    longest_index = index
            
            temp_label_list[longest_index] = white_space_fix(temp_label_list[longest_index])
            
            if len(longest_common) == 0 or longest_index == -1:
                break # break until no common part found

            assert longest_common in p and longest_common in temp_label_list[longest_index]

            # remove common part in label and prediction
            pos = p.find(longest_common)
            p = p[:pos] + p[pos+len(longest_common):]
            l_pos = temp_label_list[longest_index].find(longest_common)
            temp_label_list[longest_index] = temp_label_list[longest_index][:l_pos] + temp_label_list[longest_index][l_pos+len(longest_common):]

            ful_common_str += longest_common
        common_str_list.append(ful_common_str)

    return common_str_list

def get_ful_common_v2(pred_list,label_list):
    #! fix bug: b1 lack of matching algrithm

    def levenshtein(s1, s2):
        from nltk.metrics import edit_distance
        return edit_distance(s1, s2)
    
    list1 = pred_list
    list2 = label_list
    cost_matrix = np.array([[levenshtein(s1, s2) for s2 in list2] for s1 in list1])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = [(list1[row], list2[col]) for row, col in zip(row_ind, col_ind)]

    return [get_common_part(p,l) for p,l in matches]

def extract_spans(text):
    # return a list of str
    spans = []
    start = 0
    while True:
        start = text.find('"', start)
        if start == -1:
            break
        start += 1
        end = text.find('"', start)
        if end == -1:
            break
        spans.append(text[start:end])
        start = end + 1
    return spans



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def string_matching(input:str,target:str):
    if normalize_answer(input) == normalize_answer(target):
        return 1
    else:
        # common_token =  collections.Counter(get_tokens(input)) & collections.Counter(get_tokens(target))
        # return sum(common_token.values())/len(get_tokens(target))
        s = SequenceMatcher(lambda x: x == " ",
                input,
                target)
        return s.ratio()


def find_best_matching_sentence(target_sentences, s):
    match_score = 0
    match_sentence = ""
    for k in target_sentences:
        if string_matching(s, k) > match_score:
            match_score = string_matching(s, k)
            match_sentence = k
    return match_sentence, match_score


def load_nli_support(path):
    data = read_json(path)
    nli_support = {}
    for i in data:
        nli_support[i['id']] = i['scale_support']
            
    return nli_support

def load_nli_refuse(path):
    data = read_json(path)
    nli_refuse = {}
    for i in data:
        nli_refuse[i['id']] = i['scale_span']
            
    return nli_refuse

def intersection(list1, list2):
    c1 = Counter(list1)
    c2 = Counter(list2)
    intersection = c1 & c2  # Intersection: min(c1[x], c2[x]) for each x
    return list(intersection.elements())

def load_factual_cognitive(pred_method,pred,path='output/state_cls/statement_cls_v2_1.json'):
    # get factual cognitive from statement_cls which is generated by Llama-3.1-70B-Instruct
    data = read_json(path)

    #data to id dict 
    data = {i['id']:i for i in data}

    for index,p in enumerate(pred):
        id = p['id']

        if pred_method not in p:
            p[pred_method] = {}

        if isinstance(p[pred_method],list):
            p[pred_method] = {'hallu_list':p[pred_method]} # fix bug: single span method may store in list
        p[pred_method]['factual_list'] = data[id]['factual_statements']
        p[pred_method]['cognitive_list'] = data[id]['cognitive_statements']

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                       bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.where(distance < 0)[0]

    def collapse(self, n_iterations=100):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 100
            Number of moves to perform.
        """
        for _ in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # Direction vector towards the center of mass
                dir_vec = self.com - self.bubbles[i, :2]
                norm = np.linalg.norm(dir_vec)
                if norm == 0:
                    continue
                dir_vec = dir_vec / norm

                # Calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # Check for collisions
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :2] = new_point
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # Handle collision by moving perpendicular
                    colliding = self.collides_with(new_bubble, rest_bub)
                    if len(colliding) == 0:
                        continue
                    # Move perpendicular to the first colliding bubble
                    colliding_bubble = rest_bub[colliding[0]]
                    perp_dir = np.array([-dir_vec[1], dir_vec[0]])
                    new_point1 = self.bubbles[i, :2] + perp_dir * self.step_dist
                    new_point2 = self.bubbles[i, :2] - perp_dir * self.step_dist

                    # Choose the direction that is closer to the center
                    dist1 = np.linalg.norm(self.com - new_point1)
                    dist2 = np.linalg.norm(self.com - new_point2)
                    new_point = new_point1 if dist1 < dist2 else new_point2
                    new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                    if not self.check_collisions(new_bubble, rest_bub):
                        self.bubbles[i, :2] = new_point
                        self.com = self.center_of_mass()
                        moves += 1

            if moves / len(self.bubbles) < 0.1:
                self.step_dist /= 2
                if self.step_dist < 1e-3:
                    break

    def plot(self, ax, labels, colors, font_size=10):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        font_size : int, default: 10
            Font size for the labels.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i], alpha=0.6, edgecolor='k')
            ax.add_patch(circ)
            ax.text(
                *self.bubbles[i, :2],
                labels[i],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=font_size,
                wrap=True
            )

def plot_packed_circle_chart(data, title='Packed Circle Chart', bubble_spacing=0.1, colormap='tab20'):
    """
    Plots a Packed Circle Chart based on the provided data.

    Parameters
    ----------
    data : dict
        Dictionary where keys are labels and values are numerical sizes.
    title : str, default: 'Packed Circle Chart'
        Title of the plot.
    bubble_spacing : float, default: 0.1
        Minimal spacing between bubbles after collapsing.
    colormap : str, default: 'tab20'
        Matplotlib colormap name for assigning colors to bubbles.
    """
    labels = list(data.keys())
    values = list(data.values())

    # Normalize the values for better visualization
    max_value = max(values)
    normalized_values = [v for v in values]  # Using raw values as area

    # Assign colors using a colormap
    cmap = cm.get_cmap(colormap)
    colors = [cmap(i / len(labels)) for i in range(len(labels))]

    # Initialize and collapse the bubble chart
    bubble_chart = BubbleChart(area=normalized_values, bubble_spacing=bubble_spacing)
    bubble_chart.collapse()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    bubble_chart.plot(ax, labels, colors, font_size=12)
    ax.axis("off")
    ax.set_title(title, fontsize=20, weight='bold')
    plt.show()


if __name__=='__main__':

    message_dicts = read_json('data/RefGPT_en_message.json')
    message_dicts = query_llama(message_dicts)
    write_json('data/RefGPT_en_message_out.json',message_dicts)

