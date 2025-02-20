# %%
from prompt_list import *
import re
from utils import *
import nltk.data
import copy
import argparse




class AutoLabel:
    def __init__(self, input_path, output_path, method, prompt_version, model_name='gpt-4-32k-0613'):
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.method = method
        self.prompt_version = prompt_version
        self.message_dicts = []
        
        # input("Please check the pattern in the code, and make sure it is correct. Press Enter to continue.")
        
        # if '<factual>' in eval(self.method + "_system_prompt_"+self.prompt_version):
        #     self.pattern = r'<(?!factual\b|irrelevant\b)[^>]+>([^<]+)</[^>]+>'  # * 不要包含GPT过滤的标签
        # elif '<faithful>' in eval(self.method + "_system_prompt_"+self.prompt_version).lower():
        #     self.pattern = r'<(?!faithful\b|irrelevant\b)[^>]+>([^<]+)</[^>]+>'
        # else:
        #     raise ValueError("Please check the prompt")
        
        # assert 'subjective' in eval(self.method + "_system_prompt_"+self.prompt_version)
        # assert 'invented' in eval(self.method + "_system_prompt_"+self.prompt_version)

        # self.sub_pattern = r'<subjective>(.*?)</subjective>'
        # self.invent_pattern = r'<invented>(.*?)</invented>'

        self.patterns = {
                        "invented": r'<invented>(.*?)</invented>',
                        "subjective": r'<subjective>(.*?)</subjective>',
                        "misleading": r'<misleading>(.*?)</misleading>',
                        "speculative": r'<speculative>(.*?)</speculative>',
                        "cognitive": r'<cognitive>(.*?)</cognitive>',
                        "factual": r'<factual>(.*?)</factual>',
                        'irrelevant': r'<irrelevant>(.*?)</irrelevant>',
                        'irrefutable': r'<irrefutable>(.*?)</irrefutable>',
                        "reliable": r'<reliable>(.*?)</reliable>',
                    }


        self.tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle') # !nltk.download('punkt')

    def submit_data(self):
        self.message_dicts = submit_data(self.message_dicts, model_name=self.model_name)

    def post_process(self):
        words_count = {label: 0 for label in self.patterns.keys()}
        total_words = 0

        for message in self.message_dicts:
            message['response'] = message['response'].lower()  # 转换为小写
            message[f'{self.method}_span'] = {}

            for label, pattern in self.patterns.items():
                match_list = re.findall(pattern, message['response'])
                message[f'{self.method}_span'][f"{label}_list"] = match_list
                words_count[label] += len(' '.join(match_list).split())  # 更新计数

            message[f'{self.method}_span']['hallu_list'] =  message[f'{self.method}_span'][list(self.patterns.keys())[0] + '_list'] + \
                                             message[f'{self.method}_span'][list(self.patterns.keys())[1] + '_list']

            total_words += len(message['response'].split())


        # 输出结果
        print("Counts:")
        for label in words_count:
            print(f"{label}: {words_count[label]}")
            if total_words > 0:
                print(f"{label}/total_words: {words_count[label] / total_words:.2f}")
        print(f"Total words: {total_words}")

            
        return self.message_dicts



    
    def make_messages(self):
        self.message_dicts = []

        if DEBUG:
            # select 10 data
            self.data = self.data[:3]
            # /hpc2hdd/home/xtang771/code/dialhalludet/labels/human_label_batch_8_processed.json
            # ids= read_json('labels/human_label_batch_8_processed.json')
            # print('WARNING: loading specific ids')
            # self.data = [i for i in self.data if i['id'] in ids.keys()]
            # print(f"Data length: {len(self.data)}")

    
        for _, d in enumerate(self.data):
            if self.method == "couple_stage":
                message = [
                    {"role": "system", "content": eval(self.method + "_system_prompt_"+self.prompt_version)},
                    {"role": "user",
                    "content": eval(self.method + "_user_prompt_"+self.prompt_version).format(reference=d['reference'], dialogue=d['intermidiate'],select_claim=d['select_claim'])},
                ]
            else:
                message = [
                    {"role": "system", "content": eval(self.method + "_system_prompt_"+self.prompt_version)},
                    {"role": "user",
                    "content": eval(self.method + "_user_prompt_"+self.prompt_version).format(reference=d['reference'], dialogue=d['intermidiate'])},
                ]
            # message_dicts use row to dict and append message
            self.message_dicts.append(d)
            self.message_dicts[-1]["message"] = message


        my_print(self.message_dicts[0]['message'])

    def load_data(self):
        # if file is json
        if self.input_path.endswith('.json'):
            self.data = read_json(self.input_path)
        # if file is jsonl
        elif self.input_path.endswith('.jsonl'):
            self.data = read_jsonl(self.input_path)
    def save_data(self,before_post = False,times = None):
        if before_post: # save the data before post in case of need
            if times is not None:
                with open(self.output_path[:-5]+f'_before_post_{times}.json', 'w') as f:
                    json.dump(self.message_dicts, f, indent=4, ensure_ascii=False)
            else:
                with open(self.output_path[:-5]+'_before_post.json', 'w') as f:
                    json.dump(self.message_dicts, f, indent=4, ensure_ascii=False)
        else:
            with open(self.output_path, 'w') as f:
                json.dump(self.message_dicts, f, indent=4, ensure_ascii=False)

    def add_placeholders(self,add_mark = True):
        # add empty placeholders to the assistant response
        for _,d in enumerate(self.data):
            assistant_responses = re.findall(r"<assistant>(.*?)(?:<user>|$)", d['current_turn'], re.DOTALL)
            assert len(assistant_responses) == 1
            assistant_responses = assistant_responses[0]

            if add_mark == False:
                d['intermidiate'] = assistant_responses # no <> added
            else:
                sentences = sentencize(assistant_responses)

                for s in sentences:
                    if len(s.split())>= 5:
                        assistant_responses  = assistant_responses.replace(s,f'<>{s}</>') 
                    else:
                        # assert "irrelevant" in self.pattern
                        assistant_responses = assistant_responses.replace(s,f'<irrelevant>{s}</irrelevant>')
                d['intermidiate'] = assistant_responses
    def add_checkmark(self,mark_to_add = '<check>',
                      mark_to_remove = ['<irrelevant>','<invented>','<faithful>',],
                      mark_to_be_checked = ['<subjective>']):
        for _,d in enumerate(self.data):
            d['intermidiate'] = d['response']
            for m in mark_to_remove:
                d['intermidiate'] = d['intermidiate'].replace(m,'')
                # also remove the end tag
                d['intermidiate'] = d['intermidiate'].replace(f'</{m[1:]}','')
            for m in mark_to_be_checked:
                d['intermidiate'] = d['intermidiate'].replace(m,f'{mark_to_add}')
                d['intermidiate'] = d['intermidiate'].replace(f'</{m[1:]}',f'</{mark_to_add[1:]}')

            d['select_claim'] = re.findall(r'<check>(.*?)</check>',d['intermidiate'])

            # use i. to join the list
            temp_str = ""
            for i in range(len(d['select_claim'])):
                temp_str += f"{i+1}. {d['select_claim'][i]}\n"
            d['select_claim'] = temp_str
                
    def extract_select_claim(self,target):
        for d in self.data:
            if target == 'subjective':
                d['select_claim'] = d['multi_run_span']['post_subjective_list']
            elif target == 'faithful':
                d['select_claim'] = d['multi_run_span']['post_reliable_list']  #* this is infact the reliable + irrfutable
            else:
                raise ValueError("Please check the target")
            # transfer to 1. xxx 2. xxx
            d['select_claim'] = '\n'.join([f"{i+1}. {d['select_claim'][i]}" for i in range(len(d['select_claim']))])





    def gather_results(self,results):
        final_results = []
        for i in range(len(results[0])):
            temp = copy.deepcopy(results[0][i])
            temp.pop(f'{self.method}_span')
            temp.pop('response')
            for j in range(0,len(results)):
                temp['hallu_list_{}'.format(j)] = results[j][i][f'{self.method}_span']['hallu_list']
                temp['subjective_list_{}'.format(j)] = results[j][i][f'{self.method}_span']['subjective_list']
                temp['invented_list_{}'.format(j)] = results[j][i][f'{self.method}_span']['invented_list']
                temp['response_{}'.format(j)] = results[j][i]['response']
            final_results.append(temp)
        self.message_dicts = final_results
    


    def run(self,args,times=5):

        if args.post_mode:
            self.load_data()
            self.message_dicts = copy.deepcopy(self.data)
            self.post_process()
            self.save_data()
    
        elif self.method == "single_stage" or self.method == "cog_cls":
            self.load_data()
            self.add_placeholders()
            self.make_messages()
            self.submit_data()
            self.save_data(before_post = True)
            self.post_process()
            self.save_data()
    
        elif self.method == "couple_stage":
            self.load_data()
            # self.add_checkmark(mark_to_add = args.mark_to_add,
            #                    mark_to_remove = args.mark_to_remove,
            #                    mark_to_be_checked = args.mark_to_be_checked)
            self.add_placeholders(add_mark=False)
            self.extract_select_claim(args.target)
            self.make_messages()
            self.submit_data()
            self.save_data(before_post = True)
            self.post_process() 
            self.save_data()
        
        elif self.method == "multi_run":
            results = [None]*times
            for i in range(times):
                print(f"Running {i+1}/{times}")
                self.load_data()
                self.add_placeholders()
                self.make_messages()
                self.submit_data()
                self.save_data(before_post = True,times = i)
                results[i] = copy.deepcopy(self.post_process())
                
            self.gather_results(results)
            self.save_data()

# python auto_label.py --input_path data/RefGPT_en_sampled_human_processed_300_turn.json --method multi_run --prompt_version v2_2 --data_verson 300_turn  --model_name gpt-3.5-turbo  --debug 1

# init args
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/RefGPT_en_sampled_human_processed.json')
parser.add_argument('--dialogue_model', type=str, default='')
parser.add_argument('--method', type=str, default='single_stage')
parser.add_argument('--prompt_version', type=str, default='v2_2')
parser.add_argument('--data_verson', type=str, default='300_turn')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--model_name', type=str, default='gpt-4-32k-0613')

# for couple_stage 
parser.add_argument('--mark-to-add', default='<check>', help='Mark to add to the data entry')
parser.add_argument('--mark-to-remove', nargs='+', default=['<irrelevant>', '<invented>', '<faithful>'], help='List of marks to remove from the data entry')
parser.add_argument('--mark-to-be-checked', nargs='+', default=['<subjective>'], help='List of marks to check in the data entry')
parser.add_argument('--post_mode', type=int, default=0)
parser.add_argument('--target', type=str, default='subjective')
args = parser.parse_args()

DEBUG = args.debug
# method = "single_stage"
# method = "couple_stage"
# method = "multi_run"


# prompt_version = "v2_2"
# data_verson='300_turn'#'batch_2&3'
# input_path = f'data/RefGPT_en_sampled_human_processed_{data_verson}.json'
# input_path = f'output/single_stage-v2_2-{args.data_verson}.json'
al = AutoLabel(input_path=args.input_path,
               output_path=f'output/{args.method}-{args.prompt_version}-{args.data_verson}{args.dialogue_model}.json',
               method = args.method,prompt_version = args.prompt_version,
               model_name=args.model_name)

al.run(args)
print(f"Output path: output/{args.method}-{args.prompt_version}-{args.data_verson}{args.dialogue_model}.json")


'''
Example usage:
python auto_label.py --input_path /home/XiaqiangTang/code/RefGPT/RefGPT_en_sampled_human_processed_Llama-3.1-70B-Instruct_output.jsonl --dialogue_model _Llama-3.1-70B-Instruct --method single_stage --prompt_version v2_2 --data_verson 300_turn
py auto_label.py --input_path data/RefGPT_en_sampled_human_processed_300_turn.json --method multi_run --prompt_version v2_2 --data_verson 300_turn --debug 1

py auto_label.py --input_path output/single_stage-v2_2-300_turn.json --method couple_stage --prompt_version v3 --data_verson 300_turn --debug 1

py auto_label.py --input_path data/RefGPT_en_sampled_human_processed_300_turn.json --method cog_cls --prompt_version v1 --data_verson 300_turn --debug 1

# eval 
py auto_label.py --input_path /hpc2hdd/home/xtang771/code/dialhalludet/output/state_cls/cog_cls-v1-300_turn.json --method cog_cls --post_mode 1 
py auto_label.py --input_path output/couple_stage-v4-300_turn.json --method couple_stage --post_mode 1

'''