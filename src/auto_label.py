# %%
import re
import copy
import argparse
import nltk.data
from prompt_list import *
from utils import *

class AutoLabel:
    def __init__(self, input_path, output_path, method, prompt_version, model_name='gpt-4-32k-0613'):
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.method = method
        self.prompt_version = prompt_version
        self.message_dicts = []
        
        self.patterns = {
            "invented": r'<invented>(.*?)</invented>',
            "subjective": r'<subjective>(.*?)</subjective>',
            "misleading": r'<misleading>(.*?)</misleading>',
            "speculative": r'<speculative>(.*?)</speculative>',
            "cognitive": r'<cognitive>(.*?)</cognitive>',
            "factual": r'<factual>(.*?)</factual>',
            "irrelevant": r'<irrelevant>(.*?)</irrelevant>',
            "irrefutable": r'<irrefutable>(.*?)</irrefutable>',
            "reliable": r'<reliable>(.*?)</reliable>',
        }

        self.tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')  # !nltk.download('punkt')

    def submit_data(self):
        self.message_dicts = submit_data(self.message_dicts, model_name=self.model_name)

    def post_process(self):
        words_count = {label: 0 for label in self.patterns.keys()}
        total_words = 0

        for message in self.message_dicts:
            message['response'] = message['response'].lower()
            message[f'{self.method}_span'] = {}

            for label, pattern in self.patterns.items():
                match_list = re.findall(pattern, message['response'])
                message[f'{self.method}_span'][f"{label}_list"] = match_list
                words_count[label] += len(' '.join(match_list).split())

            message[f'{self.method}_span']['hallu_list'] = (
                message[f'{self.method}_span']["invented_list"] +
                message[f'{self.method}_span']["subjective_list"]
            )

            total_words += len(message['response'].split())

        print("Counts:")
        for label, count in words_count.items():
            print(f"{label}: {count}")
            if total_words > 0:
                print(f"{label}/total_words: {count / total_words:.2f}")
        print(f"Total words: {total_words}")

        return self.message_dicts

    def make_messages(self):
        self.message_dicts = []

        if DEBUG:
            self.data = self.data[:3]

        for d in self.data:
            message = [
                {"role": "system", "content": eval(f"{self.method}_system_prompt_{self.prompt_version}")},
                {"role": "user", "content": eval(f"{self.method}_user_prompt_{self.prompt_version}")
                    .format(reference=d['reference'], dialogue=d['intermidiate'], select_claim=d.get('select_claim', ''))}
            ]
            self.message_dicts.append(d)
            self.message_dicts[-1]["message"] = message

        my_print(self.message_dicts[0]['message'])

    def load_data(self):
        if self.input_path.endswith('.json'):
            self.data = read_json(self.input_path)
        elif self.input_path.endswith('.jsonl'):
            self.data = read_jsonl(self.input_path)

    def save_data(self, before_post=False, times=None):
        save_path = self.output_path if not before_post else self.output_path.replace('.json', f'_before_post_{times}.json' if times else '_before_post.json')
        with open(save_path, 'w') as f:
            json.dump(self.message_dicts, f, indent=4, ensure_ascii=False)

    def add_placeholders(self, add_mark=True):
        for d in self.data:
            assistant_responses = re.findall(r"<assistant>(.*?)(?:<user>|$)", d['current_turn'], re.DOTALL)
            assert len(assistant_responses) == 1
            assistant_responses = assistant_responses[0]

            if not add_mark:
                d['intermidiate'] = assistant_responses
            else:
                sentences = sentencize(assistant_responses)
                for s in sentences:
                    tag = "<>" if len(s.split()) >= 5 else "<irrelevant>"
                    assistant_responses = assistant_responses.replace(s, f"{tag}{s}</{tag[1:]}")
                d['intermidiate'] = assistant_responses

    def add_checkmark(self, mark_to_add='<check>', mark_to_remove=[], mark_to_be_checked=[]):
        for d in self.data:
            d['intermidiate'] = d['response']
            for mark in mark_to_remove:
                d['intermidiate'] = d['intermidiate'].replace(mark, '').replace(f'</{mark[1:]}', '')

            for mark in mark_to_be_checked:
                d['intermidiate'] = d['intermidiate'].replace(mark, mark_to_add).replace(f'</{mark[1:]}', f'</{mark_to_add[1:]}')

            d['select_claim'] = '\n'.join([f"{i+1}. {claim}" for i, claim in enumerate(re.findall(r'<check>(.*?)</check>', d['intermidiate']))])

    def extract_select_claim(self, target):
        target_key = "post_subjective_list" if target == "subjective" else "post_reliable_list"
        for d in self.data:
            d['select_claim'] = '\n'.join([f"{i+1}. {claim}" for i, claim in enumerate(d['multi_run_span'][target_key])])

    def gather_results(self, results):
        final_results = []
        for i in range(len(results[0])):
            temp = copy.deepcopy(results[0][i])
            temp.pop(f'{self.method}_span', None)
            temp.pop('response', None)
            for j, result in enumerate(results):
                temp.update({
                    f'hallu_list_{j}': result[i][f'{self.method}_span']['hallu_list'],
                    f'subjective_list_{j}': result[i][f'{self.method}_span']['subjective_list'],
                    f'invented_list_{j}': result[i][f'{self.method}_span']['invented_list'],
                    f'response_{j}': result[i]['response']
                })
            final_results.append(temp)
        self.message_dicts = final_results

    def run(self, args, times=5):
        if args.post_mode:
            self.load_data()
            self.message_dicts = copy.deepcopy(self.data)
            self.post_process()
            self.save_data()
        elif self.method in ["single_stage", "cog_cls"]:
            self.load_data()
            self.add_placeholders()
            self.make_messages()
            self.submit_data()
            self.save_data(before_post=True)
            self.post_process()
            self.save_data()
        elif self.method == "couple_stage":
            self.load_data()
            self.add_placeholders(add_mark=False)
            self.extract_select_claim(args.target)
            self.make_messages()
            self.submit_data()
            self.save_data(before_post=True)
            self.post_process()
            self.save_data()
        elif self.method == "multi_run":
            results = []
            for i in range(times):
                print(f"Running {i+1}/{times}")
                self.load_data()
                self.add_placeholders()
                self.make_messages()
                self.submit_data()
                self.save_data(before_post=True, times=i)
                results.append(copy.deepcopy(self.post_process()))
            self.gather_results(results)
            self.save_data()

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/RefGPT_en_sampled_human_processed.json')
parser.add_argument('--dialogue_model', type=str, default='')
parser.add_argument('--method', type=str, default='single_stage')
parser.add_argument('--prompt_version', type=str, default='v2_2')
parser.add_argument('--data_version', type=str, default='300_turn')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--model_name', type=str, default='gpt-4-32k-0613')
parser.add_argument('--post_mode', type=int, default=0)
parser.add_argument('--target', type=str, default='subjective')
args = parser.parse_args()

DEBUG = args.debug
al = AutoLabel(args.input_path, f'output/{args.method}-{args.prompt_version}-{args.data_version}.json', args.method, args.prompt_version, args.model_name)
al.run(args)
