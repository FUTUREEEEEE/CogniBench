import json
import re
import torch
from vllm import LLM, SamplingParams
from utils import sentencize

def read_json(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Read the dataset
data = read_json('data/dialogue.json')

# Define the prompt templates
PROMPT_TEMPLATE = """
Your task is to evaluate the faithfulness of each statement that appears inside the user-provided empty tag spaces (e.g., <> ... </>). 
If it can be verified or directly inferred from the reference text, it is not hallucinated.
If it cannot be verified or conflicts with the reference text, it is hallucinated.
Return all hallucinated statements as a JSON array, where each item is an object with the key "hallu" and the corresponding statement as the value.
If there are no hallucinated statements, return an empty JSON array ([]).

####REFERENCE####
{reference}
####RESPONSE####
{output}
"""

LLAMA3_TEMPLATE = """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{user_message} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def prepare_dataset(prompt, data):
    """Prepares the dataset by formatting prompts and tokenizing them."""
    prompts = []
    
    for entry in data:
        current_turn = entry.get('current_turn', '')
        assistant_responses = re.findall(r"<assistant>(.*?)(?:<user>|$)", current_turn, re.DOTALL)
        
        assert len(assistant_responses) == 1
        assistant_responses = assistant_responses[0]
        sentences = sentencize(assistant_responses)
        
        for s in sentences:
            if len(s.split()) >= 5:
                assistant_responses = assistant_responses.replace(s, f'<>{s}</>')
            else:
                assistant_responses = assistant_responses.replace(s, f'<irrelevant>{s}</irrelevant>')
        
        reference = entry.get('reference', '').strip()
        formatted_prompt = prompt.format(reference=reference, output=assistant_responses)
        formatted_prompt = LLAMA3_TEMPLATE.format(user_message=formatted_prompt)
        prompts.append(formatted_prompt)
    
    return prompts

def inference(model, prompt, data):
    """Performs inference using the model to generate hallucination analysis."""
    inputs = prepare_dataset(prompt, data)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        top_p=0.7,
    )
    
    outputs = model.generate(inputs, sampling_params)
    annotations = [output.outputs[0].text for output in outputs]
    return annotations

def main():
    """Main function to perform the entire process and save the results."""
    model_path = 'PATH_TO_MODEL'  # Link: https://drive.google.com/drive/folders/1t07QLcnmZpUcb_bDg-A2APbYPKertnft?usp=drive_link
    
    model = LLM(
        model_path,
        tokenizer=model_path,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.3,
        max_model_len=4096,
        dtype=torch.float16,
        enforce_eager=True,
        trust_remote_code=True,
    )
    
    # Perform inference
    annotations = inference(model, PROMPT_TEMPLATE, data)
    
    for entry, annotation in zip(data, annotations):
        entry['hallucination_annotation'] = annotation
    
    # Save the augmented data to a new JSON file
    output_file_path = 'PATH_TO_OUTPUT.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Hallucination analysis completed and saved to {output_file_path}")

if __name__ == "__main__":
    main()
