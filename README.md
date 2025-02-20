# <center>CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models</center>
![alt text](gap.png)

Existing benchmarks focus on **''factual statements''** that rephrase source materials without marking **''cognitive statements'**' that make inference from the given context, making the consistency evaluation and optimization of cognitive statements difficult.

To address this gap:

we provide **a series of tools** to evaluate the cognitive faithfulness of LLMs, including:

1. **CogniBench**: Sentence-level faithfulness annotations using increasing levels of rigorousness criteria.
2. **Auto-labeling pipeline**: Utilizes LLMs as judges to assess the faithfulness of advanced LLMs and expand the CogniBench dataset into CogniBench-L.
3. **CogniDet**: A fine-tuned 8B model effective for low-cost hallucination detection in both factual and cognitive statements.


## CogniBench

CogniBench is the first knowledge-grounded dialogue dataset and framework for assessing cognitive faithfulness.

Illustrated CogniBench Example 


Dataset Structure
Here is an example dialogue:



## Auto-labeling pipeline
For example to evaluate the faithfulness of the Llama-3.1

First generate the data using https://github.com/mutonix/RefGPT


```
python auto_label.py --input_path data/Llama-3.1-70B-Instruct_processed.jsonl --method multi_run --prompt_version v2_2 --data_verson 300_turn --model_name  gpt-4-1106-preview-nlp --dialogue_model
```



## CogniDet

