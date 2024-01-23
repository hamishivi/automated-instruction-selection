import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from minimal_multitask.nn_influence_utils import compute_influences
from scripts.create_llama_encodings import encode_with_messages_format
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import pandas as pd
import numpy as np
import sys
import argparse
from peft import LoraConfig, TaskType, get_peft_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m')
parser.add_argument('--lora_rank', type=int, default=-1)
parser.add_argument('--saved_instances', type=str, default='')
parser.add_argument('--random_select', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--use_fast_tokenizer', action='store_true')
parser.add_argument('--use_flash_attention_2', action='store_true')
parser.add_argument('--leak_test_data', action='store_true')
parser.add_argument('--train_dataset', choices=['alpaca', 'lima'], default='alpaca')
parser.add_argument('--llama', action='store_true')
args = parser.parse_args()

kwargs = {}
if args.use_flash_attention_2:
    kwargs["use_flash_attention_2"] = True
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    **kwargs
).cuda()
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
if args.lora_rank > -1:
    modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if 'llama' in args.model_name or args.llama:
        modules = ["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=args.lora_rank, 
        lora_alpha=2, 
        lora_dropout=0,
        target_modules=modules
    )
    model = get_peft_model(model, peft_config)

# load and process train dataset
if args.train_dataset == 'alpaca':
    train_dataset = load_dataset('json', data_files='data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl')
    train_dataset = train_dataset['train']
elif args.train_dataset == 'lima':
    train_dataset = load_dataset('GAIR/lima', use_auth_token=True, split='train')
    def convert_lima(example):
        messages = [
            {'role': 'user', 'content': example['conversations'][0]},
            {'role': 'assistant', 'content': example['conversations'][1]}
        ]
        return {'messages': messages}
    train_dataset = train_dataset.map(convert_lima)
train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 1024, True, False))
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
if args.saved_instances != "":
    train_indices = json.load(open(args.saved_instances, "r"))
    train_dataset = train_dataset.select(train_indices)
if args.random_select > 0:
    train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.random_select))
    print(f"Randomly selected {args.random_select} train instances")
# train on mix of train and test data
if args.leak_test_data:
    from minimal_multitask.data import DATASETS
    test_dataset = DATASETS['squad'](tokenizer).get_all_test_prompts()
    train_dataset = train_dataset.select(range(len(test_dataset)))
    new_dataset = []
    for sample in test_dataset:
        train_dataset = train_dataset.add_item({k: v.tolist() for k, v in sample.items()})

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    args=TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,
        report_to="none",
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=10,
        num_train_epochs=2,
        bf16=True,
        tf32=True
    ),
)


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

# run evaluation over mmlu
choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=False):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text

data_dir = '/net/nfs.cirrascale/allennlp/yizhongw/open-instruct/data/eval/mmlu'

@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, add_special_tokens=True, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens,
            truncation=True, max_length=1024
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs

@torch.no_grad()
def eval_hf_model(subject, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    chat_formatting_function = create_prompt_with_tulu_chat_format
    for i in range(0, test_df.shape[0]):
        k = 0
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # messages = [{"role": "user", "content": prompt}]
        # prompt = chat_formatting_function(messages, add_bos=False)
        # if prompt[-1] in ["\n", " "]:
        #     prompt += "The answer is:"
        # else:
        #     prompt += " The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, max_length=2048, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            # messages = [{"role": "user", "content": prompt}]
            # prompt = chat_formatting_function(messages, add_bos=False)
            # if prompt[-1] in ["\n", " "]:
            #     prompt += "The answer is:"
            # else:
            #     prompt += " The answer is:"
                    
            tokenized_prompt = tokenizer(prompt, truncation=False, max_length=2048, add_special_tokens=False).input_ids
        prompts.append(prompt)

    # get the answer for all examples
    # note: here we cannot directly use convert_tokens_to_ids because the some tokenizers will automatically add space prefix.
    answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[0] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def eval_mmlu(model):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: 5]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        cors, acc, probs = eval_hf_model(subject, model, tokenizer, dev_df, test_df, 32)
            
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                "results_mmlu", "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

# eval_mmlu(model)
print("-----")
print("training")
print("-----")
trainer.train()
trainer.save_model()