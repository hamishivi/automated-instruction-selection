'''
Compute influence, but compute train gradients first and save in index.
This lets us speed up influence queries for different test instances without
having to recompute train gradients.
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from minimal_multitask.nn_influence_utils import compute_gradients, compute_influences_train_index
from scripts.create_llama_encodings import encode_with_messages_format
from datasets import load_dataset
from tqdm import tqdm
import faiss
import argparse
import os
import pickle
from data import DATASETS

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--instance_to_influences', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_dataset', type=str, choices=['mmlu', 'alpacafarm'], default='mmlu')
parser.add_argument('--index_path', type=str)
args = parser.parse_args()


torch.manual_seed(args.seed)
kwargs = {"torch_dtype": torch.bfloat16}
if 'llama' in args.model_name:
    kwargs['use_flash_attention_2'] = True


model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    **kwargs,
    device_map="auto",  # use multiple gpus if you can
)
# loading sets requires_grad to False, so we need to set it back to True
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True

if args.tokenizer is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)



# load and process train dataset
train_dataset = load_dataset('json', data_files='data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl')
train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 512, True, False))
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
train_dataset = train_dataset['train']

# test dataset
def construct_test_sample(sample):
    prompt, label = sample['prompts'], sample['labels']
    # note no space between prompt and label
    # have to put max length! sometimes gpt-4 will generate a lot of text
    inputs = tokenizer(prompt + label + tokenizer.eos_token, return_tensors='pt', max_length=1024, truncation=True)
    input_len = len(tokenizer(prompt).input_ids)
    labels = inputs['input_ids'][0].clone()
    labels[:input_len] = -100
    return {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
        'labels': labels
    }

# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset].get_all_test_prompts(tokenizer)
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")


print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# construct dataloaders
batch_train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
instance_train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
eval_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

# construct an index over the gradients on the train data
# use inner product.
num_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])
grad_index = faiss.index_factory(num_params, "Flat", faiss.METRIC_INNER_PRODUCT)

# we add to the index in batches to speed things up?
grad_batch = 512
accum_grads = []
# save gradients for visualisation later.
samples = []
counter = 0
if True: # not os.path.exists(args.index_path):
    for index, train_inputs in enumerate(tqdm(instance_train_data_loader)):
        grad_z = compute_gradients(
                n_gpu=1,
                device=torch.device("cuda:0"),
                model=model,
                inputs=train_inputs,
                params_filter=params_filter,
                weight_decay=0.0,
                weight_decay_ignores=weight_decay_ignores
        )
        # flatten
        grad_z = torch.cat([g.reshape(-1) for g in grad_z], axis=0).detach()
        accum_grads.append(grad_z.detach().cpu().to(torch.float32))
        if index % grad_batch == 0:
            # add to index
            grad_index.add(torch.stack(accum_grads).numpy())
            accum_grads = []
    faiss.write_index(grad_index, args.index_path)
    # del and reload so we can use mmap (save memory!)
    del grad_index

grad_index = faiss.read_index(args.index_path, faiss.IO_FLAG_MMAP)

s_test = None
stored_grads = None
if os.path.exists(args.instance_to_influences):
    instance_to_influences = pickle.load(open(args.instance_to_influences, "rb"))
else:
    instance_to_influences = {}
for index, instance in tqdm(enumerate(eval_data_loader), total=len(eval_data_loader)):
    # load from saved file
    # if index in instance_to_influences:
    #     continue
    x = 100
    influences, topk_indices, _ = compute_influences_train_index( n_gpu=1, device=torch.device("cuda"), model=model, test_inputs=[instance], batch_train_data_loader=batch_train_data_loader, instance_train_data_loader=instance_train_data_loader, train_index=grad_index, top_k=args.top_k, params_filter=params_filter, weight_decay=0.0, weight_decay_ignores=weight_decay_ignores, s_test_damp=5e-3, s_test_scale=1e6, s_test_num_samples=x, s_test_iterations=1, precomputed_s_test=None, grad_zs=stored_grads, random_projector=projector if args.use_fjlt else None)
    # create dict?
    index_to_influence = {ind: influence for influence, ind in zip(influences[0], topk_indices[0])}
    instance_to_influences[index] = index_to_influence
    # periodically save to disk to avoid losing progress
    if index % 100 == 0:
        with open(args.instance_to_influences, "wb") as f:
            pickle.dump(instance_to_influences, f)
        print(f"Saved to {args.instance_to_influences} at step {index}")
