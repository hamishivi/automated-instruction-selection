'''
Compute influence, but compute train gradients first and save in index.
This lets us speed up influence queries for different test instances without
having to recompute train gradients.
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from minimal_multitask.nn_influence_utils import compute_influences_train_index, get_trak_projector, compute_vectorised_gradients
from scripts.create_llama_encodings import encode_with_messages_format
from datasets import load_dataset
from tqdm import tqdm
import faiss
import argparse
import os
import time
import pickle
from minimal_multitask.data import DATASETS
from trak.projectors import ProjectionType
from transformers import DataCollatorForSeq2Seq

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--instance_to_influences', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_dataset', type=str, choices=DATASETS.keys(), default='mmlu')
parser.add_argument('--index_path', type=str)
# be careful with this one! leaks test data into train set so we can sanity check the retrieval
parser.add_argument('--leak_test_data', action='store_true')
parser.add_argument('--save_index', action='store_true')
# if passed, we decompose the influence into the per-token influences.
# not sure how to accumulate this yet, so will yuck you into a debug loop too for now.
parser.add_argument('--per_test_token_influence', action='store_true')
# normalise the calculated influences
parser.add_argument('--normalise_influences', action='store_true')
# create plots for debugging
parser.add_argument('--create_plots', action='store_true')
parser.add_argument('--s_test_num_samples', type=int, default=100)
# from less- using random transform. -1 means no random transform
parser.add_argument('--random_transform', type=int, default=-1)
# how many grads to save before calling the projector.
# projection is costly, so we want to batch it.
parser.add_argument('--grad_batch', type=int, default=2)
# if set, apply some size reduction tricks to the faiss index
# Note: if set, we should make grad_batch massive to train the index on,
# and get good results.
parser.add_argument('--quantize_faiss', action='store_true')
# if set, use vanilla gradients instead of s_test
parser.add_argument('--vanilla_gradients', action='store_true')
# mark we are using a llama model.
parser.add_argument('--llama_model', action='store_true')
# train dataset
parser.add_argument('--train_dataset', type=str, default='alpaca')
args = parser.parse_args()

torch.manual_seed(args.seed)
kwargs = {"torch_dtype": torch.bfloat16}
if 'llama' in args.model_name or args.llama_model:
    kwargs['attn_implementation'] = "eager"  # flash doesnt work with second order grad.


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

# resize matrix to fit tokenizer, just in case...
model.resize_token_embeddings(len(tokenizer))

# load and process train dataset
if args.train_dataset == 'alpaca':
    train_dataset = load_dataset('json', data_files='data/camel_datasets/stanford_alpaca/stanford_alpaca_data.jsonl')['train']
    train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 512, True, False), num_proc=16)
elif args.train_dataset == 'tulu2':
    train_dataset = load_dataset('allenai/tulu-v2-sft-mixture', split='train')
    train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 2048, True, False), num_proc=16)
else:
    if os.path.exists(args.train_dataset):
        train_dataset = load_dataset('json', data_files=args.train_dataset)['train']
        train_dataset = train_dataset.map(lambda x: encode_with_messages_format(x, tokenizer, 2048, True, False), num_proc=16)
    else:
        raise ValueError(f"Invalid train dataset: {args.train_dataset}")
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


# test dataset - mostly handled in data.py
if args.eval_dataset in DATASETS:
    test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts()
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")


if args.leak_test_data:
    # shrink the training data for quicker testing
    train_dataset = train_dataset.select(range(len(test_dataset)))
    # add test data to train data
    for sample in test_dataset:
        train_dataset = train_dataset.add_item({k: v.tolist() for k, v in sample.items()})

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# helper debug function: plotting length against influence values or ranks.
def compute_length_vs_influence(topk_indices, influences, save_dir="figures", filter_nops=False):
    from matplotlib import pyplot as plt
    train_dataset_lengths = [len([tok for tok in x['labels'] if tok != -100]) for x in train_dataset]
    
    def sample_is_nop(idx):
        if train_dataset_lengths[idx] <= 1:
            return True
        if 'nooutput' in tokenizer.decode(train_dataset[idx]['input_ids']).lower():
            return True
        return False
    is_nop = [sample_is_nop(idx) for idx in range(len(train_dataset))]

    # remove -1 indices
    topk_indices = [x for x in topk_indices[0] if x >= 0]
    influences = influences[0][:len(topk_indices)]

    # plot: rank against length
    # scatter is_nop and not is_nop separately
    plt.scatter(list(range(len(topk_indices))), [train_dataset_lengths[i.item()] for i in topk_indices], c=[0 if is_nop[i.item()] else 1 for i in topk_indices], alpha=0.5)
    plt.xlabel("Influence Rank")
    plt.ylabel("Length")
    plt.savefig(os.path.join(save_dir, "rank_vs_length.png"))
    plt.clf()
    # plot: influence against length
    plt.scatter(influences, [train_dataset_lengths[i.item()] for i in topk_indices], c=[0 if is_nop[i.item()] else 1 for i in topk_indices], alpha=0.5)
    plt.xlabel("Influence Score")
    plt.ylabel("Length")
    plt.savefig(os.path.join(save_dir, "influence_vs_length.png"))
    plt.clf()

# construct dataloaders
batch_train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),)
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

num_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])

# if doing a random transform, set this up here
index_dim_size = num_params
if args.random_transform != -1:
    device = torch.device("cuda")
    trak_projector_class = get_trak_projector(device)
    projector = trak_projector_class(
        grad_dim=num_params,
        proj_dim=args.random_transform,
        seed=args.seed,
        device=device,
        proj_type=ProjectionType.rademacher,
        block_size=2,  # fixed for now
        model_id=0, # we only have one model
        max_batch_size=16, # could tune..
        dtype=torch.bfloat16,
    )
    index_dim_size = args.random_transform
else:
    projector = None


# construct an index over the gradients on the train data
# use inner product.
if args.quantize_faiss:
    # quantization/hnsw settings from DEFT (https://github.com/allenai/data-efficient-finetuning)
    encoding_dim = 512
    neighbors_per_node = 512
    index_factory_string = f"OPQ8_{encoding_dim},HNSW{neighbors_per_node},PQ8"
    grad_index = faiss.index_factory(index_dim_size, index_factory_string)
    # We cannot access the HNSW parameters directly. `index` is of type IndexPreTransform. We need to downcast
    # the actual index to do this.
    hnswpq_index = faiss.downcast_index(grad_index.index)
    hnswpq_index.hnsw.efConstruction = 200
    hnswpq_index.hnsw.efSearch = 128
else:
    grad_index = faiss.index_factory(index_dim_size, "Flat", faiss.METRIC_INNER_PRODUCT)

# we add to the index in batches to speed things up?
grad_batch = args.grad_batch
accum_grads = []
# save gradients for visualisation later.
samples = []
counter = 0
if not os.path.exists(args.index_path):
    for index, train_inputs in enumerate(tqdm(instance_train_data_loader)):
        grad_z = compute_vectorised_gradients(
            n_gpu=1,
            device=torch.device("cuda:0"),
            model=model,
            inputs=train_inputs,
            params_filter=params_filter,
            weight_decay=0.0,
            weight_decay_ignores=weight_decay_ignores
        ).to(torch.float32)
        accum_grads.append(grad_z.flatten())
        # project down.
        if index % grad_batch == 0:
            with torch.no_grad():
                accum_grads = torch.stack(accum_grads, dim=0)
                # project down.
                if args.random_transform != -1:
                    accum_grads = projector.project(accum_grads, model_id=0)
                accum_grads = accum_grads.detach().cpu().numpy()
            # add to index
            vecs_to_add = accum_grads
            if args.normalise_influences:
                faiss.normalize_L2(vecs_to_add)
            # train if not already
            if not grad_index.is_trained and args.quantize_faiss:
                grad_index.train(vecs_to_add)
            grad_index.add(vecs_to_add)
            accum_grads = []
            torch.cuda.empty_cache()
    # add remaining
    if len(accum_grads) > 0:
        with torch.no_grad():
            accum_grads = torch.stack(accum_grads, dim=0)
            # project down.
            if args.random_transform != -1:
                accum_grads = projector.project(accum_grads, model_id=0)
            accum_grads = accum_grads.detach().cpu().numpy()
        # add to index
        vecs_to_add = accum_grads
        if args.normalise_influences:
            faiss.normalize_L2(vecs_to_add)
        grad_index.add(vecs_to_add)
        accum_grads = []
    if args.save_index:
        faiss.write_index(grad_index, args.index_path)
        # del and reload so we can use mmap (save memory!)
        del grad_index

if os.path.exists(args.index_path):
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
    x = args.s_test_num_samples
    if args.per_test_token_influence:
        instance_length = instance['labels'].shape[-1]
        one_hots = torch.nn.functional.one_hot(torch.arange(instance_length), num_classes=instance_length)
        all_onehot_labels = torch.where(one_hots == 1, instance['labels'], -100)
        first_noninput_index = (instance['labels'] == -100).sum()
        # for every token, compute the influence
        all_token_influences = []
        all_topk_indices = []
        for i in tqdm(list(range(first_noninput_index, instance['labels'].shape[-1]))):
            influences, topk_indices, _ = compute_influences_train_index(
                n_gpu=1,
                device=torch.device("cuda"),
                model=model,
                test_inputs=[{'input_ids': instance['input_ids'],'attention_mask': instance['attention_mask'], 'labels': all_onehot_labels[i]}],
                batch_train_data_loader=batch_train_data_loader,
                instance_train_data_loader=instance_train_data_loader,
                train_index=grad_index,
                top_k=args.top_k,
                params_filter=params_filter,
                weight_decay=0.0,
                weight_decay_ignores=weight_decay_ignores,
                s_test_damp=5e-3,
                s_test_scale=1e6,
                s_test_num_samples=x,
                s_test_iterations=1,
                precomputed_s_test=None,
                grad_zs=stored_grads,
                normalize=args.normalise_influences,
                projector=projector,
                vanilla_gradients=args.vanilla_gradients
            )
            all_token_influences.append(influences)
            all_topk_indices.append(topk_indices)
        instance_to_influences[index] = (all_token_influences, all_topk_indices)
        # just dump this all to disk for now...
        with open(args.instance_to_influences, "wb") as f:
            print("Dumping sample...")
            pickle.dump(instance_to_influences, f)
    else:
        influences, topk_indices, _ = compute_influences_train_index(
            n_gpu=1,
            device=torch.device("cuda"),
            model=model,
            test_inputs=[instance],
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            train_index=grad_index,
            top_k=args.top_k,
            params_filter=params_filter,
            weight_decay=0.0,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=5e-3,
            s_test_scale=1e6,
            s_test_num_samples=x,
            s_test_iterations=1,
            precomputed_s_test=None,
            grad_zs=stored_grads,
            normalize=args.normalise_influences,
            projector=projector,
            vanilla_gradients=args.vanilla_gradients
        )
        # clear cache, required...
        torch.cuda.empty_cache()
        if index == 0 and args.create_plots:
            compute_length_vs_influence(topk_indices, influences, filter_nops=True)
        # create dict?
        index_to_influence = {ind: influence for influence, ind in zip(influences[0], topk_indices[0])}
        instance_to_influences[index] = index_to_influence
        # periodically save to disk to avoid losing progress
        if index % 100 == 0:
            with open(args.instance_to_influences, "wb") as f:
                pickle.dump(instance_to_influences, f)
            print(f"Saved to {args.instance_to_influences} at step {index}")

# final dump.
with open(args.instance_to_influences, "wb") as f:
    pickle.dump(instance_to_influences, f)
print(f"Saved to {args.instance_to_influences} at step {index}")
