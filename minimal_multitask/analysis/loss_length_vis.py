from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse
import tqdm

from minimal_multitask.data import DATASETS


def compute_loss(args):
    kwargs = {}
    if args.not_use_flash_attention_2:
        kwargs["use_flash_attention_2"] = False
    else:
        kwargs["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.bfloat16, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.eval_dataset in DATASETS:
        test_dataset = DATASETS[args.eval_dataset](tokenizer).get_all_test_prompts(
            num_samples=args.num_eval_samples, seed=args.seed
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    loss_coll = {}

    with torch.no_grad():
        model.eval()
        for i in tqdm.tqdm(range(len(test_dataset))):
            output = model(test_dataset[i]["input_ids"].unsqueeze(0).cuda(), return_dict=True)
            logits = output["logits"]
            labels = test_dataset[i]["labels"].unsqueeze(0).cuda()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none")
            loss = loss[loss != 0].squeeze().tolist()
            for idx, l in enumerate(loss):
                if idx not in loss_coll:
                    loss_coll[idx] = []
                loss_coll[idx].append(l)
    idx = list(loss_coll.keys())
    value = [sum(loss_coll[i]) / len(loss_coll[i]) for i in idx]
    plt.plot(idx, value)
    plt.xlabel("Token index")
    plt.ylabel("Average loss")
    plt.savefig(f"{args.save_dir}/{args.model_name.split('/')[-1]}_loss_vs_length.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--eval_dataset", type=str, choices=DATASETS.keys(), default="alpacafarm")
    parser.add_argument("--num_eval_samples", type=int, default=805)
    parser.add_argument("--save_dir", default="results/llama_7b/visualizations")
    parser.add_argument("--not_use_flash_attention_2", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    compute_loss(args)
