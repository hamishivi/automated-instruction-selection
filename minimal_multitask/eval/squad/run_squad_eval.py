import argparse
import tqdm
import json

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
try:
    import vllm
except ImportError:
    print("VLLM not installed. Will not be able to use VLLM.")
    vllm = None
from transformers import AutoTokenizer, AutoModelForCausalLM
from minimal_multitask.eval.alpaca_eval.run_alpaca_eval import create_prompt_with_tulu_chat_format
from minimal_multitask.eval.squad.squad_eval_1 import evaluate

SQUAD_SHOTS = [
    "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\n\nTo whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?\n\nSaint Bernadette Soubirous",
    "Burke was born in Dublin, Ireland. His mother Mary née Nagle (c. 1702 – 1770) was a Roman Catholic who hailed from a déclassé County Cork family (and a cousin of Nano Nagle), whereas his father, a successful solicitor, Richard (died 1761), was a member of the Church of Ireland; it remains unclear whether this is the same Richard Burke who converted from Catholicism. The Burke dynasty descends from an Anglo-Norman knight surnamed de Burgh (latinised as de Burgo) who arrived in Ireland in 1185 following Henry II of England's 1171 invasion of Ireland.\n\nWhere was Burke born?\n\nDublin, Ireland",
    "The term high definition once described a series of television systems originating from August 1936; however, these systems were only high definition when compared to earlier systems that were based on mechanical systems with as few as 30 lines of resolution. The ongoing competition between companies and nations to create true \"HDTV\" spanned the entire 20th century, as each new system became more HD than the last.In the beginning of the 21st century, this race has continued with 4k, 5k and current 8K systems.\n\nThe term \"high definition\" originally described televisions systems from what year?\n\n1936"
]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    squad_og = load_dataset("squad", split="validation")

    # convert everything to tulu
    def convert_squad_sample(sample):
        prompt = "\n".join(SQUAD_SHOTS) + '\n' + sample["context"] + "\n\n" + sample["question"]
        label = sample["answers"]["text"][0]
        messages = [{"role": "user", "content": prompt}]
        return {"prompt": create_prompt_with_tulu_chat_format(messages, tokenizer, add_bos=False), "labels": label}

    squad = squad_og.map(convert_squad_sample, load_from_cache_file=False)
    prompts = squad["prompt"]
    print("Length of the datasets: ", len(prompts))

    # predict over dataset
    model_results = []
    if args.use_vllm:
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path
            if args.tokenizer_name_or_path is not None
            else args.model_name_or_path,
            # tokenizer_mode="slow",
            tensor_parallel_size=torch.cuda.device_count(),
        )
        sampling_params = vllm.SamplingParams(
            temperature=0,  # greedy decoding
            max_tokens=8192,
        )
        outputs = model.generate(prompts, sampling_params)
        model_results = [it.outputs[0].text for it in outputs]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.unk_token
        hf_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
        ds = Dataset.from_dict(hf_prompts)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        ds_loader = DataLoader(ds, batch_size=args.batch_size)

        model.generation_config.max_new_tokens = 100
        # model.generation_config.eos_token_id = [2, 21106, 829] # Temporariy fix to stop when generating ".</", although it's weird that the model generate this instead of eos token
        if args.decoding_algo == "greedy":
            model.generation_config.temperature = 0.0
            model.generation_config.do_sample = False
        elif args.decoding_algo == "sampling":
            model.generation_config.temperature = args.temperature
        outputs = []
        for batch in tqdm.tqdm(ds_loader):
            output = model.generate(batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda())
            generation = output[:, batch["input_ids"].shape[1]:]
            outputs.extend(generation)
        model_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    with open(args.generation_file, "w") as fout:
        for i, output in enumerate(model_results):
            # if output.endswith("</"):
            #     output = (output.strip())[:-2]
            fout.write(json.dumps({"Prompt": prompts[i], "Generation": (output.strip())}) + "\n")

    references = [{"id": sample["id"], "answers": sample["answers"]} for sample in squad_og]
    outputs = [{"id": squad_og[i]["id"], "prediction_text": output} for i, output in enumerate(model_results)]
    results = evaluate(references=references, predictions=outputs)

    print("Results on all squad:")
    print(results)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(f"Results on all squad:\n{results}")
    if args.metrics_file:
        results_dict = results
        with open(args.metrics_file, "w") as f:
            f.write(json.dumps(results_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--decoding_algo", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--metrics_file", type=str, default=None)
    parser.add_argument("--generation_file", type=str, default=None)
    args = parser.parse_args()

    if not args.save_dir:
        args.save_dir = args.model_name_or_path

    main(args)
