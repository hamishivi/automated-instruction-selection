import torch
import tqdm


@torch.no_grad()
def get_next_word_predictions(
    model,
    tokenizer,
    prompts,
    candidate_token_ids=None,
    batch_size=1,
    return_token_predictions=False,
    disable_tqdm=False,
):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        # for t5, pad token is bos token
        decoder_input_ids = tokenizer.eos_token_id * torch.ones_like(
            batch_input_ids, device=model.device
        )

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(
            batch_input_ids, attention_mask, decoder_input_ids=decoder_input_ids
        ).logits[:, -1, :]
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

    assert len(predictions) == len(
        prompts
    ), "number of predictions should be equal to number of prompts"
    return predictions, probs


def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    auto_device_map=True,
    load_in_8bit=False,
    load_in_half=False,
    use_fast_tokenizer=False,
    padding_side="left",
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_in_8bit:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, device_map="auto", load_in_8bit=True
        )
    else:
        if auto_device_map:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
            if torch.cuda.is_available():
                model = model.cuda()
        if load_in_half:
            model = model.half()
    model.eval()
    return model, tokenizer
