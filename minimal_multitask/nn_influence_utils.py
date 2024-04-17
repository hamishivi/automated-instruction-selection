# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# from https://github.com/salesforce/fast-influence-functions/blob/master/influence_utils/nn_influence_utils.py

import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from torch.nn import CrossEntropyLoss
from typing import Dict, List, Union, Optional, Tuple, Iterator, Any
import faiss
from trak.projectors import BasicProjector, CudaProjector

# from LESS code
def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl
        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except Exception as e:
        print(f"Failed to use CudaProjector: {e}")
        projector = BasicProjector
        print("Using BasicProjector")
    return projector

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def convert_ids_to_string(
        tokenizer: PreTrainedTokenizer,
        ids: torch.LongTensor) -> str:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(tokens)


def get_loss_with_weight_decay(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        loss_averaging='mean') -> float:
    # loss_averaging = 'sum'
    model.train()
    if isinstance(inputs, list):
        assert len(inputs) == 1
        inputs = inputs[0]
    for k, v in inputs.items():
        inputs[k] = v.to(device)
        if len(inputs[k].shape) > 2:
            inputs[k] = inputs[k].squeeze()
            inputs[k] = inputs[k][None,]
        elif len(inputs[k].shape) == 1:
            inputs[k] = inputs[k][None,]
        assert len(inputs[k].shape) == 2, print(f"{k} input has shape {inputs[k].shape}")
    outputs = model(**inputs)
    # model averages tokens, here we are gonna sum them.
    logits, labels = outputs.logits, inputs["labels"]
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction=loss_averaging)
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    if n_gpu > 1:
        # TODO: setup multi-gpu code (required for larger models)
        assert False
    return loss

def compute_vectorised_gradients(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
) -> List[torch.FloatTensor]:
    model.zero_grad()
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    loss = model(**inputs).loss
    loss.backward()
    grads = []
    for name, param in model.named_parameters():
        if name not in params_filter:
            grads.append(param.grad.view(-1).detach())
    return torch.cat(grads, dim=0)


def compute_gradients(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        per_token_grads: bool = False,
) -> List[torch.FloatTensor]:

    if params_filter is None:
        params_filter = []

    model.zero_grad()
    loss = get_loss_with_weight_decay(
        device=device, n_gpu=n_gpu,
        model=model, inputs=inputs,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores,
        loss_averaging='mean' if not per_token_grads else 'none')
    if per_token_grads:
        grads = []
        for index_loss in range(loss.shape[0]):
            grad = torch.autograd.grad(
                outputs=loss[index_loss],
                inputs=[
                    param for name, param
                    in model.named_parameters()
                    if name not in params_filter],
                create_graph=True)
            grads.append(grad)
        return grads
    else:
        return torch.autograd.grad(
            outputs=loss,
            inputs=[
                param for name, param
                in model.named_parameters()
                if name not in params_filter],
            create_graph=True)


def compute_hessian_vector_products(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        vectors: torch.FloatTensor,
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]]
) -> List[torch.FloatTensor]:

    if params_filter is None:
        params_filter = []

    model.zero_grad()
    loss = get_loss_with_weight_decay(
        model=model, n_gpu=n_gpu,
        device=device, inputs=inputs,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        create_graph=True)

    model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        grad_outputs=vectors,
        only_inputs=True
    )

    return grad_grad_tuple


def compute_s_test(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        train_data_loaders: List[torch.utils.data.DataLoader],
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        damp: float,
        scale: float,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        vanilla_gradients: bool = False,
) -> List[torch.FloatTensor]:

    v = compute_gradients(
        model=model,
        n_gpu=n_gpu,
        device=device,
        inputs=test_inputs,
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    # dont do any hvp stuff, just return the vanilla gradients
    if vanilla_gradients:
        return v

    # Technically, it's hv^-1
    last_estimate = list(v).copy()
    cumulative_num_samples = 0
    with tqdm(total=num_samples) as pbar:
        for data_loader in train_data_loaders:
            for i, inputs in enumerate(data_loader):
                this_estimate = compute_hessian_vector_products(
                    model=model,
                    n_gpu=n_gpu,
                    device=device,
                    vectors=last_estimate,
                    inputs=inputs,
                    params_filter=params_filter,
                    weight_decay=weight_decay,
                    weight_decay_ignores=weight_decay_ignores)
                # Recursively caclulate h_estimate
                # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
                with torch.no_grad():
                    new_estimate = [
                        a + (1 - damp) * b - c / scale
                        for a, b, c in zip(v, last_estimate, this_estimate)
                    ]

                pbar.update(1)
                if verbose is True:
                    new_estimate_norm = new_estimate[0].norm().item()
                    last_estimate_norm = last_estimate[0].norm().item()
                    estimate_norm_diff = new_estimate_norm - last_estimate_norm
                    pbar.set_description(f"{new_estimate_norm:.5f} | {estimate_norm_diff:.5f}")

                cumulative_num_samples += 1
                last_estimate = new_estimate
                if num_samples is not None and i > num_samples:
                    break

    # References:
    # https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475
    # Do this for each iteration of estimation
    # Since we use one estimation, we put this at the end
    inverse_hvp = [X / scale for X in last_estimate]

    # Sanity check
    # Note that in parallel settings, we should have `num_samples`
    # whereas in sequential settings we would have `num_samples + 2`.
    # This is caused by some loose stop condition. In parallel settings,
    # We only allocate `num_samples` data to reduce communication overhead.
    # Should probably make this more consistent sometime.
    if cumulative_num_samples not in [num_samples, num_samples + 2]:
        raise ValueError(f"cumulative_num_samples={cumulative_num_samples} f"
                         f"but num_samples={num_samples}: Untested Territory")

    return inverse_hvp

# rather than take a hard argmax, here we sample from influence scores
def sample_from_influences(index_to_influence, num_samples):
    influences = np.array(list(index_to_influence.values()))
    influences = influences / influences.sum()
    sampled_indices = np.random.choice(list(index_to_influence.keys()), num_samples, p=influences)
    return sampled_indices



def compute_grad_zs(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
) -> List[List[torch.FloatTensor]]:

    if weight_decay_ignores is None:
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    grad_zs = []
    for inputs in data_loader:
        grad_z = compute_gradients(
            n_gpu=n_gpu, device=device,
            model=model, inputs=inputs,
            params_filter=params_filter,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores)
        with torch.no_grad():
            grad_zs.append([X.cpu() for X in grad_z])

    return grad_zs


def compute_influences(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        s_test_iterations: int = 1,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None,
        grad_zs: Optional[List[torch.FloatTensor]] = None,
) -> Tuple[Dict[int, float], Dict[int, Dict], List[torch.FloatTensor]]:

    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    if precomputed_s_test is not None:
        s_test = precomputed_s_test
    else:
        s_test = None
        for _ in range(s_test_iterations):
            _s_test = compute_s_test(
                n_gpu=n_gpu,
                device=device,
                model=model,
                test_inputs=test_inputs,
                train_data_loaders=[batch_train_data_loader],
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores,
                damp=s_test_damp,
                scale=s_test_scale,
                num_samples=s_test_num_samples)

            # Sum the values across runs
            if s_test is None:
                s_test = _s_test
            else:
                s_test = [
                    a + b for a, b in zip(s_test, _s_test)
                ]
        # Do the averaging
        s_test = [a / s_test_iterations for a in s_test]

    influences = {}
    train_inputs_collections = {}
    for index, train_inputs in enumerate(tqdm(instance_train_data_loader)):

        # Skip indices when a subset is specified to be included
        if (train_indices_to_include is not None) and (
                index not in train_indices_to_include):
            continue

        if grad_zs is None:
            grad_z = compute_gradients(
                n_gpu=n_gpu,
                device=device,
                model=model,
                inputs=train_inputs,
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores)
        else:
            # put on gpu from cpu for faster computations later
            grad_z = grad_zs[index]
            grad_z = [x.to(device) for x in grad_z]
        # stored_grads += [[x.detach().cpu() for x in grad_z]]

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, s_test)]

        influences[index] = sum(influence).item()
        train_inputs_collections[index] = train_inputs

    return influences, train_inputs_collections, s_test

def compute_influences_train_index(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        train_index,  # faiss index with train grads
        top_k: int = 10,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        s_test_iterations: int = 1,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None,
        grad_zs: Optional[List[torch.FloatTensor]] = None,
        low_rank_approx: Optional[bool] = False,
        normalize: Optional[bool] = False,
        projector: Optional[Any] = None,  # optional random transform projector.
        vanilla_gradients: bool = False,
) -> Tuple[Dict[int, float], Dict[int, Dict], List[torch.FloatTensor]]:

    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    if precomputed_s_test is not None:
        s_test = precomputed_s_test
    else:
        s_test = None
        for _ in range(s_test_iterations):
            _s_test = compute_s_test(
                n_gpu=n_gpu,
                device=device,
                model=model,
                test_inputs=test_inputs,
                train_data_loaders=[batch_train_data_loader],
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores,
                damp=s_test_damp,
                scale=s_test_scale,
                num_samples=s_test_num_samples,
                vanilla_gradients=vanilla_gradients,
            )

            # Sum the values across runs
            if s_test is None:
                s_test = _s_test
            else:
                s_test = [
                    a + b for a, b in zip(s_test, _s_test)
                ]
        # Do the averaging
        s_test = [a / s_test_iterations for a in s_test]

    influences = {}

    # flatten s_test
    s_test = torch.cat([g.reshape(-1) for g in s_test], axis=0)
    # if we are using a projector, project the s_test
    if projector is not None:
        s_test = projector.project(s_test.view(1, -1), model_id=0)
    # query over index - note the negative!
    # no negative here as we invert the sign. we want minimal!
    vec_to_search = s_test.cpu().numpy()
    if normalize:
        faiss.normalize_L2(vec_to_search)
    influences, topk_indices = train_index.search(vec_to_search, top_k)
    # negate here so the influence scores are right.
    return -influences, topk_indices, -s_test.cpu().numpy()

# same as before but batched over multiple test instances
def compute_influences_batched(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: torch.utils.data.DataLoader,
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        s_test_iterations: int = 1,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None,
        grad_zs: Optional[List[torch.FloatTensor]] = None,
        test_batches: int = -1,  # if > 0, we will split test inputs into batches
) -> Tuple[Dict[int, float], Dict[int, Dict], List[torch.FloatTensor]]:

    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]
    influences = [{} for _ in range(len(test_inputs))]
    train_inputs_collections = [{} for _ in range(len(test_inputs))]
    # split test inputs into batches in order
    test_inputs_batched = []
    batch_chunk = len(test_inputs) // test_batches
    for i in range(test_batches):
        current_batch = []
        counter = 0
        for x in test_inputs:
            if counter >= batch_chunk:
                break
            current_batch.append(x)
            counter += 1

        s_tests = []
        counterx = 5
        for test_input in current_batch:
            if precomputed_s_test is not None:
                s_test = precomputed_s_test
            else:
                s_test = None
                for _ in range(s_test_iterations):
                    _s_test = compute_s_test(
                        n_gpu=n_gpu,
                        device=device,
                        model=model,
                        test_inputs=[test_input],
                        train_data_loaders=[batch_train_data_loader],
                        params_filter=params_filter,
                        weight_decay=weight_decay,
                        weight_decay_ignores=weight_decay_ignores,
                        damp=s_test_damp,
                        scale=s_test_scale,
                        num_samples=s_test_num_samples)

                    # Sum the values across runs
                    if s_test is None:
                        s_test = _s_test
                    else:
                        s_test = [
                            a + b for a, b in zip(s_test, _s_test)
                        ]
                # Do the averaging
                s_test = [a / s_test_iterations for a in s_test]

            s_test = [x.detach().cpu() for x in s_test]
            s_tests.append(s_test)
            counterx -= 1
            if counterx <= 0:
                break
        
        # turn s_test into batched version - flatten and stack
        s_tests = torch.stack([torch.cat([x.reshape(-1) for x in stest], axis=0) for stest in s_tests], axis=0).cuda()
        
        for index, train_inputs in enumerate(tqdm(instance_train_data_loader)):

            # Skip indices when a subset is specified to be included
            if (train_indices_to_include is not None) and (
                    index not in train_indices_to_include):
                continue

            if grad_zs is None:
                grad_z = compute_gradients(
                    n_gpu=n_gpu,
                    device=device,
                    model=model,
                    inputs=train_inputs,
                    params_filter=params_filter,
                    weight_decay=weight_decay,
                    weight_decay_ignores=weight_decay_ignores)
            else:
                # put on gpu from cpu for faster computations later
                grad_z = grad_zs[index]
                grad_z = [x.detach().to('cpu') for x in grad_z]
            grad_z = torch.cat([x.detach().reshape(-1) for x in grad_z], dim=0)
            with torch.no_grad():
                influence_scores = - torch.matmul(s_tests, grad_z).squeeze()
            for test_index, inf in enumerate(influence_scores):
                influences[test_index][index] = inf.item()
                train_inputs_collections[test_index][index] = train_inputs
            del grad_z

    return influences, train_inputs_collections, s_test