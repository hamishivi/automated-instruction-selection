from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import random

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

long_sequence = "The Pythia Scaling Suite is a collection of models developed to facilitate interpretability research (see paper). It contains two sets of eight models of sizes 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B. For each size, there are two models: one trained on the Pile, and one trained on the Pile after the dataset has been globally deduplicated. All 8 model sizes are trained on the exact same data, in the exact same order. We also provide 154 intermediate checkpoints per model, hosted on Hugging Face as branches. The Pythia model suite was deliberately designed to promote scientific research on large language models, especially interpretability research. Despite not centering downstream performance as a design goal, we find the models match or exceed the performance of similar and same-sized models, such as those in the OPT and GPT-Neo suites."

tokenized_sequence = tokenizer(long_sequence, return_tensors="pt").input_ids

# we will accumulatively calculate the grad norm of:
# (a) the loss of the token on its own
# (b) the cumulative sequence loss
sequence_norms = []
token_norms = []
token_losses = []
sequence_maxes = []
token_maxes = []
per_dim_values_sequence = []
per_dim_values_token = []
dim_ranks = []
for i in range(len(tokenized_sequence[0])):
    model.zero_grad()
    input_up_to = tokenized_sequence[:, :i+1]
    outputs_all = model(input_up_to, labels=input_up_to)
    outputs_all.loss.backward()
    grads = torch.cat([p.grad.flatten().detach() for p in model.parameters()])
    print(f"Norm of grads up to token {i}: {grads.norm(p=2).item()}")
    print(f"Max of grads up to token {i}: {grads.abs().max().item()}")
    print(f"Argmax of grads up to token {i}: {grads.abs().argmax().item()}")
    # print(grads[41764803].item(), end=" ")
    # print(grads[41064803].item(), end=" ")
    sequence_maxes.append(grads.abs().max().item())
    sequence_norms.append(grads.norm(p=2).item())
    model.zero_grad()
    just_last_labels = input_up_to.clone()
    just_last_labels[:, :-1] = -100
    outputs_last = model(input_up_to, labels=just_last_labels)
    outputs_last.loss.backward()
    token_losses.append(outputs_last.loss.item())
    grads = torch.cat([p.grad.flatten().detach() for p in model.parameters()])
    print(f"Norm of grads for token {i}: {grads.norm(p=2).item()}")
    print(f"Max of grads up to token {i}: {grads.abs().max().item()}")
    print(f"Argmax of grads up to token {i}: {grads.abs().argmax().item()}")
    # print(grads[41764803].item(), end=" ")
    # print(grads[41064803].item())
    token_norms.append(grads.norm(p=2).item())
    token_maxes.append(grads.abs().max().item())
    # dim_ranks.append(grads)



fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes[0, 0].plot(sequence_norms, label="Sequence Norm")
axes[0, 0].set_title("Sequence Norm")
axes[1, 0].plot(token_norms, label="Token Norm")
axes[1, 0].set_title("Token Norm")
axes[2, 0].plot(token_losses, label="Token Loss")
axes[2, 0].set_title("Token Loss")
axes[0, 1].plot(sequence_maxes, label="Sequence Max")
axes[0, 1].set_title("Sequence Max")
axes[1, 1].plot(token_maxes, label="Token Max")
axes[1, 1].set_title("Token Max")
plt.show()
