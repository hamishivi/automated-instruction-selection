'''
A li'l script to compare two models: when we compare matching params,
what is most different and what is most similar?
Not at all principled right now... what are good distance metrics for 
deep learning models?
'''
import argparse
from transformers import AutoModel
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model1", type=str, help="Name of first model")
parser.add_argument("--model2", type=str, help="Name of second model")
args = parser.parse_args()

model1 = AutoModel.from_pretrained(args.model1)
model2 = AutoModel.from_pretrained(args.model2, use_auth_token='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')

# create a dict of name -> param for easy lookup
model2_dict = {n: p for n, p in model2.named_parameters()}

print(f"Parameter\t{args.model1} mean\t{args.model1} var\t{args.model2} mean\t{args.model2} var\tPairwise Dist")
for name, parameter1 in model1.named_parameters():
    if name not in model2_dict:
        print(f"{name} in {args.model1} but not {args.model2}")
    mean1 = parameter1.mean()
    var1 = parameter1.var()
    mean2 = model2_dict[name].mean()
    var2 = model2_dict[name].var()
    print(f"{name}\t{mean1}\t{var1}\t{mean2}\t{var2}\t{F.pairwise_distance(parameter1.flatten(), model2_dict[name].flatten())}")
print('----------------------------')