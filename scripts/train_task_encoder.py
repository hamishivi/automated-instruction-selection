"""
Script for training a task encoder on flan data.
The idea here is to train a sentence transformer model to model the notion of tasks rather than regular text.
"""
import json
import random
from typing import Any, Dict, List

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader
from tqdm import tqdm

random_generator = random.Random(42)

# prepare model - roberta-large with mean pool
word_embedding_model = models.Transformer("roberta-large", max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda")

# prepare data - take the flan data, reduce down to ~100 examples per task
print("creating training data")
data = [
    json.loads(x)
    for x in open(
        "/net/nfs.cirrascale/allennlp/hamishi/flan_data/flanv2-1M-v2.jsonl", "r"
    ).readlines()
]
tasks = list(set([json.loads(x["info"])["_task_name"] for x in data]))
task_data: Dict[str, List] = {task: [] for task in tasks}
for x in tqdm(data):
    task = json.loads(x["info"])["_task_name"]
    if len(task_data[task]) < 100:
        task_data[task].append(x)
# now, we create 50/50 split of same-task diff-task pairs.
# first create same-task pairs
same_task_pairs = []
for task in tqdm(tasks):
    for i in range(len(task_data[task])):
        for j in range(i + 1, len(task_data[task])):
            same_task_pairs.append(
                InputExample(texts=[task_data[task][i], task_data[task][j]], label=1.0)
            )
# now create diff-task pairs
# rather than enumerate, just randomly sample from the product of all tasks to avoid enumerating over all pairs
diff_task_pairs: List[Any] = []
while len(diff_task_pairs) < len(same_task_pairs):
    task1 = random.choice(tasks)
    task2 = random.choice(tasks)
    if task1 == task2:
        continue
    diff_task_pairs.append(
        InputExample(
            texts=[random.choice(task_data[task1]), random.choice(task_data[task2])], label=-1.0
        )
    )

# shuffle both, downsample diff task to same size as same task
random_generator.shuffle(same_task_pairs)
random_generator.shuffle(diff_task_pairs)
diff_task_pairs = diff_task_pairs[: len(same_task_pairs)]
train_dataloader = DataLoader(diff_task_pairs + same_task_pairs, shuffle=True, batch_size=16)

print("Training on {} pairs".format(len(diff_task_pairs + same_task_pairs)))

# define loss - cosine sim.
train_loss = losses.CosineSimilarityLoss(model)

# train!
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    checkpoint_save_steps=1000,
    warmup_steps=100,
    checkpoint_path="roberta-large-task-encoder",
)

model.save("roberta-large-task-encoder")
