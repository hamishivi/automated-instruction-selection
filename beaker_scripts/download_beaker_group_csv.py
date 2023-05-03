from beaker import Beaker

beak = Beaker.from_env(default_workspace="ai2/minimal-multitask-finetuning")

results = beak.group.export_experiments("hamishivi/mmft-sweep-t0-eval")

with open("task_run.csv", "wb") as w:
    for b in results:
        w.write(b)
