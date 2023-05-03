from beaker import Beaker

small_task_list = [line.strip() for line in open("data/tasks1.txt", "r").readlines()]

beaker = Beaker.from_env(default_workspace="ai2/minimal-multitask-finetuning")


for experiment in beaker.workspace.iter_experiments("ai2/minimal-multitask-finetuning"):
    job = experiment.jobs[-1]
    dataset = beaker.job.results(job.id)
    if "evaluate" in experiment.name or not job.is_finalized:
        continue
    if dataset is None:
        print("No dataset for", experiment.name)
        continue
    # only grab tasks in our small task list
    if not any([t in experiment.name for t in small_task_list]):
        continue
    print(experiment.name, dataset.id)


# command would look like:
# python minimal_multitask/train.py  --output_dir /results  --do_eval --eval_tasks evaluate --evaluation_strategy no --bf16 --model_name /inputs --predict_with_generate --metrics_output /results/metrics.json --generation_max_length 256 --dataset dataset:/inputs  # noqa
