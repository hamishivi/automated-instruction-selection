"""
Simple script to cancel and delete all jobs in the minimal multitask workspace
that haven't completed succesfully. Deleting these manually would take
waaaaay too long.
"""
from beaker import Beaker

beak = Beaker.from_env(default_workspace="ai2/minimal-multitask-finetuning")

# workspace = beak.get(workspace="ai2/minimal-multitask-finetuning")

# we will cancel/delete if the job is one of these
job_status_to_cancel = [
    "created",
    "scheduled",
    "failed",
]

# run through experiments.
for experiment in beak.workspace.iter_experiments(workspace="ai2/minimal-multitask-finetuning"):
    id, name = experiment.id, experiment.name
    # all gantry experiments have 1 step/job
    job = experiment.jobs[0]
    print(id, job.status.failed, job.status.exit_code, job.status.canceled_for)
    if job.status.exit_code is not None and job.status.exit_code > 0:
        print(f"Cancelling and deleting experiment {id}: {name}")
        beak.experiment.stop(experiment)
        beak.experiment.delete(experiment)

print("done!")
