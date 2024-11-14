import argparse
from beaker import Beaker
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, required=True)
parser.add_argument(
    "--workspace", type=str, default="ai2/minimal-multitask-finetuning", help="workspace to search for experiments"
)
args = parser.parse_args()

beaker = Beaker.from_env(default_workspace=args.workspace)

model_to_launch = []
taskname_to_launch = []
for exp in beaker.workspace.iter_experiments(workspace=args.workspace, match=args.prefix):
    spec = exp.jobs[0].execution.spec
    for env_var in spec.env_vars:
        if env_var.name == "BEAKER_RESULT_DATASET_ID":
            model_to_launch.append(env_var.value)
        if env_var.name == "GANTRY_TASK_NAME":
            taskname_to_launch.append(env_var.value)

# Launch eval with shell_scripts/eval/eval_beaker.sh
for model, taskname in zip(model_to_launch, taskname_to_launch):
    result = subprocess.run(["bash", "shell_scripts/eval/eval_beaker.sh", taskname, model], capture_output=True, text=True)
    print("Output:", result.stdout)
    print("Error:", result.stderr)
