import argparse
import os
import subprocess
from typing import List

from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--job",
    choices=["delete", "house_price_estimation"],
    default="house_price_estimation",
    type=str,
    required=False,
)


def execute_tasks(job: str, tasks: List[str]):
    for task in tasks:
        subprocess.run(
            [
                "dbx",
                "execute",
                f"--cluster-id={os.environ['CLUSTER_ID']}",
                f"--job={job}",
                f"--task={task}",
            ],
            cwd="package",
        )


if __name__ == "__main__":
    load_dotenv()
    project_tasks = [
        "creating_feature_tables",
        "training_model",
        "inferencing_model",
    ]

    args = parser.parse_args()
    if args.job != "delete":
        subprocess.run(["poetry", "build"], cwd="package")

    execute_tasks(
        job=args.job,
        tasks=["delete_project"] if args.job == "delete" else project_tasks,
    )
