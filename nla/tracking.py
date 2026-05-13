from __future__ import annotations

import wandb
from typing import Dict, Any


class WandbTracker:
    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
    ):
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
        )

    def log(self, metrics: Dict[str, float], step: int | None = None):
        wandb.log(metrics, step=step)

    def save_model(self, path: str):
        artifact = wandb.Artifact("model", type="checkpoint")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self):
        wandb.finish()