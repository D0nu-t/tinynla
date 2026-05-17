"""
nla/tracking.py

WandB experiment tracker with offline mode support.

Set tracking.mode = "offline" in base.yaml to prevent the Windows IPC
socket crash that occurs during long layer sweeps. Sync afterward with:
    wandb sync wandb/run-<id>
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import wandb


class WandbTracker:
    """
    Thin wrapper over wandb with configurable sync mode.

    Args:
        project:   WandB project name.
        run_name:  Display name for the run.
        config:    Full experiment config dict (logged to WandB).
        mode:      "online" | "offline" | "disabled".
                   "offline" writes locally; sync manually afterward.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        #mode: str = "online",
        mode: str = "disabled",
    ):
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            mode=mode,
        )
        # Global step counter — used to enforce monotonic step logging.
        # The bug: tracker.log(step=epoch) was called after tracker.log(step=step)
        # within the same run, stepping backward. We track globally and always
        # increment, ignoring the caller's step argument.
        self._global_step = 0

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics at the next global step.

        WandB requires monotonically increasing steps. We ignore the caller's
        step value and use our own counter, which prevents the
        'Tried to log to step N that is less than current step M' warning
        that appeared when epoch-level logs were interleaved with step-level logs.
        """
        wandb.log(metrics, step=self._global_step)
        self._global_step += 1

    def save_model(self, path: str) -> None:
        artifact = wandb.Artifact("model", type="checkpoint")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        wandb.finish()