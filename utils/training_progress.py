from typing import Optional, Dict, Callable
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskID
)


class TrainingProgress:
    def __init__(self):
        self.progress = Progress(
            TextColumn("{task.description:<35}"),
            BarColumn(bar_width=25),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("\u2022"),
            TimeRemainingColumn(),
        )
        self._current_tf = None
        self._tf_task_id = None
        self._model_tasks = {}

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, *args):
        self.progress.__exit__(*args)

    def begin_tf(self, tf_label, attempt, max_attempts, num_models=3):
        self._current_tf = tf_label
        self._tf_task_id = self.progress.add_task(
            f"[bold cyan]{tf_label}[/] attempt {attempt}/{max_attempts}",
            total=num_models + 1
        )

    def begin_model(self, model_name, total=200):
        task_id = self.progress.add_task("  " + model_name, total=total)
        self._model_tasks[model_name] = task_id
        return task_id

    def update_model(self, model_name, completed):
        if model_name in self._model_tasks:
            self.progress.update(self._model_tasks[model_name], completed=completed)

    def end_model(self, model_name):
        if model_name in self._model_tasks:
            task = self.progress.tasks[self._model_tasks[model_name]]
            self.progress.update(self._model_tasks[model_name], completed=task.total)
            if self._tf_task_id is not None:
                self.progress.advance(self._tf_task_id)

    def begin_oos(self):
        if self._current_tf:
            task_id = self.progress.add_task("  OOS Validation", total=1)
            self._model_tasks["_oos"] = task_id
            return task_id

    def end_oos(self):
        if "_oos" in self._model_tasks:
            self.progress.update(self._model_tasks["_oos"], completed=1)
            if self._tf_task_id is not None:
                self.progress.advance(self._tf_task_id)

    def end_tf(self):
        if self._tf_task_id is not None:
            task = self.progress.tasks[self._tf_task_id]
            self.progress.update(self._tf_task_id, completed=task.total)

    def make_model_callback(self, model_name):
        def callback(iteration):
            self.update_model(model_name, iteration)
        return callback