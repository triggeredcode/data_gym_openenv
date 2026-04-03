"""
Task registry for DataGym.

Each task defines:
  - A description of what needs to be cleaned
  - A function that generates the dirty DataFrame
  - A function that generates the expected clean DataFrame
  - Optional hints (easy tasks only)
  - Max steps allowed
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd


@dataclass
class Task:
    task_id: str
    difficulty: str  # easy | medium | hard
    description: str
    hint: Optional[str]
    max_steps: int
    setup_dirty: Callable[[], pd.DataFrame]
    setup_clean: Callable[[], pd.DataFrame]
    custom_grade: Optional[Callable] = None  # fn(result_df, expected_df) -> float override


TASK_REGISTRY: Dict[str, Task] = {}


def register_task(task: Task):
    TASK_REGISTRY[task.task_id] = task
    return task


def get_task(task_id: str) -> Task:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def get_tasks_by_difficulty(difficulty: str) -> List[Task]:
    return [t for t in TASK_REGISTRY.values() if t.difficulty == difficulty]


def list_tasks() -> List[Dict]:
    return [
        {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "description": t.description,
        }
        for t in TASK_REGISTRY.values()
    ]


# Import task modules to trigger registration
from . import easy, medium, hard  # noqa: E402, F401
