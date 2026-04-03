"""
DataGym Environment Implementation.

Trains AI agents to clean and transform messy real-world data by writing
Python/pandas code. The agent's DataFrame is compared cell-by-cell against
a known clean ground truth.
"""

import io
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from uuid import uuid4

import numpy as np
import pandas as pd
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import DataAction, DataObservation, DataState
except ImportError:
    from models import DataAction, DataObservation, DataState

from .grading import grade_dataframe, describe_issues, column_info_str, target_schema_str, score_breakdown
from .tasks import get_task, get_tasks_by_difficulty, list_tasks, TASK_REGISTRY


FORBIDDEN_PATTERNS = [
    r"\bos\.\w+",
    r"\bsubprocess\b",
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\(",
    r"\bsystem\s*\(",
    r"\bshutil\b",
    r"\bpathlib\b",
    r"\brequests\b",
    r"\burllib\b",
    r"\bsocket\b",
]


class DataGymEnvironment(Environment):
    """Sandboxed data cleaning environment with cell-level grading."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = DataState()
        self._task = None
        self._current_df: pd.DataFrame = pd.DataFrame()
        self._expected_df: pd.DataFrame = pd.DataFrame()

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> DataObservation:
        if task_id and task_id in TASK_REGISTRY:
            self._task = get_task(task_id)
        else:
            difficulty = kwargs.get("difficulty", "easy")
            tasks = get_tasks_by_difficulty(difficulty)
            if not tasks:
                tasks = get_tasks_by_difficulty("easy")
            import random
            rng = random.Random(seed)
            self._task = rng.choice(tasks)

        self._current_df = self._task.setup_dirty().copy()
        self._expected_df = self._task.setup_clean().copy()

        grader = self._task.custom_grade or grade_dataframe
        initial_score = grader(self._current_df, self._expected_df)

        self._state = DataState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            max_steps=self._task.max_steps,
            current_step=0,
            best_score=initial_score,
            task_completed=False,
        )

        return DataObservation(
            done=False,
            reward=0.0,
            task_description=self._task.description,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            data_preview=self._current_df.head(20).to_string(index=True),
            data_shape=str(self._current_df.shape),
            column_info=column_info_str(self._current_df),
            issues_found=describe_issues(self._current_df),
            code_output="",
            code_error="",
            current_score=initial_score,
            step_number=0,
            max_steps=self._task.max_steps,
            target_schema=target_schema_str(self._expected_df),
            hint=self._task.hint if self._task.difficulty == "easy" else None,
        )

    def step(self, action: DataAction, timeout_s=None, **kwargs) -> DataObservation:
        self._state.step_count += 1
        self._state.current_step += 1

        code = action.code.strip()

        # Safety check
        safety_ok, safety_msg = self._check_safety(code)
        if not safety_ok:
            return self._make_observation(
                code_output="",
                code_error=f"Blocked: {safety_msg}",
                penalty=0.1,
            )

        # Execute code
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        namespace = {
            "df": self._current_df.copy(),
            "pd": pd,
            "np": np,
            "re": re,
            "json": __import__("json"),
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, {"__builtins__": self._safe_builtins()}, namespace)

            if "result" in namespace and isinstance(namespace["result"], pd.DataFrame):
                self._current_df = namespace["result"].copy()
            elif isinstance(namespace.get("df"), pd.DataFrame):
                self._current_df = namespace["df"].copy()

            code_output = stdout_capture.getvalue()[:5000]
            code_error = stderr_capture.getvalue()[:2000]

        except ImportError:
            code_output = stdout_capture.getvalue()[:2000]
            code_error = (
                "ImportError: import statements are not allowed in the sandbox. "
                "The following are already available: df (DataFrame), pd (pandas), "
                "np (numpy), re (regex), json. Use them directly without importing."
            )

        except Exception as e:
            code_output = stdout_capture.getvalue()[:2000]
            code_error = f"{type(e).__name__}: {str(e)[:500]}"

        return self._make_observation(
            code_output=code_output,
            code_error=code_error,
        )

    def _make_observation(self, code_output: str, code_error: str, penalty: float = 0.0) -> DataObservation:
        if self._task.custom_grade:
            score = self._task.custom_grade(self._current_df, self._expected_df)
        else:
            score = grade_dataframe(self._current_df, self._expected_df)

        score = max(0.0, score - penalty)
        prev_best = self._state.best_score
        self._state.best_score = max(self._state.best_score, score)

        done = (
            score >= 0.95
            or self._state.current_step >= self._state.max_steps
        )
        self._state.task_completed = score >= 0.95

        reward = score
        if done and score < 0.3:
            reward = score * 0.5

        feedback_parts = []
        if code_output:
            feedback_parts.append(code_output)
        if self._state.current_step > 0 and score < 0.95:
            breakdown = score_breakdown(self._current_df, self._expected_df)
            delta = score - prev_best
            direction = "improved" if delta > 0.001 else ("regressed" if delta < -0.001 else "unchanged")
            feedback_parts.append(
                f"--- Score: {score:.3f} ({direction} from {prev_best:.3f}) ---\n{breakdown}"
            )

        return DataObservation(
            done=done,
            reward=reward,
            task_description=self._task.description,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            data_preview=self._current_df.head(20).to_string(index=True),
            data_shape=str(self._current_df.shape),
            column_info=column_info_str(self._current_df),
            issues_found=describe_issues(self._current_df),
            code_output="\n".join(feedback_parts),
            code_error=code_error,
            current_score=self._state.best_score,
            step_number=self._state.current_step,
            max_steps=self._state.max_steps,
            target_schema=target_schema_str(self._expected_df),
            hint=self._task.hint if self._task.difficulty == "easy" else None,
        )

    @property
    def state(self) -> DataState:
        return self._state

    def _check_safety(self, code: str) -> tuple:
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                return False, f"Forbidden pattern detected: {pattern}"
        return True, ""

    def _safe_builtins(self) -> dict:
        allowed = [
            "abs", "all", "any", "bool", "dict", "enumerate", "filter",
            "float", "frozenset", "hasattr", "hash", "int", "isinstance",
            "issubclass", "iter", "len", "list", "map", "max", "min",
            "next", "print", "range", "repr", "reversed", "round",
            "set", "slice", "sorted", "str", "sum", "tuple", "type",
            "zip", "True", "False", "None",
        ]
        import builtins
        return {k: getattr(builtins, k) for k in allowed if hasattr(builtins, k)}

    def close(self):
        self._current_df = pd.DataFrame()
        self._expected_df = pd.DataFrame()
