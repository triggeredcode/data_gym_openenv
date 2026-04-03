"""
Data models for the DataGym Environment.

DataGym trains AI agents to clean and transform messy real-world data.
Agents write Python/pandas code that operates on a DataFrame, and receive
cell-level feedback on correctness against a known clean ground truth.
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class DataAction(Action):
    """Python/pandas code to transform the current DataFrame.

    The code executes with these variables in scope:
      - df: the current pandas DataFrame (modify in-place or assign to `result`)
      - pd: pandas module
      - np: numpy module
      - re: regex module
    """

    code: str = Field(..., description="Python/pandas code to clean or transform the data")


class DataObservation(Observation):
    """Current state of the data after an action, plus task context."""

    task_description: str = Field(default="", description="What needs to be cleaned/fixed")
    task_id: str = Field(default="", description="Unique task identifier")
    difficulty: str = Field(default="easy", description="easy | medium | hard")

    data_preview: str = Field(default="", description="First rows of the current DataFrame as formatted table")
    data_shape: str = Field(default="", description="(rows, cols) shape string")
    column_info: str = Field(default="", description="Column names, dtypes, null counts")
    issues_found: str = Field(default="", description="Detected data quality issues")

    code_output: str = Field(default="", description="stdout from last code execution")
    code_error: str = Field(default="", description="Error message if code failed")

    current_score: float = Field(default=0.0, description="Score based on current DataFrame vs expected")
    step_number: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=10, description="Maximum steps allowed")

    target_schema: str = Field(default="", description="Expected column names and types after cleaning")
    hint: Optional[str] = Field(default=None, description="Hint for easy tasks only")


class DataState(State):
    """Internal state of a DataGym episode."""

    task_id: str = Field(default="", description="Current task identifier")
    difficulty: str = Field(default="easy", description="Task difficulty level")
    max_steps: int = Field(default=10, description="Max steps for this task")
    current_step: int = Field(default=0, description="Current step count")
    best_score: float = Field(default=0.0, description="Best score achieved so far")
    task_completed: bool = Field(default=False, description="Whether task is solved")
