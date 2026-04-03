"""DataGym Environment Client.

WebSocket-based client for interacting with the DataGym server.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import DataAction, DataObservation, DataState
except (ImportError, ModuleNotFoundError):
    from models import DataAction, DataObservation, DataState


class DataGymEnv(EnvClient[DataAction, DataObservation, DataState]):
    """
    Client for the DataGym Environment.

    Example (sync):
        >>> env = DataGymEnv(base_url="http://localhost:8000").sync()
        >>> with env:
        ...     result = env.reset(task_id="e1_fix_numeric_types")
        ...     result = env.step(DataAction(code="df['price'] = df['price'].astype(float)"))
        ...     print(f"Score: {result.reward}")

    Example (async):
        >>> async with DataGymEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id="e1_fix_numeric_types")
        ...     result = await env.step(DataAction(code="..."))
    """

    def _step_payload(self, action: DataAction) -> Dict[str, Any]:
        return {"code": action.code}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DataObservation]:
        obs_data = payload.get("observation", {})
        observation = DataObservation(
            done=obs_data.get("done", False),
            reward=payload.get("reward"),
            task_description=obs_data.get("task_description", ""),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            data_preview=obs_data.get("data_preview", ""),
            data_shape=obs_data.get("data_shape", ""),
            column_info=obs_data.get("column_info", ""),
            issues_found=obs_data.get("issues_found", ""),
            code_output=obs_data.get("code_output", ""),
            code_error=obs_data.get("code_error", ""),
            current_score=obs_data.get("current_score", 0.0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 10),
            target_schema=obs_data.get("target_schema", ""),
            hint=obs_data.get("hint"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DataState:
        return DataState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            max_steps=payload.get("max_steps", 10),
            current_step=payload.get("current_step", 0),
            best_score=payload.get("best_score", 0.0),
            task_completed=payload.get("task_completed", False),
        )
