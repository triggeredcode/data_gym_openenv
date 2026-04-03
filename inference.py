"""
Inference Script for DataGym Environment
===================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Uses OpenAI Client for all LLM calls
- Emits structured [START], [STEP], [END] stdout logs

Usage:
    export HF_TOKEN=hf_...
    python inference.py

    # Or with custom settings:
    export API_BASE_URL=http://localhost:11434/v1
    export MODEL_NAME=qwen2.5:7b
    python inference.py
"""

import asyncio
import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import DataGymEnv
from models import DataAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL") or "https://triggeredcode-data-gym.hf.space"

BENCHMARK = "data_gym"
MAX_STEPS = 5
TEMPERATURE = 0.1
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.8

SYSTEM_PROMPT = textwrap.dedent("""
    You are a pandas/Python data cleaning expert. Given a task description,
    data preview, and column info, output ONLY the Python code to clean the data.

    Rules:
    - Do NOT use import statements — pd, np, re, json are already loaded in scope
    - df is the working DataFrame — modify it in-place OR assign to `result`
    - Output ONLY code, no explanations or markdown
    - Use pandas operations (str.replace, astype, fillna, drop_duplicates, etc.)
    - Do NOT use os, subprocess, open, eval, or exec
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", "\\n")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_prompt(obs) -> str:
    parts = [f"Task: {obs.task_description}"]
    parts.append(f"\nData preview:\n{obs.data_preview}")
    parts.append(f"\nColumn info:\n{obs.column_info}")
    parts.append(f"\nTarget schema:\n{obs.target_schema}")
    if obs.issues_found:
        parts.append(f"\nDetected issues:\n{obs.issues_found}")
    if obs.hint:
        parts.append(f"\nHint: {obs.hint}")
    if obs.step_number > 0:
        parts.append(f"\nPrevious code output: {obs.code_output[:300]}")
        if obs.code_error:
            parts.append(f"Previous error: {obs.code_error[:300]}")
        parts.append(f"Current score: {obs.current_score:.3f}")
        parts.append("Fix the issues and try again.")
    parts.append("\nPython code:")
    return "\n".join(parts)


def extract_code(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in text:
        blocks = re.findall(r"```(?:python\n)?(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
    lines = [l for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines) if lines else "result = df"


async def run_task(env, client, task_id, task_difficulty):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    best_score = 0.0
    success = False

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            prompt = build_prompt(obs)
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                raw = response.choices[0].message.content or ""
                code = extract_code(raw)
            except Exception as e:
                code = f"print('LLM error: {e}')"

            result = await env.step(DataAction(code=code))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = obs.code_error if obs.code_error else None

            rewards.append(reward)
            steps_taken = step
            best_score = max(best_score, reward)

            log_step(step=step, action=code, reward=reward, done=done, error=error)

            if done:
                break

        score = best_score
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        log_step(step=steps_taken + 1, action="ERROR", reward=0.0, done=True, error=str(exc))
        score = 0.0

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


async def main() -> None:
    if not API_KEY:
        print("Error: Set HF_TOKEN or API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataGymEnv(base_url=ENV_URL)

    results = []

    async with env:
        import httpx
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{ENV_URL}/tasks")
            tasks = resp.json()["tasks"]

        for task_info in tasks:
            tid = task_info["task_id"]
            diff = task_info.get("difficulty", "?")
            result = await run_task(env, client, tid, diff)
            results.append(result)

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for r in results if r["success"])

    print(f"\n{'='*60}", flush=True)
    print(f"DataGym Results: {avg:.3f} avg, {passed}/{len(results)} passed", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
