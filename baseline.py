"""
Baseline inference script for DataGym.

Runs an LLM agent against all tasks and reports scores.
Supports OpenAI API (default) or any OpenAI-compatible endpoint.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

    # Local model
    python baseline.py --base-url http://localhost:1234/v1 --model qwen/qwen3-4b --api-key lm-studio
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from models import DataAction
from client import DataGymEnv

SYSTEM_PROMPT = """You are a pandas/Python data cleaning expert. Given a task description, data preview, and column info, output ONLY the Python code to clean the data.

Rules:
- You have access to: df (DataFrame), pd (pandas), np (numpy), re (regex), json
- Either modify df in-place OR assign to `result`
- Output ONLY code, no explanations
- Use pandas operations (str.replace, astype, fillna, drop_duplicates, etc.)
- Do NOT use os, subprocess, open, eval, or exec"""


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


async def run_baseline(env_url, llm, model, task_ids=None, max_attempts=2):
    env = DataGymEnv(base_url=env_url)
    results = []

    async with env:
        if task_ids:
            tasks = [{"task_id": t} for t in task_ids]
        else:
            import httpx
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{env_url}/tasks")
                tasks = resp.json()["tasks"]

        for task_info in tasks:
            tid = task_info["task_id"]
            diff = task_info.get("difficulty", "?")
            t0 = time.time()

            result = await env.reset(task_id=tid)
            obs = result.observation
            best = 0.0
            best_code = ""

            for attempt in range(max_attempts):
                prompt = build_prompt(obs)
                try:
                    response = llm.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=1024,
                        temperature=0.1,
                    )
                    raw = response.choices[0].message.content or ""
                    code = extract_code(raw)
                except Exception as e:
                    code = f"print('LLM error: {e}')"

                result = await env.step(DataAction(code=code))
                obs = result.observation
                score = result.reward or 0.0

                if score > best:
                    best = score
                    best_code = code[:200]

                if result.done or score >= 0.95:
                    break

            elapsed = time.time() - t0
            entry = {"task_id": tid, "difficulty": diff, "score": best,
                     "attempts": attempt + 1, "time_s": round(elapsed, 2)}
            results.append(entry)
            tag = "PASS" if best >= 0.8 else ("WARN" if best >= 0.5 else "FAIL")
            print(f"[{tag}] {tid:25s} [{diff:6s}] score={best:.3f} ({attempt+1} attempts, {elapsed:.1f}s)")

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    scores = [r["score"] for r in results]
    print(f"Overall: {sum(scores)/len(scores):.3f} avg ({len(scores)} tasks)")
    for d in ["easy", "medium", "hard"]:
        ds = [r["score"] for r in results if r["difficulty"] == d]
        if ds:
            print(f"  {d:7s}: {sum(ds)/len(ds):.3f} avg ({len(ds)} tasks)")
    passed = sum(1 for s in scores if s >= 0.8)
    print(f"  Passed (>=0.8): {passed}/{len(scores)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DataGym baseline inference")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set OPENAI_API_KEY or pass --api-key")
        sys.exit(1)

    llm = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"DataGym Baseline — model={args.model}, env={args.env_url}")
    print("-" * 60)

    results = asyncio.run(run_baseline(
        env_url=args.env_url, llm=llm, model=args.model,
        task_ids=args.tasks, max_attempts=args.max_attempts,
    ))
    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
