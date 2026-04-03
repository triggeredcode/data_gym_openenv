"""
FastAPI application for the DataGym Environment.

Exposes the DataGymEnvironment via HTTP and WebSocket endpoints,
plus competition-specific endpoints: /tasks, /grader, /baseline.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import DataAction, DataObservation
    from .data_gym_environment import DataGymEnvironment
except (ImportError, ModuleNotFoundError):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import DataAction, DataObservation
    from server.data_gym_environment import DataGymEnvironment

app = create_app(
    DataGymEnvironment,
    DataAction,
    DataObservation,
    env_name="data_gym",
    max_concurrent_envs=4,
)


# ── Competition-specific endpoints ──────────────────────────────────────────

from fastapi import FastAPI

try:
    from .tasks import list_tasks as _list_tasks, TASK_REGISTRY
except (ImportError, ModuleNotFoundError):
    from server.tasks import list_tasks as _list_tasks, TASK_REGISTRY

if isinstance(app, FastAPI):

    @app.get("/tasks")
    async def get_tasks():
        return {
            "tasks": _list_tasks(),
            "action_schema": DataAction.model_json_schema(),
        }

    @app.get("/grader")
    async def grader_info():
        return {
            "description": "Tasks are graded 0.0-1.0 via cell-level DataFrame comparison "
                          "against a clean ground truth.",
            "scoring": {
                "column_structure": "20% — correct columns exist with right names",
                "row_count": "10% — correct number of rows",
                "cell_accuracy": "70% — percentage of individual cells matching expected values",
            },
            "details": {
                "numeric_tolerance": "1e-6 for float comparison",
                "string_matching": "whitespace-normalized, case handling varies by task",
                "safety_penalty": "-0.1 for forbidden operations (os, subprocess, eval, etc.)",
                "note": "Some tasks use strict grading (exact string match) where format normalization is the objective",
            },
        }

    @app.post("/baseline")
    async def run_baseline():
        GOLDEN_CODE = {
            "e1_fix_numeric_types": "df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(' ', '').astype(float)\ndf['quantity'] = df['quantity'].astype(str).str.replace(',', '').astype(int)\ndf['discount'] = df['discount'].astype(str).str.replace('%', '').astype(float) / 100\nresult = df",
            "e2_handle_nulls": "null_vals = ['N/A', 'null', 'None', '--', '']\ndf['email'] = df['email'].replace(null_vals, pd.NA)\ndf['age'] = df['age'].fillna(df['age'].median())\nresult = df",
            "e3_standardize_dates": "df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False).dt.strftime('%Y-%m-%d')\nresult = df",
            "e4_fix_text": "for col in ['name', 'city', 'category']:\n    df[col] = df[col].str.strip().str.title()\nresult = df",
            "e5_remove_duplicates": "df = df.drop_duplicates().reset_index(drop=True)\nresult = df",
            "m1_split_and_clean": "parts = df['location'].str.extract(r'^(.+),\\s*([A-Z]{2})\\s+(\\d{5})$')\ndf['city'] = parts[0]\ndf['state'] = parts[1]\ndf['zip_code'] = parts[2]\ndf = df.drop(columns=['location'])\ndf['salary'] = df['salary'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)\nresult = df",
            "m2_dedup_merge": "df['name'] = df['name'].str.title()\ndf = df.sort_values('score', ascending=False).drop_duplicates(subset='email', keep='first').sort_values('score', ascending=False).reset_index(drop=True)\ndf['id'] = range(1, len(df)+1)\nresult = df",
            "m3_types_and_outliers": "df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.strip().astype(float)\ndf['rating'] = pd.to_numeric(df['rating'].replace('N/A', pd.NA), errors='coerce')\ndf.loc[df['stock'] < 0, 'stock'] = 0\nmedian_stock = df.loc[df['stock'] <= 10000, 'stock'].median()\ndf.loc[df['stock'] > 10000, 'stock'] = int(median_stock)\ndf['weight_kg'] = df['weight_kg'].abs()\nresult = df",
            "m4_parse_logs": "parts = df['log_entry'].str.extract(r'^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) (\\w+) (.+)$')\ndf['timestamp'] = parts[0]\ndf['level'] = parts[1]\ndf['message'] = parts[2]\ndf = df.drop(columns=['log_entry'])\nresult = df",
            "m5_wide_to_long": "rows = []\nfor _, row in df.iterrows():\n    for subj in ['math', 'science', 'english']:\n        for q in [1, 2]:\n            rows.append({'student': row['student'], 'subject': subj, 'quarter': f'Q{q}', 'score': row[f'{subj}_q{q}']})\nresult = pd.DataFrame(rows).sort_values(['student', 'subject', 'quarter']).reset_index(drop=True)",
            "h1_full_pipeline": "df = df.drop_duplicates(subset='transaction_id', keep='first')\ndf['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False).dt.strftime('%Y-%m-%d')\nnull_vals = ['N/A', 'null', 'None']\ndf['amount'] = df['amount'].replace(null_vals, pd.NA)\ndf = df.dropna(subset=['amount'])\ndf['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '').astype(float)\ndf['category'] = df['category'].str.title()\ndf['status'] = df['status'].str.title()\nresult = df.reset_index(drop=True)",
            "h2_cross_table": "customers = df[df['_table']=='customers'][['customer_id','name','email']].copy()\ncustomers['customer_id'] = customers['customer_id'].astype(int)\ncustomers['name'] = customers['name'].str.title()\norders = df[df['_table']=='orders'][['order_id','customer_id','product','price','quantity','total']].copy()\nfor c in ['order_id','customer_id','quantity']: orders[c] = orders[c].astype(int)\nfor c in ['price','total']: orders[c] = orders[c].astype(float)\norders['total'] = orders['price'] * orders['quantity']\nresult = orders.merge(customers[['customer_id','name']], on='customer_id', how='left')[['order_id','customer_id','name','product','price','quantity','total']].reset_index(drop=True)",
            "h3_json_extract": "records = df['raw_data'].apply(json.loads)\ndf['name'] = records.apply(lambda x: x['name'])\ndf['age'] = records.apply(lambda x: x.get('age'))\ndf['age'] = pd.to_numeric(df['age'], errors='coerce')\ndf['num_skills'] = records.apply(lambda x: len(x.get('skills', [])))\ndf['primary_skill'] = records.apply(lambda x: x['skills'][0] if x.get('skills') else None)\ndf = df.drop(columns=['raw_data']).reset_index(drop=True)\nresult = df",
            "h4_timeseries_clean": "df = df.drop_duplicates(subset='timestamp', keep='first')\ndf['timestamp'] = pd.to_datetime(df['timestamp'])\ndf = df.set_index('timestamp')\nfull_range = pd.date_range('2024-01-01 00:00', periods=12, freq='h')\ndf = df.reindex(full_range)\ndf = df.interpolate(method='linear')\ndf = df.reset_index()\ndf.columns = ['timestamp', 'temperature', 'humidity']\ndf['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')\nresult = df",
            "h5_real_world": "df = df.drop_duplicates(subset='emp_id', keep='first')\ndf['name'] = df['name'].str.strip().str.title()\ndf['department'] = df['department'].str.title()\ndf['hire_date'] = pd.to_datetime(df['hire_date'], format='mixed', dayfirst=False).dt.strftime('%Y-%m-%d')\nnull_vals = ['N/A', 'null', 'None']\ndf['salary'] = df['salary'].replace(null_vals, pd.NA)\ndf['salary'] = df['salary'].astype(str).str.replace('$', '', regex=False).str.replace(',', '').replace('<NA>', pd.NA)\ndf['salary'] = pd.to_numeric(df['salary'], errors='coerce')\ndf['performance_rating'] = df['performance_rating'].replace(null_vals, pd.NA)\ndf['performance_rating'] = pd.to_numeric(df['performance_rating'], errors='coerce')\nresult = df.reset_index(drop=True)",
        }

        from .grading import grade_dataframe

        env = DataGymEnvironment()
        results = []
        for task_id, code in GOLDEN_CODE.items():
            if task_id not in TASK_REGISTRY:
                continue
            env.reset(task_id=task_id)
            obs = env.step(DataAction(code=code))
            results.append({
                "task_id": task_id,
                "score": obs.reward,
                "done": obs.done,
            })
        env.close()

        scores = [r["score"] for r in results]
        return {
            "results": results,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "tasks_evaluated": len(results),
        }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
