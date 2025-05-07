import random 
from datasets import load_dataset 
import json 
random.seed(42)

def get_openai_key(dir="abs_path_to_openai_api"):
    with open(dir, 'r') as f:
        key = f.read()
    return key 
import openai
from pathlib import Path 

META_PREFIX = Path("abs_path_to")
META_PREFIX.mkdir(parents=True, exist_ok=True) 

key = get_openai_key()
openai.api_key = key 
from tqdm import tqdm

with open(META_PREFIX / "claude_results_incorrect_first_time.json", 'r') as json_file:
    ds = json.load(json_file) 


tasks = []

# Forcing OpenAI to generate correct response 

SUFFICE_A =         "\n\n For this sample case, the correct verdict is [[A]]. Please concentrate on producing a thorough, high-quality reasoning trace—fully aligned with the evaluation guidelines—showing why Chatbot A outperforms Chatbot B. "
SUFFICE_B =         "\n\n For this sample case, the correct verdict is [[B]]. Please concentrate on producing a thorough, high-quality reasoning trace—fully aligned with the evaluation guidelines—showing why Chatbot B outperforms Chatbot A. "
for item in ds:
    msg = item['context_messages'] 
    sft_response = item['sft_response'] 
    winner = item['winner']
    SUFFIX = SUFFICE_A if winner == "model_a" else SUFFICE_B
    msg[1]['content'] += f"{SUFFIX}The previous (wrong) judgement was: {sft_response}"
    task = {
        "custom_id": item['custom_id'],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "o3-2025-04-16",
            # "model": "o1-2024-12-17",
            "messages": item['context_messages'],
        },
    }
    tasks.append(task)



file_name = META_PREFIX / "SecondTime_OpenAI_Input.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')




import asyncio, json
from openai import AsyncOpenAI
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio              # tqdm ≥4.66


client = AsyncOpenAI(api_key=key)

async def run_batch_with_progress(jsonl_path: str, sleep_s: int = 10):
    # 1️⃣ upload input file
    batch_file = await client.files.create(
        file=Path(jsonl_path),
        purpose="batch",
    )

    # 2️⃣ create the batch job
    job = await client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # total requests is known (= len of your JSONL) – but we can also read it later
    total_tasks = sum(1 for _ in open(jsonl_path))

    # 3️⃣ progress bar ─────────────────────────────────────────────
    pbar = tqdm_asyncio(total=total_tasks, desc=f"Batch {job.id[:8]}")
    prev_done = 0

    while job.status not in ("completed", "failed", "cancelled"):
        await asyncio.sleep(sleep_s)
        job = await client.batches.retrieve(job.id)  # refresh

        done_now = job.request_counts.completed + job.request_counts.failed
        pbar.update(done_now - prev_done)            # advance bar
        prev_done = done_now                         # remember progress
        pbar.set_postfix(status=job.status)

    pbar.close()

    if job.status != "completed":
        raise RuntimeError(f"Batch ended with status {job.status}")

    # 4️⃣ download results
    out_bytes = (await client.files.content(job.output_file_id)).content
    res_path = META_PREFIX / "SecondTime_OpenAI_results.jsonl"
    res_path.write_bytes(out_bytes)
    print("Results saved ✔")

asyncio.run(run_batch_with_progress(file_name))