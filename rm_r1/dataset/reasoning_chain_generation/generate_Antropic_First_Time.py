#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Async batch runner for Claude 3.7 Sonnet.
It:
  • loads the same subset_12_percent dataset you saved earlier
  • builds Anthropic-style requests
  • submits a Message-Batch job
  • shows a tqdm progress bar while it runs
  • streams the JSONL results to disk
"""

import asyncio, json, random
from pathlib import Path

from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio       # tqdm ≥4.66

import anthropic
from anthropic import AsyncAnthropic

# ---------------------------------------------------------------------
# keys / paths
# ---------------------------------------------------------------------
def get_anthropic_key(path="abs_path_to_txt_file_for_your_key") -> str:
    return Path(path).read_text().strip()

API_KEY       = get_anthropic_key()
META_PREFIX   = Path(
    "Absolute-Result_Dir"
)
META_PREFIX.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# build the batch requests
# ---------------------------------------------------------------------
ds_disk = "absolute_path_to_local_disk" # You may revise this part to load from huggingface
ds      = load_from_disk(ds_disk)#.select(range(200))        # 3 for a quick smoke‑test
random.seed(42)

requests = []
corresponding_ds_json = []

for idx, item in enumerate(ds):
    # Split OpenAI‑style context_messages into Anthropic fields
    system_prompt = None
    messages      = []

    for m in item["context_messages"]:
        role = m["role"]
        if role == "system":
            system_prompt = m["content"]        # Anthropic keeps system at top level
        else:
            # Anthropic messages must be "user" or "assistant"
            messages.append({"role": role, "content": m["content"]})

    req = {
        "custom_id": f"task-{idx}",
        "params": {
            "model":       "claude-3-7-sonnet-20250219",
            # "model":       "claude-3-5-sonnet-20241022",
            "max_tokens":  8192,
            "messages":    messages,
        },
    }
    if system_prompt:
        req["params"]["system"] = system_prompt

    requests.append(req)

    # save a light mapping for any downstream bookkeeping / scoring
    corresponding_ds_json.append(
        {
            "custom_id": f"task-{idx}",
            "context_messages": item["context_messages"],
            "winner": item["winner"],
        }
    )

with open(META_PREFIX / "corresponding_input_ds.jsonl", "w") as fh:
    json.dump(corresponding_ds_json, fh, indent=2)

# ---------------------------------------------------------------------
# async helper: create, poll, download
# ---------------------------------------------------------------------
client = AsyncAnthropic(api_key=API_KEY)


async def run_batch_with_progress(reqs, sleep_s: int = 10):
    # 1️⃣ create the batch
    batch = await client.beta.messages.batches.create(requests=reqs)
    total = len(reqs)

    # 2️⃣ progress bar
    pbar, prev_done = tqdm_asyncio(total=total, desc=f"Batch {batch.id[:8]}"), 0

    while batch.processing_status != "ended":
        await asyncio.sleep(sleep_s)
        batch = await client.beta.messages.batches.retrieve(batch.id)

        done_now = (
            batch.request_counts.succeeded
            + batch.request_counts.errored
            + batch.request_counts.canceled
            + batch.request_counts.expired
        )
        pbar.update(done_now - prev_done)
        prev_done = done_now
        pbar.set_postfix(status=batch.processing_status)

    pbar.close()

    # 3️⃣ stream results
    res_path = META_PREFIX / "batch_results.jsonl"
    stream   = await client.beta.messages.batches.results(batch.id)

    with open(res_path, "w") as fh:
        async for entry in stream:
            # entry is a BetaMessageBatchIndividualResponse (Pydantic v2 BaseModel)
            fh.write(entry.model_dump_json() + "\n")

    print(f"✔ Results saved to {res_path}")


# ---------------------------------------------------------------------
# go
# ---------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_batch_with_progress(requests))
