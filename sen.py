from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HF_TOKEN")


login(token=api_key)


ds = load_dataset("DeepMostInnovations/saas-sales-conversations")

os.makedirs("data", exist_ok=True)
for split, dataset in ds.items():
    save_path = f"data/{split}.jsonl"
    dataset.to_json(save_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved {split} split ({len(dataset)} rows) to {save_path}")

print(ds["train"][0])