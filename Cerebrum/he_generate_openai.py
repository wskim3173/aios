import os, json, time, argparse
from datasets import load_dataset
from openai import OpenAI

def build_prompt(prompt):
    # Use the original HumanEval prompt as-is, but encourage starting from the function signature
    return (
        "You are a Python coding assistant. "
        "Write ONLY the function implementation required by the prompt. "
        "No extra text, tests, or comments.\n\n"
        f"{prompt}\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max_num", type=int, default=164)
    ap.add_argument("--output_file", default="benchmarks/humaneval/llm_eval_prediction.jsonl")
    args = ap.parse_args()

    os.makedirs("benchmarks/humaneval", exist_ok=True)

    ds = load_dataset("openai/openai_humaneval", split="test")
    client = OpenAI()

    cnt = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for ex in ds:
            if cnt >= args.max_num:
                break
            task_id = ex["task_id"]            # e.g., "HumanEval/0"
            prompt  = build_prompt(ex["prompt"])

            # Call the model
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            code = resp.choices[0].message.content.strip()

            # Remove code block markers if present
            if code.startswith("```"):
                code = code.strip("`")
                if code.startswith("python"):
                    code = code[len("python"):].lstrip()

            # Save as JSONL (HumanEval format)
            f.write(json.dumps(
                {"task_id": task_id, "completion": code},
                ensure_ascii=False
            ) + "\n")
            f.flush()
            cnt += 1
            time.sleep(0.4)  # For rate limiting (if needed)

    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
