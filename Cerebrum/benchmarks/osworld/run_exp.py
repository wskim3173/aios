import argparse
import datetime
import json
import logging
import os
import sys

from tqdm import tqdm

from cerebrum.utils.communication import get_mcp_server_path, aios_kernel_url

logger = logging.getLogger("desktopenv.experiment")

mcp_server_path = get_mcp_server_path(aios_kernel_url)
# mcp_server_path = os.path.join(os.getcwd(), "aios/tool/mcp_server.py")

# mcp_server_path = os.path.join("/Users/km1558/Documents/projects/dongyuanjushi/OpenAGI2.0", "aios/tool/mcp_server.py")

from cerebrum.example.agents import run_cu_agent

import asyncio

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="a11y_tree",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="benchmarks/osworld/evaluation_examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="benchmarks/osworld/evaluation_examples/test_all.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    args = parser.parse_args()

    return args


async def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    scores = []
    max_steps = args.max_steps
    
    # domains = list(test_all_meta.keys())
    
    # domains = ["os"]
    # domains = ["chrome"]
    # domains = ["thunderbird"]
    # domains = ["libreoffice_impress"]
    # domains = ["vs_code"]
    # domains = ["multi_apps"]
    domains = ["vs_code","multi_apps"]

    total_num_examples = 0
    total_num_success = 0
    
    with open(os.path.join(os.path.dirname(__file__), "easy_ids.json"), "r", encoding="utf-8") as f:
        easy_ids = json.load(f)
    
    for domain in tqdm(domains, desc="Domain"):
        
        # if domain == "chrome":
        #     continue
        # if domain != "libreoffice_writer": 
        #     continue
        num_examples = len(test_all_meta[domain])
        total_num_examples += num_examples
        
        
        for i, example_id in enumerate(tqdm(test_all_meta[domain], desc="Example", leave=False)):
            # if example_id not in rerun_ids:
            #     continue
            
            # breakpoint()
            
            if i not in easy_ids[domain]:
                continue
            
            trajectory_file = os.path.join("/Users/km1558/Documents/projects/dongyuanjushi/Cerebrum/cerebrum/example/agents/cu_agent", "cache", domain, example_id, "trajectories.json")
            
            # breakpoint()
            # if os.path.exists(trajectory_file):
            #     # with open(trajectory_file, "r", encoding="utf-8") as f:
            #     #     trajectory = json.load(f)
            #     #     scores.append(trajectory["reward"])
            #     continue
            
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)
                example["domain"] = domain
                
            
            print(f"Index: {i}")    
            print(f"[Domain]: {domain}")
            print(f"[Example ID]: {example_id}")

            instruction = example["instruction"]

            print(f"[Instruction]: {instruction}")
        
        # breakpoint()
            # wandb each example config settings
            result = await run_cu_agent(
                task_config=example,
                mcp_server_path=mcp_server_path
            )
            print(result)
            
            scores.append(result["score"])
            # run.config.update(cfg_args)

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)
        
    # env.close()
    # logger.info(f"Average score: {sum(scores) / len(scores)}")
    print(f"Average score: {sum(scores) / len(scores)}, {sum(scores)}/{len(scores)}")

def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # empty all files under example_id
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result

if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    
    args.test_config_base_dir = os.path.join(os.getcwd(), args.test_config_base_dir)
    args.test_all_meta_path = os.path.join(os.getcwd(), args.test_all_meta_path)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )
    asyncio.run(test(args, test_file_list))