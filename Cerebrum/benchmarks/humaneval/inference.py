import json
import os
import re
from typing import List
from datasets import load_dataset
from ..experiment_core import MetaData, AGENT_TYPE_MAPPING_AIOS, run_inference
from ..utils import get_parser


def parse_result(result: str):
    start_idx = result.find("<FINAL_ANSWER>")
    end_idx = result.find("</FINAL_ANSWER>")
    if start_idx != -1 and end_idx != -1:
        result = result[start_idx + len("<FINAL_ANSWER>"):end_idx]
    else:
        result = ""

    return result


def write_output_func(result_list: List, output_file: str):
    with open(output_file, "w", encoding="utf-8") as file:
        for result in result_list:
            json_line = json.dumps(result)
            file.write(json_line + "\n")
    print(f"Write results num: {len(result_list)}")


def process_one_func(data, meta_data: MetaData):
    
    agent = AGENT_TYPE_MAPPING_AIOS[meta_data.agent_type](meta_data.on_aios)
    result = agent.run_humaneval(data["prompt"])
    # breakpoint()
    result = parse_result(result)
    # breakpoint()
    
    check_program = (
        data["prompt"]
        + result
        + "\n"
        + data["test"]
        + "\n"
        + f"check({data['entry_point']})"
    )
    task_id = data["task_id"].split("/")[-1]
    
    path = os.path.join(os.path.dirname(__file__), "programs")
    with open(os.path.join(path, f"program{task_id}.py"), "w") as f:
        f.write(check_program)

    prediction = {
        "task_id": data["task_id"],
        "completion": result,
    }
    return prediction


if __name__ == '__main__':
    main_parser = get_parser()
    main_args = main_parser.parse_args()

    agent_type = "humaneval:" + main_args.agent_type
    dataset = load_dataset(main_args.data_name, split=main_args.split)

    meta = MetaData(
        dataset=dataset,
        agent_type=agent_type,
        output_file=main_args.output_file,
        on_aios=main_args.on_aios,
        max_num=main_args.max_num,
        # aios_args=vars(main_args),
    )

    run_inference(
        meta_data=meta,
        process_one_func=process_one_func,
        write_output_func=write_output_func,
    )