import json
import re
from typing import List

from datasets import load_dataset
from ..experiment_core import MetaData, run_inference, AGENT_TYPE_MAPPING_AIOS
from ..utils import get_parser


def write_output_func(result_list: List, output_file: str):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)
    # logger.log(f"Write results num: {len(result_list)}", level="info")


def process_one_func(data, meta_data: MetaData):
    agent = AGENT_TYPE_MAPPING_AIOS[meta_data.agent_type](meta_data.on_aios)
    
    # breakpoint()
    result = agent.run_gaia(**data)

    match = re.search(r'FINAL ANSWER: (.+)', result)
    if match:
        result = match.group(1)

    prediction = {
        "task_id": data["task_id"],
        "result": result,
    }
    return prediction


if __name__ == '__main__':
    main_parser = get_parser()
    main_args = main_parser.parse_args()

    agent_type = "gaia:" + main_args.agent_type
    dataset = load_dataset(main_args.data_name, "2023_all", split=main_args.split)
    
    print(dataset[:1])
    
    breakpoint()

    meta = MetaData(
        dataset=dataset,
        agent_type=agent_type,
        output_file=main_args.output_file,
        on_aios=main_args.on_aios
        # max_num=main_args.max_num,
        # aios_args=vars(global_args),
    )

    run_inference(
        meta_data=meta,
        process_one_func=process_one_func,
        write_output_func=write_output_func,
    )