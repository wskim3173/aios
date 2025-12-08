from typing import Any, Callable

from pydantic import BaseModel
from tqdm import tqdm

#from .agents.react import ReActAgent
from .agents.react_code import ReActAgent
from .agents.cot.cot import CoT

AGENT_TYPE_MAPPING_AIOS = {
    "swe:react": ReActAgent,
    "humaneval:react_code": ReActAgent,
    "gaia:react": ReActAgent,
    "humaneval:cot": CoT,
}


class MetaData(BaseModel):
    dataset: Any
    split: str = None
    agent_type: str
    output_file: str
    on_aios: bool = True
    max_num: int = None


def run(process_one_func, meta_data: MetaData, write_output_func=None):
    total_result = []
    if meta_data.split:
        dataset = meta_data.dataset[meta_data.split]
    else:
        dataset = meta_data.dataset

    for data in tqdm(dataset):
        if meta_data.max_num is not None:
            if meta_data.max_num > 0:
                meta_data.max_num -= 1
            else:
                print(f"Max num {meta_data.max_num} reached")
                break

        result = process_one_func(data, meta_data)
        total_result.append(result)

    if write_output_func:
        write_output_func(total_result, meta_data.output_file)
    return total_result


def run_inference(meta_data: MetaData, process_one_func: Callable, write_output_func: Callable = None):
    run(
        process_one_func=process_one_func,
        meta_data=meta_data,
        write_output_func=write_output_func,
    )