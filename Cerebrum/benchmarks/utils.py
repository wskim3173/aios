import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="interpreter")
    parser.add_argument("--data_name", type=str, default="gaia-benchmark/GAIA")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output_file", type=str, default="prediction.json")
    parser.add_argument("--on_aios", action="store_true")
    parser.add_argument("--max_num", type=int, default=None)
    return parser
