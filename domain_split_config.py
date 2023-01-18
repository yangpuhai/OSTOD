import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MultiWOZ_2.0", help="MultiWOZ_2.0 or MultiWOZ_2.1")
    parser.add_argument("--gen_dir", type=str, default="./T5_generator", help="Path for generator")
    parser.add_argument("--fil_dir", type=str, default="./T5_filter", help="Path for filter")
    parser.add_argument("--saving_dir", type=str, default="domain_split_result", help="Path for saving")
    parser.add_argument("--gen_history", type=int, default=0)
    parser.add_argument("--fil_history", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for test")
    parser.add_argument("--mode", type=str, default="retelling", help="normal or retelling") 
    parser.add_argument("--set", type=str, default="all", help="train or dev or test or all")
    args = parser.parse_args()
    return args
