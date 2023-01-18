import json
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
from functools import partial

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

random.seed(42)

def add_sepcial_tokens(tokenizer):
    special_tokens = ['[usr]','[sys]']
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        return item_info

    def __len__(self):
        return len(self.data)


def read_data_MWOZ(args, data_path, dev_file, test_file, tokenizer, mode=""):
    print(("Reading all files from {}".format(data_path)))
    data_train = []
    data_dev = []
    data_test = []

    test_list = [l.strip().lower().replace('.json', '') for l in open(test_file, 'r').readlines()]
    dev_list = [l.strip().lower().replace('.json', '') for l in open(dev_file, 'r').readlines()]

    # read files
    with open(data_path) as f:
        dials = json.load(f)
        for dial_idx, dial_dict in dials.items():
            # Reading data
            dial_idx = dial_idx.replace('.json', '')
            dialogue_history = []
            for ti, turn in enumerate(dial_dict["log"]):

                system_act = turn["sysact_nodelx_dict"]
                if turn["retelling_sysact_nodelx_dict"] != {} and mode == 'retelling_test':
                    system_act = turn["retelling_sysact_nodelx_dict"]
                
                act_list = []
                for key , values in system_act.items():
                    actinfo = ' '.join(key.split('-'))
                    if values:
                        act_list.append((actinfo + " ( " + " , ".join([name + ' ' + value for name, value in values]) + " )").lower())
                    else:
                        act_list.append(actinfo.lower())
                act_str = " ; ".join(act_list)

                input_text = "[usr] " + turn["user"] + f" {tokenizer.sep_token}" + " [sys] " + act_str + f" {tokenizer.eos_token}"
                if args["history"] != 0:
                    if dialogue_history != []:
                        input_text = ' '.join(dialogue_history[-args["history"]:]) + " [usr] " + turn["user"] + f" {tokenizer.sep_token}" + " [sys] " + act_str + f" {tokenizer.eos_token}"

                output_text = turn["resp_nodelx"] + f" {tokenizer.eos_token}"
                dialogue_history.append("[usr] " + turn["user"] + " [sys] " + turn["resp_nodelx"])

                if system_act == {}:
                    continue

                if turn["retelling_sysact_nodelx_dict"] == {} and mode == 'retelling_test':
                    continue

                data_detail = {
                            "dialogue_idx": dial_idx,
                            "turn_idx": turn["turn_num"],
                            "turn_domain": turn["turn_domain"],
                            "system":turn["resp_nodelx"], 
                            "user":turn["user"],
                            "system_act":system_act,
                            "act_list":act_list,
                            "act_str":act_str,
                            "input_text":input_text,
                            "output_text":output_text
                            }
                if dial_idx in dev_list:
                    data_dev.append(data_detail)
                elif dial_idx in test_list:
                    data_test.append(data_detail)
                else:
                    data_train.append(data_detail)

    return data_train, data_dev, data_test


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["input_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


def prepare_data(args, tokenizer):
    dev_file = '../data/data/multi-woz/valListFile.json'
    test_file = '../data/data/multi-woz/testListFile.json'
    if args['dataset'] == "MultiWOZ_2.0":
        data_path = '../data/data/MultiWOZ_2.0_processed/data_for_our.json'
    elif args['dataset'] == "MultiWOZ_2.1":
        data_path = '../data/data/MultiWOZ_2.1_processed/data_for_our.json'
    else:
        print('Please select dataset in MultiWOZ_2.0, MultiWOZ_2.1')
        exit()

    if args['mode'] == 'retelling_test':
        data_train, data_dev, data_test = read_data_MWOZ(args, data_path, dev_file, test_file, tokenizer, "retelling_test")
    else:
        data_train, data_dev, data_test = read_data_MWOZ(args, data_path, dev_file, test_file, tokenizer)

    print('train_examples:', len(data_train))
    print('dev_examples:', len(data_dev))
    print('test_examples:', len(data_test))

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)

    return train_loader, dev_loader, test_loader
