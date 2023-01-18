import json
from torch.utils.data import DataLoader, Dataset
import os, re
import random
from functools import partial

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

random.seed(42)


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


def match_value(act_values, utter_list):
    result = []
    for utter in utter_list:
        match = True
        for v in act_values:
            if isinstance(v, list):
                match_v = [v1 for v1 in v if v1 in utter]
                if match_v == []:
                    match = False
            else:
                if v not in utter:
                    match = False
        if match:
            result.append(utter)
    return result

def extract_act_list(act_str):
    pattern = re.compile(r'\((.*)\)')
    list_gold = act_str.split(' , ')
    new_list = []
    for l in list_gold:
        if "(" in l and ")" in l:
            act = ' '.join(l.split()[:2])
            slots = pattern.findall(l)[0].strip().split()
            for slot in slots:
                new_list.append(act + ' ' + slot)
        else:
            new_list.append(l)
    return new_list

def match_candidate(act_str, candidates, candidate_pred):
    base_act_list = extract_act_list(act_str)
    result = []
    for c, cp in zip(candidates, candidate_pred):
        c_act_list = extract_act_list(c)
        error_num = len(set(base_act_list)-set(c_act_list)) + len(set(c_act_list)-set(base_act_list))
        result.append([cp, error_num])
    result.sort(key=lambda e:e[1], reverse=False)
    return result[0][0]


def create_gen_data(args, dialogue_idx, turn, di, domain, system_act, gen_tokenizer, dialogue_history):
    gen_act_list = []
    act_values = []
    for key , values in system_act.items():
        actinfo = ' '.join(key.split('-'))
        gen_act_list.append((actinfo + " ( " + " , ".join([name + ' ' + value for name, value in values]) + " )").lower())
        if "Request" not in key:
            for s, v in values:
                s = s.lower()
                v = gen_tokenizer.decode(gen_tokenizer.encode(v.lower()), skip_special_tokens=True)
                if s == "parking":
                    if "parking" not in act_values:
                        act_values.append("parking")
                elif s == "internet":
                    if ["internet", "wifi"] not in act_values:
                        act_values.append(["internet", "wifi"])
                else:
                    if v not in ["none", "not mentioned", "?"] and v not in act_values:
                        act_values.append(v)

    gen_act_str = " ; ".join(gen_act_list)

    gen_input_text = "[usr] " + turn["user"] + f" {gen_tokenizer.sep_token}" + " [sys] " + gen_act_str + f" {gen_tokenizer.eos_token}"
    if args["gen_history"] != 0:
        if dialogue_history != []:
            gen_input_text = ' '.join(dialogue_history[-args["gen_history"]:]) + " [usr] " + turn["user"] + f" {gen_tokenizer.sep_token}" + " [sys] " + gen_act_str + f" {gen_tokenizer.eos_token}"

    gen_output_text = turn["resp_nodelx"] + f" {gen_tokenizer.eos_token}"

    gen_data_detail = {
                    "dialogue_idx": dialogue_idx,
                    "turn_idx": turn["turn_num"],
                    "split_idx":di,
                    "turn_domain": turn["turn_domain"],
                    "split_domain":domain,
                    "system":turn["resp_nodelx"], 
                    "user":turn["user"],
                    "act_values":act_values,
                    "system_act":system_act,
                    "act_str":gen_act_str,
                    "input_text":gen_input_text,
                    "output_text":gen_output_text
                    }

    return gen_data_detail

def create_fil_data(args, dialogue_idx, turn, di, domain, system_act, fil_tokenizer, dialogue_history):
    fil_act_list = []
    for key , values in system_act.items():
        actinfo = ' '.join(key.split('-'))
        name_list = []
        for name , value in values:
            if name != 'none' and name not in name_list:
                name_list.append(name)
        if name_list != []:
            fil_act_list.append((actinfo + " ( " + " ".join([name for name in name_list]) + " )").lower())
        else:
            fil_act_list.append((actinfo).lower())

    fil_act_str = " , ".join(fil_act_list)
    fil_input_text = "[usr] " + turn["user"] + f" {fil_tokenizer.sep_token}" + " [sys] " + turn["resp_nodelx"] + f" {fil_tokenizer.eos_token}"
    if args["fil_history"] != 0:
        if dialogue_history != []:
            fil_input_text = ' '.join(dialogue_history[-args["fil_history"]:]) + " [usr] " + turn["user"] + f" {fil_tokenizer.sep_token}" + " [sys] " + turn["resp_nodelx"] + f" {fil_tokenizer.eos_token}"

    fil_output_text = fil_act_str + f" {fil_tokenizer.eos_token}"

    fil_data_detail = {
                    "dialogue_idx": dialogue_idx,
                    "turn_idx": turn["turn_num"],
                    "split_idx":di,
                    "turn_domain": turn["turn_domain"],
                    "split_domain":domain,
                    "system":turn["resp_nodelx"], 
                    "user":turn["user"],
                    "system_act":system_act,
                    "act_str":fil_act_str,
                    "input_text":fil_input_text,
                    "output_text":fil_output_text
                    }

    return fil_data_detail


def act_split(turn_domain, system_act):
    domain_act_dict = {}
    greet = False
    if 'general-greet' in system_act:
        greet = True
    last_domain = ''
    for domain in turn_domain.split():
        domain = domain.replace('[', '').replace(']', '')
        if domain == 'general':
            continue
        for act, slot_info in system_act.items():
            act_domain = act.split('-')[0]
            if act_domain == domain:
                if domain not in domain_act_dict:
                    last_domain = domain
                    if domain_act_dict == {} and greet:
                        domain_act_dict[domain] = {'general-greet':[]}
                    else:
                        domain_act_dict[domain] = {}
                domain_act_dict[domain][act] = slot_info
    for act, slot_info in system_act.items():
        act_domain = act.split('-')[0]
        if act_domain != 'general':
            continue
        if last_domain:
            if act != 'general-greet':
                domain_act_dict[last_domain][act] = slot_info
        else:
            if act_domain not in domain_act_dict:
                domain_act_dict[act_domain] = {}
            domain_act_dict[act_domain][act] = slot_info
    domain_list = []
    for k in domain_act_dict:
        domain_list.append(k)

    return domain_list, domain_act_dict


def read_data_MWOZ(args, data_path, dev_file, test_file, gen_tokenizer, fil_tokenizer):
    print(("Reading all files from {}".format(data_path)))
    gen_data_train = []
    fil_data_train = []
    gen_data_dev = []
    fil_data_dev = []
    gen_data_test = []
    fil_data_test = []
    gen_data_all = []
    fil_data_all = []

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

                dialogue_history.append("[usr] " + turn["user"] + " [sys] " + turn["resp_nodelx"])

                system_act = turn["sysact_nodelx_dict"]
                ori_domain_list, _ = act_split(turn["turn_domain"], system_act)

                if args["mode"] == 'retelling':
                    system_act = turn["retelling_sysact_nodelx_dict"]
                    if system_act == {} and len(ori_domain_list)>1:
                        system_act = turn["sysact_nodelx_dict"]

                if system_act == {}:
                    continue

                domain_list, domain_act_dict = act_split(turn["turn_domain"], system_act)

                for di, domain in enumerate(domain_list):

                    fil_data_detail = create_fil_data(args, dial_idx, turn, di, domain, domain_act_dict[domain], fil_tokenizer, dialogue_history[:-1])
                    
                    gen_data_detail = create_gen_data(args, dial_idx, turn, di, domain, domain_act_dict[domain], gen_tokenizer, dialogue_history[:-1])
                    gen_data_detail["fil_input_text"] = fil_data_detail["input_text"]
                    gen_data_detail["fil_act_str"] = fil_data_detail["act_str"]

                    fil_data_all.append(fil_data_detail)
                    gen_data_all.append(gen_data_detail)

                    if dial_idx in dev_list:
                        fil_data_dev.append(fil_data_detail)
                        gen_data_dev.append(gen_data_detail)
                    elif dial_idx in test_list:
                        fil_data_test.append(fil_data_detail)
                        gen_data_test.append(gen_data_detail)
                    else:
                        fil_data_train.append(fil_data_detail)
                        gen_data_train.append(gen_data_detail)

    return gen_data_train, fil_data_train, gen_data_dev, fil_data_dev, gen_data_test, fil_data_test, gen_data_all, fil_data_all


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


def prepare_data(args, gen_tokenizer, fil_tokenizer):
    dev_file = './data/data/multi-woz/valListFile.json'
    test_file = './data/data/multi-woz/testListFile.json'
    if args['dataset'] == "MultiWOZ_2.0":
        data_path = './data/data/MultiWOZ_2.0_processed/data_for_our.json'
    elif args['dataset'] == "MultiWOZ_2.1":
        data_path = './data/data/MultiWOZ_2.1_processed/data_for_our.json'
    else:
        print('Please select dataset in MultiWOZ_2.0, MultiWOZ_2.1')
        exit()

    gen_train, fil_train, gen_dev, fil_dev, gen_test, fil_test, gen_all, fil_all = read_data_MWOZ(args, data_path, dev_file, test_file, gen_tokenizer, fil_tokenizer)
    
    if args['set'] == "train":
        gen_data = gen_train
        fil_data = fil_train
        print('train_examples:', len(gen_train))
    elif args['set'] == "dev":
        gen_data = gen_dev
        fil_data = fil_dev
        print('dev_examples:', len(gen_dev))
    elif args['set'] == "test":
        gen_data = gen_test
        fil_data = fil_test
        print('test_examples:', len(gen_test))
    elif args['set'] == "all":
        gen_data = gen_all
        fil_data = fil_all
        print('all_examples:', len(gen_all))
    else:
        print('Please select set in train, dev, test, all')
        exit()
    
    gen_dataset = DSTDataset(gen_data, args)
    fil_dataset = DSTDataset(fil_data, args)
    gen_loader = DataLoader(gen_dataset, batch_size=args["batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=gen_tokenizer), num_workers=16)
    fil_loader = DataLoader(fil_dataset, batch_size=args["batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=fil_tokenizer), num_workers=16)

    return gen_loader, fil_loader
