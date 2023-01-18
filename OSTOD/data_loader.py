import json
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
from tqdm import tqdm
import random
from copy import deepcopy
from functools import partial
import spacy
from db_ops import MultiWozDB, db_tokens, book_tokens
from eval import extract_label

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

train_scale_dict = {'0.05':400, '0.1':800, '0.2':1600, '0.5':4000, '1.0':8434}

random.seed(42)

def add_sepcial_tokens(tokenizer):
    special_tokens = ['[usr]','[sys]', '[s_domain]', '[e_domain]', '[s_choice]', '[e_choice]', '[s_price]', '[e_price]',
    '[s_type]', '[e_type]', '[s_parking]', '[e_parking]', '[s_day]', '[e_day]', '[s_stay]', '[e_stay]', '[s_people]', '[e_people]', 
    '[s_reference]', '[e_reference]', '[s_name]', '[e_name]', '[s_address]', '[e_address]', '[s_phone]', '[e_phone]', 
    '[s_postcode]', '[e_postcode]', '[s_destination]', '[e_destination]', '[s_arrive]', '[e_arrive]', '[s_departure]', '[e_departure]',
    '[s_leave]', '[e_leave]', '[s_stars]', '[e_stars]', '[s_internet]', '[e_internet]', '[s_area]', '[e_area]', '[s_time]', '[e_time]',
    '[s_food]', '[e_food]', '[s_department]', '[e_department]', '[s_id]', '[e_id]', '[s_car]', '[e_car]']
    special_tokens.extend(db_tokens)
    special_tokens.extend(book_tokens)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)


def match_candidate(dbs, output_texts, pre_state, tokenizer, turn_domain):
    db = MultiWozDB(dbs)
    result = []
    for token, text in zip(list(reversed(db_tokens)), list(reversed(output_texts))):
    # for token, text in zip(db_tokens, output_texts):
        text = ' '.join(text.replace(tokenizer.sep_token,'').replace(tokenizer.eos_token,'').replace(tokenizer.pad_token,'').strip().split())
        turn_label, pred_domain = extract_label(text)
        db_state = deepcopy(pre_state)
        db_state.update(turn_label)
        # print('given_db: ', token)
        # print('turn_label: ', turn_label)
        # print('pred_text: ', text)
        db_pointer = state_to_DBpointer(db, db_state, turn_domain)
        # print('pred_db: ', db_pointer)
        flag = False
        if db_pointer == token:
            flag = True
        result.append([flag, text, turn_label])
    result.sort(key=lambda r:len(r[2]), reverse=True)
    for r in result:
        if r[0]:
            return r[1:]
    return result[0][1:]


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


def delex_by_action(text, action):
    ori_text = deepcopy(text)
    parking = False
    internet = False
    all_slot = []
    for act, slot_list in action.items():
        if 'request' in act:
            continue
        for slot, value in slot_list:
            if slot == 'none':
                continue
            if slot == 'parking':
                parking = True
                continue
            if slot == 'internet':
                internet = True
                continue
            if value == 'none':
                continue
            if [slot, value] not in all_slot:
                all_slot.append([slot, value])
    all_slot.sort(key=lambda s: len(s[1]), reverse=True)
    slot_names = [s[0] for s in all_slot]
    if 'people' in slot_names and 'stay' in slot_names:
        p_idx = slot_names.index('people')
        s_idx = slot_names.index('stay')
        if p_idx > s_idx:
            p_info = deepcopy(all_slot[p_idx])
            all_slot[p_idx] = deepcopy(all_slot[s_idx])
            all_slot[s_idx] = p_info
    not_found = []
    for slot, value in all_slot:
        if value not in text:
            not_found.append([slot, value])
        delex_v = '[s_%s] [e_%s]'%(slot, slot)
        if len(value.split())>1:
            if '.' in value and value not in text:
                value_list = [value, ' '.join(value.replace('.', ' .').split()), ' '.join(value.replace(' .', '.').split())]
                for v in value_list:
                    if v in text and not_found[-1][1] == value:
                        not_found.pop()
                    text = text.replace(v, delex_v)
            else:
                text = text.replace(value, delex_v)
        else:
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value:
                    if slot in ['people', 'stay']:
                        if idx < len(tokens)-1:
                            next_token = tokens[idx+1]
                            if slot == 'people' and next_token in ['nights', 'night', 'days', 'day']:
                                continue
                            if slot == 'stay' and next_token in ['people', 'person']:
                                continue
                            tokens[idx] = delex_v
                        else:
                            tokens[idx] = delex_v
                    else:
                        tokens[idx] = delex_v
                else:
                    if slot == 'type':
                        if value + 's' == tk or value +'es' == tk:
                            tokens[idx] = delex_v
            text = ' '.join(tokens)
    if parking:
        if 'parking' not in text:
            not_found.append(['parking', 'parking'])
        text = text.replace('parking', '[s_parking] [e_parking]')
    if internet:
        if 'internet' not in text and 'wifi' not in text and 'wi - fi' not in text:
            not_found.append(['internet', 'internet'])
        for v in ['internet', 'wifi', 'wi - fi']:
            text = text.replace(v, '[s_internet] [e_internet]')

    text = text.replace('[s_name] [e_name] [s_name] [e_name]', '[s_name] [e_name]')
    text = text.replace('[s_address] [e_address] [s_address] [e_address]', '[s_address] [e_address]')
    text = text.replace('[s_address] [e_address] , [s_address] [e_address] , [s_address] [e_address]', '[s_address] [e_address]')
    text = text.replace('[s_address] [e_address] , [s_address] [e_address]', '[s_address] [e_address]')
    text = text.replace('[s_postcode] [e_postcode] , [s_postcode] [e_postcode] , [s_postcode] [e_postcode]', '[s_postcode] [e_postcode]')
    text = text.replace('[s_postcode] [e_postcode] , [s_postcode] [e_postcode]', '[s_postcode] [e_postcode]')
    modify_tokens = {'[e_reference].':'[e_reference] .', '[e_destination].':'[e_destination] .', '[e_type]s':'[e_type] -s',
                    '12:54([s_time]':'12:54 ( [s_time]', 'm[s_price]':'[s_price]', '[e_postcode].':'[e_postcode] .',
                    '[e_leave].':'[e_leave] .', 'is[s_price]':'is [s_price]', '[e_name].':'[e_name] .', 'is[s_address]':'is [s_address]',
                    '[e_day].':'[e_day] .', 'i[s_reference]':'[s_reference]', '[e_departure].':'[e_departure] .',
                    '[e_address].':'[e_address] .', 'address-[s_address]':'address - [s_address]', '[e_price]bwp1':'[e_price]',
                    "[e_name]'phone":"[e_name] ' phone", "[e_name]'postcode":"[e_name] ' postcode", 'in[s_address]':'in [s_address]',
                    'area.[s_address]':'area . [s_address]'}
    for t in modify_tokens:
        text = text.replace(t, modify_tokens[t])

    return text


def delex_with_state(user_text, system_text_delex, state):
    correct = True
    wrong_slot = []
    tokens = system_text_delex.split()
    for slot, value in state.items():
        if value == 'dontcare':
            continue
        slot = slot.split('-')[1]
        if slot == 'pricerange':
            slot = 'price'
        if slot == 'internet':
            value = 'internet'
            if 'wifi' in user_text:
                value = 'wifi'
        if slot == 'parking':
            value = 'parking'
        slot_token = '[s_%s]'%slot
        if slot_token in tokens:
            slot_index = tokens.index(slot_token)
            tokens[slot_index] =slot_token + ' ' + value
        else:
            wrong_slot.append([slot_token, value])
            correct = False
    return ' '.join(tokens), correct


def extract_domain(action):
    act_domain = []
    for a in action:
        d = a.split("-")[0]
        if d not in act_domain:
            act_domain.append(d)
    if len(act_domain)>1 and 'general' in act_domain:
        act_domain.remove('general')
    return act_domain


def state_to_constraint_dict(state):
    constraint_dict = {}
    for k,v in state.items():
        domain, slot = k.split('-')
        if domain not in constraint_dict:
            constraint_dict[domain] = {}
        constraint_dict[domain][slot] = v
    return constraint_dict


def state_to_DBpointer(db, state, turn_domain):
    constraint_dict = state_to_constraint_dict(state)
    # print(constraint_dict)
    matnums = db.get_match_num(constraint_dict)
    match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
    match = matnums[match_dom]
    # vector = self.db.addDBPointer(match_dom, match)
    vector = db.addDBIndicator(match_dom, match)
    return vector


def load_data_from_file(args, processed_path):
    train_processed = json.load(open(processed_path+'/train.json'))
    dev_processed = json.load(open(processed_path+'/dev.json'))
    test_processed = json.load(open(processed_path+'/test.json'))
    data_train = []
    data_dev = []
    data_test = []
    train_num = len(train_processed)
    if str(args["scale"]) in train_scale_dict:
        train_num = train_scale_dict[str(args["scale"])]
    else:
        print('Please specify the training data size')
        exit()
    train_dials = random.sample(train_processed.keys(), train_num)
    train_processed = {k:v for k,v in train_processed.items() if k in train_dials}
    for dial_idx, dial_dict in train_processed.items():
        for ti, turn in enumerate(dial_dict["log"]):
            data_train.append(turn)
    for dial_idx, dial_dict in dev_processed.items():
        for ti, turn in enumerate(dial_dict["log"]):
            data_dev.append(turn)
    for dial_idx, dial_dict in test_processed.items():
        for ti, turn in enumerate(dial_dict["log"]):
            data_test.append(turn)
    return data_train, data_dev, data_test


def read_data_MWOZ(args, data_path, retelling_path, processed_path, dev_file, test_file, dbs, tokenizer):
    if os.path.exists(processed_path):
        data_train, data_dev, data_test = load_data_from_file(args, processed_path)
        return data_train, data_dev, data_test

    os.mkdir(processed_path)
    print(("Reading all files from {}".format(data_path)))
    nlp = spacy.load('en_core_web_sm')
    db = MultiWozDB(dbs)

    data_train = []
    data_dev = []
    data_test = []
    data_train_save = {}
    data_dev_save = {}
    data_test_save = {}

    test_list = [l.strip().lower().replace('.json', '') for l in open(test_file, 'r').readlines()]
    dev_list = [l.strip().lower().replace('.json', '') for l in open(dev_file, 'r').readlines()]

    retelling_data = json.load(open(retelling_path))

    # read files
    max_len = 0
    with open(data_path) as f:
        dials = json.load(f)
        for dial_idx, dial_dict in tqdm(list(dials.items())):
            # Reading data
            dial_idx = dial_idx.replace('.json', '')
            context = []
            context_delex = []

            if dial_idx in dev_list:
                data_dev_save[dial_idx] = {'goal': dial_dict['goal']}
            elif dial_idx in test_list:
                data_test_save[dial_idx] = {'goal': dial_dict['goal']}
            else:
                data_train_save[dial_idx] = {'goal': dial_dict['goal']}
            
            log = []
            
            for ti, turn in enumerate(dial_dict["log"]):
                # if(turn["domain"] not in EXPERIMENT_DOMAINS):
                #     continue
                turn_idx = str(turn["turn_num"])
                user_text = turn["user"]
                present_turn = turn["present_turn_dict"]
                activated_state = turn["activated_constraint_dict"]
                turn_domain = [d[1:-1] for d in turn['turn_domain'].split()]

                re_resp = False
                delex_correct = True
                if dial_idx in retelling_data:
                    if turn_idx in retelling_data[dial_idx]:
                        re_resp = True
                        system_text_delex = []
                        system_text = []
                        for split_idx in range(len(retelling_data[dial_idx][turn_idx])):
                            split_idx = str(split_idx)
                            split_text = retelling_data[dial_idx][turn_idx][split_idx]["pred_utter"]
                            split_text = ' '.join([token.text for token in nlp(split_text)]).strip()
                            split_act = retelling_data[dial_idx][turn_idx][split_idx]["split_act"]
                            split_text_delex = delex_by_action(split_text, split_act)
                            split_domain = retelling_data[dial_idx][turn_idx][split_idx]["split_domain"]
                            domain_present_turn = {k:v for k,v in present_turn.items() if k.split('-')[0]==split_domain}
                            split_text_delex, correct = delex_with_state(user_text, split_text_delex, domain_present_turn)
                            if not correct:
                                delex_correct = False
                            system_text.append(split_text)
                            system_text_delex.append('[s_domain] %s [e_domain] %s'%(split_domain, split_text_delex))
                        system_text = ' '.join(system_text)
                        system_text_delex = ' '.join(system_text_delex)
                
                if not re_resp:
                    system_text = turn["resp_nodelx"]
                    system_act = turn["sysact_nodelx_dict"]
                    system_text_delex = delex_by_action(system_text, system_act)
                    act_domain = extract_domain(system_act)
                    if act_domain != []:
                        text_domain = act_domain[0]
                    else:
                        if len(turn_domain) == 1:
                            text_domain = turn_domain[0]
                        else:
                            text_domain = turn_domain[1]
                    system_text_delex, correct = delex_with_state(user_text, system_text_delex, present_turn)
                    if not correct:
                        delex_correct = False
                    system_text_delex = '[s_domain] %s [e_domain] %s'%(text_domain, system_text_delex)

                pointer = [int(i) for i in turn['pointer'].split(',')]
                db_pointer = state_to_DBpointer(db, activated_state, turn_domain)

                if pointer[-2:] == [0, 1]:
                    book_pointer = '[book_success]'
                elif pointer[-2:] == [1, 0]:
                    book_pointer = '[book_fail]'
                else:
                    assert pointer[-2:] == [0, 0]
                    book_pointer = '[book_nores]'
                
                db_book_pointer = ' '.join([db_pointer, book_pointer])

                input_text = (' '.join(context) + f" {tokenizer.sep_token} " + "[usr] " + user_text + f" {tokenizer.sep_token} " + db_book_pointer + f" {tokenizer.eos_token}").strip()
                input_text_delex = (' '.join(context_delex) + f" {tokenizer.sep_token} " + "[usr] " + user_text + f" {tokenizer.sep_token} " + db_book_pointer + f" {tokenizer.eos_token}").strip()
                output_text = (system_text_delex + f" {tokenizer.eos_token}").strip()
                len_output = len(tokenizer.tokenize(output_text))
                if len_output > max_len:
                    max_len = len_output

                # context.append("[usr] " + user_text + " [sys] " + system_text)
                context.append("[usr] " + user_text + " [sys] " + turn["resp_nodelx"])
                context_delex.append("[usr] " + user_text + " [sys] " + system_text_delex)

                data_detail = {
                            "dialogue_idx": dial_idx,
                            "turn_idx": turn_idx,
                            "turn_domain": turn_domain,
                            "user": user_text,
                            "system": system_text, 
                            "system_delex": system_text_delex,
                            "pointer": pointer,
                            "db_pointer": db_pointer,
                            "book_pointer":book_pointer,
                            "present_turn": present_turn,
                            "activated_state": activated_state,
                            "input_text": input_text,
                            "input_text_delex": input_text_delex,
                            "output_text": output_text,
                            }
                
                log.append(data_detail)

                if dial_idx in dev_list:
                    data_dev.append(data_detail)
                elif dial_idx in test_list:
                    data_test.append(data_detail)
                else:
                    data_train.append(data_detail)
            
            if dial_idx in dev_list:
                data_dev_save[dial_idx]['log'] = log
            elif dial_idx in test_list:
                data_test_save[dial_idx]['log'] = log
            else:
                data_train_save[dial_idx]['log'] = log

    with open(processed_path+'/train.json', 'w') as f:
        json.dump(data_train_save, f, indent=2)
    
    with open(processed_path+'/dev.json', 'w') as f:
        json.dump(data_dev_save, f, indent=2)
    
    with open(processed_path+'/test.json', 'w') as f:
        json.dump(data_test_save, f, indent=2)
    
    data_train, data_dev, data_test = load_data_from_file(args, processed_path)
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
    processed_dir = 'processed_data'
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    dbs = {'attraction': '../data/db/attraction_db_processed.json',
        'hospital': '../data/db/hospital_db_processed.json',
        'hotel': '../data/db/hotel_db_processed.json',
        'police': '../data/db/police_db_processed.json',
        'restaurant': '../data/db/restaurant_db_processed.json',
        'taxi': '../data/db/taxi_db_processed.json',
        'train': '../data/db/train_db_processed.json',
        }
    if args['dataset'] == "MultiWOZ_2.0":
        data_path = '../data/data/MultiWOZ_2.0_processed/data_for_our.json'
        retelling_path = '../domain_split_result/MultiWOZ_2.0/all/pred_result.json'
        processed_path = processed_dir + '/MultiWOZ_2.0'
    elif args['dataset'] == "MultiWOZ_2.1":
        data_path = '../data/data/MultiWOZ_2.1_processed/data_for_our.json'
        retelling_path = '../domain_split_result/MultiWOZ_2.1/all/pred_result.json'
        processed_path = processed_dir + '/MultiWOZ_2.1'
    else:
        print('Please select dataset in MultiWOZ_2.0 and MultiWOZ_2.1')
        exit()
    
    data_train, data_dev, data_test = read_data_MWOZ(args, data_path, retelling_path, processed_path, dev_file, test_file, dbs, tokenizer)

    print('train_examples:', len(data_train))
    print('dev_examples:', len(data_dev))
    print('test_examples:', len(data_test))

    if args["mode"] == 'test_sequential':
        return data_test, dbs

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)

    return train_loader, dev_loader, test_loader
