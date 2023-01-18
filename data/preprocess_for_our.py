import random
import numpy as np
from copy import deepcopy
import json
import ontology
from collections import OrderedDict
from tqdm import tqdm

random.seed(42)

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

Ordered_Domain_Act = {
    'attraction':{
        'inform':['attraction-nooffer', 'attraction-inform', 'attraction-recommend', 'attraction-select'],
        'request':['attraction-request']},
    'hotel':{
        'inform':['hotel-nooffer', 'hotel-nobook', 'hotel-offerbooked', 'hotel-inform', 'hotel-recommend', 'hotel-select', 'hotel-offerbook'],
        'request':['hotel-request']},
    'restaurant':{
        'inform':['restaurant-nooffer', 'restaurant-nobook', 'restaurant-offerbooked', 'restaurant-inform', 'restaurant-recommend', 'restaurant-select', 'restaurant-offerbook'],
        'request':['restaurant-request']},
    'taxi':{
        'inform':['taxi-inform'],
        'request':['taxi-request']},
    'train':{
        'inform':['train-nooffer', 'train-offerbooked', 'train-inform', 'train-offerbook', 'train-select'],
        'request':['train-request']},
    'police':{
        'inform':['police-inform'],
        'request':['police-request']},
    'hospital':{
        'inform':['hospital-inform'],
        'request':['hospital-request']}
}


slot_to_act_dict = { 'pricerange': 'price'}

Choices = ['several', 'many', 'a great deal', 'plenty', 'a few', 'lots', 'some', 'quite a few', 'a lot', 'a lot of', 'large number of',
 'lots of', 'multiple', 'a number of', 'plenty', 'tons', 'a list of', 'plenty of', 'a great deal of', 'numerous', 
 'dozens', 'quite a lot', 'a variety of', 'quite a number', 'quite a lot of', 'a great number', 'a large amount']


def info_present_turn(activated_state, unactivated_state, pre_state, state, sys_act):

    turn_label = {}
    for slot, value in state.items():
        if value == 'none':
            continue
        if slot not in pre_state:
            turn_label[slot] = value
        else:
            if pre_state[slot] != value:
                turn_label[slot] = value

    for s in unactivated_state:
        if s not in turn_label:
            turn_label[s] = unactivated_state[s]

    sys_act_domain = []
    for a in sys_act:
        d = a.split("-")[0]
        if d in ontology.all_domains and d not in sys_act_domain:
            sys_act_domain.append(d)

    activated_label = {}
    unactivated_label = {}
    if sys_act_domain != []:
        for s in turn_label:
            d = s.split("-")[0]
            if d in sys_act_domain:
                activated_label[s] = turn_label[s]
            else:
                unactivated_label[s] = turn_label[s]
    else:
        unactivated_label = deepcopy(turn_label)
        
    for s in activated_label:
        v = activated_label[s]
        if s not in activated_state:
            activated_state[s] = v
        else:
            if activated_state[s] != v:
                activated_state[s] = v
    
    unactivated_state = unactivated_label
    
    present_turn = deepcopy(activated_label)
    
    return sys_act_domain, activated_state, unactivated_state, present_turn



def trans_slot(s, re_acts_dict):
    s = s.split('-',1)[1]
    if s in re_acts_dict:
        s = re_acts_dict[s]
    return s


def create_act_turn(system_act_domains, sys_act, present_turn, db_num, db_sample):
    for k in list(present_turn.keys()):
        if k in ['hotel-internet', 'hotel-parking'] and present_turn[k] != 'dontcare':
            present_turn[k] = 'none'

    new_dialog_act = deepcopy(sys_act)

    act_slot_info = {}
    for act, slot_info in sys_act.items():
        if act not in act_slot_info:
            act_slot_info[act] = []
        for slot, value in slot_info:
            if slot not in act_slot_info[act]:
                act_slot_info[act].append(slot)

    retelling = 'NO'
    re_acts_dict = slot_to_act_dict
    for domain in system_act_domains:
        slot_info = {k:v for k,v in present_turn.items() if domain in k and v != 'dontcare'}
        insert = False
        if slot_info == {}:
            continue

        for act in Ordered_Domain_Act[domain]['inform']:
            if act in act_slot_info:
                slot_info1 = slot_info
                slot_info2 = {}
                if 'nobook' in act:
                    slot_info1 = {k:v for k,v in slot_info.items() if k.split('-')[1] in ['name', 'day', 'people', 'stay', 'time']}
                    slot_info2 = {k:v for k,v in slot_info.items() if k not in slot_info1}
                    
                for s,v in slot_info1.items():
                    s = trans_slot(s, re_acts_dict)
                    if s not in act_slot_info[act]:
                        new_dialog_act[act].append([s, v])
                        retelling = 'YES'
                
                if slot_info2 != {}:
                    inform_act = domain + '-inform'
                    if inform_act not in new_dialog_act:
                        new_dialog_act[inform_act] = []
                        retelling = 'YES'
                        act_count = []
                        for s,v in slot_info2.items():
                            s = trans_slot(s, re_acts_dict)
                            act_count.append([s, v])
                        temp_act = {}
                        for act in new_dialog_act:
                            temp_act[act] = new_dialog_act[act]
                            if act == domain + '-nobook':
                                temp_act[inform_act] = act_count
                        new_dialog_act = temp_act
                    else:
                        for s,v in slot_info2.items():
                            s = trans_slot(s, re_acts_dict)
                            if s not in act_slot_info[inform_act]:
                                new_dialog_act[inform_act].append([s, v])
                                retelling = 'YES'

                insert = True
                break

        if insert == False:
            new_act = domain + '-inform'
            retelling = 'YES'

            act_count = []
            for s,v in slot_info.items():
                s = trans_slot(s, re_acts_dict)
                act_count.append([s, v])
            
            if domain in ['hotel', 'restaurant'] and domain + '-request' in new_dialog_act and {k:v for k,v in slot_info.items() if k.split('-')[1] not in ['name', 'day', 'people', 'stay', 'time']}=={}:
                new_act = domain + '-offerbook'
            else:
                if domain + '-name' not in slot_info and domain + '-request' in new_dialog_act:
                    choice_num = random.sample(Choices, 1)[0]
                    if db_sample and domain == db_sample[0]:
                        db_num = int(db_num)
                        if domain in ["hotel", "restaurant", "attraction"]:
                            if db_num <= 10:
                                choice_num = db_num
                        if domain in ["train"]:
                            if db_num <= 3:
                                choice_num = db_num
                    act_count.append(['choice', str(choice_num)])
            
            temp_act = {}
            if 'general-greet' in new_dialog_act:
                new_dialog_act.pop('general-greet')
                temp_act['general-greet'] = []
            temp_act[new_act] = act_count
            for act in new_dialog_act:
                temp_act[act] = new_dialog_act[act]
            new_dialog_act = temp_act

        # slot_dc_info = {k:v for k,v in present_turn.items() if domain in k and v == 'dontcare'}
        # if slot_dc_info != {}:
        #     inform_act = domain + '-inform'
        #     if inform_act not in new_dialog_act:
        #         new_dialog_act[inform_act] = []
        #         retelling = 'YES'
        #     for s,v in slot_dc_info.items():
        #         s = trans_slot(s, re_acts_dict)
        #         new_dialog_act[inform_act].append([s, v])
    
    return new_dialog_act, retelling


def bspan_to_state_dict(bspan):
    bspan = bspan.split() if isinstance(bspan, str) else bspan
    # state_dict = OrderedDict()
    state_dict = {}
    domain = None
    conslen = len(bspan)
    for idx, cons in enumerate(bspan):
        if '[' in cons:
            if cons[1:-1] not in ontology.all_domains:
                continue
            domain = cons[1:-1]
        elif '{' in cons and cons[1:-1] in ontology.get_slot:
            if domain is None:
                continue
            slot = cons[1:-1]
            vidx = idx+1
            if vidx == conslen:
                break
            vt_collect = []
            vt = bspan[vidx]
            while vidx < conslen and '[' not in vt and '{' not in vt:
                vt_collect.append(vt)
                vidx += 1
                if vidx == conslen:
                    break
                vt = bspan[vidx]
            if vt_collect:
                state_dict[domain+'-'+slot] = ' '.join(vt_collect)
    
    return state_dict


def state_dict_to_bspan(state_dict):
    bspan = []
    # info_dict = OrderedDict()
    info_dict = {}
    for slot_n, value in state_dict.items():
        domain, slot = slot_n.split('-')
        if domain not in info_dict:
            info_dict[domain] = []
        info_dict[domain].append([slot, value])
    for domain, slot_info in info_dict.items():
        bspan.append('['+domain+']')
        for slot, value in slot_info:
            bspan.append('{'+slot+'}')
            bspan.extend(value.split())
    return ' '.join(bspan)


def aspan_to_act_dict(aspan):
    aspan = aspan.split() if isinstance(aspan, str) else aspan
    # act_dict = OrderedDict()
    act_dict = {}
    domain = None
    act = None
    conslen = len(aspan)
    for idx, cons in enumerate(aspan):
        if '[' in cons and cons[1:-1] in ontology.dialog_acts:
            domain = cons[1:-1]

        elif '[' in cons and cons[1:-1] in ontology.dialog_act_params:
            if domain is None:
                continue
            act = cons[1:-1]
            if not act_dict.get(domain+'-'+act):
                act_dict[domain+'-'+act] = []
        
        elif '{' in cons:
            if domain is None or act is None:
                continue
            vidx = idx+1
            vt = aspan[vidx]
            vt_collect = []
            while vidx < conslen and '[' not in vt and '{' not in vt:
                vt_collect.append(vt)
                vidx += 1
                if vidx == conslen:
                    break
                vt = aspan[vidx]
            if vt_collect:
                act_dict[domain+'-'+act].append([cons[1:-1], ' '.join(vt_collect)])

    return act_dict


def act_dict_to_aspan(act_dict):
    aspan = []
    # info_dict = OrderedDict()
    info_dict = {}
    greet_act = ''
    for act_info, slot_info in act_dict.items():
        if act_info == 'general-greet':
            greet_act = 'general-greet'
            continue
        domain, act = act_info.split('-')
        if domain not in info_dict:
            # info_dict[domain] = OrderedDict()
            info_dict[domain] = {}
        info_dict[domain][act] = slot_info
    if greet_act:
        aspan.append('[general]')
        aspan.append('[greet]')
    for domain, act_slot in info_dict.items():
        aspan.append('['+domain+']')
        for act, slot_info2 in act_slot.items():
            aspan.append('['+act+']')
            for slot, value in slot_info2:
                aspan.append('{'+slot+'}')
                aspan.extend(value.split())
    return ' '.join(aspan)


def act_modify(act_input, state_dict, user_act_dict, act_history, db_sample):
    new_act_dict = deepcopy(act_input)
    name_slot = ''
    name_value = ''
    for act_info, slot_info in new_act_dict.items():
        domain, act = act_info.split('-')
        if act == 'request' or domain == 'general':
            continue
        if domain in ["hotel", "restaurant", "attraction"]:
            act_slots = [slot[0] for slot in slot_info]
            if set(act_slots) & set(["postcode","address","phone","reference"]) and "name" not in act_slots:
                if domain+'-name' in state_dict:
                    new_act_dict[act_info].append(['name', state_dict[domain+'-name']])
                    continue
                flag = False
                all_history_act = [user_act_dict]
                for sa in list(reversed(act_history)):
                    all_history_act.extend(sa)
                for h_act_dict in all_history_act:
                    for h_act_info, h_slot_info in h_act_dict.items():
                        h_domain, h_act = h_act_info.split('-')
                        if h_act == 'request' or h_domain != domain:
                            continue
                        h_act_slots = [slot1[0] for slot1 in h_slot_info]
                        if "name" in h_act_slots and h_act_slots.count("name") == 1:
                            h_value = h_slot_info[h_act_slots.index("name")][1]
                            if h_value != "none":
                                new_act_dict[act_info].append(['name', h_value])
                                name_slot = domain+'-name'
                                name_value = h_value
                                flag = True
                        if flag:
                            break
                    if flag:
                        break
                if not flag:
                    if db_sample and db_sample[0] == domain and "reference" in act_slots:
                        db_value = db_sample[1]
                        if db_value != "none":
                            new_act_dict[act_info].append(['name', db_value])
                            name_slot = domain+'-name'
                            name_value = db_value

        if domain in ["train"]:
            act_slots = [slot[0] for slot in slot_info]
            if "reference" in act_slots and "id" not in act_slots:
                flag = False
                all_history_act = [user_act_dict]
                for sa in list(reversed(act_history)):
                    all_history_act.extend(sa)
                for h_act_dict in all_history_act:
                    for h_act_info, h_slot_info in h_act_dict.items():
                        h_domain, h_act = h_act_info.split('-')
                        if h_act == 'request' or h_domain != domain:
                            continue
                        h_act_slots = [slot1[0] for slot1 in h_slot_info]
                        if "id" in h_act_slots and h_act_slots.count("id") == 1:
                            h_value = h_slot_info[h_act_slots.index("id")][1]
                            if h_value != "none":
                                new_act_dict[act_info].append(['id', h_value])
                                flag = True
                        if flag:
                            break
                    if flag:
                        break
                if not flag:
                    if db_sample and db_sample[0] == domain:
                        db_value = db_sample[1]
                        if db_value != "none":
                            new_act_dict[act_info].append(['id', db_value])

    return new_act_dict, name_slot, name_value


def preprocess_mwoz(data_path, target_path):
    data = json.load(open(data_path))
    for dial_idx, dial in tqdm(data.items()):
        activated_state_dict = {}
        unactivated_state_dict = {}
        pre_state_dict = {}
        act_history = []
        slot_modified_dict = {}
        for turn_idx, turn in enumerate(dial['log']):
            state_dict = bspan_to_state_dict(turn['constraint'])
            act_dict = aspan_to_act_dict(turn['sysact_nodelx'])
            user_act_dict = aspan_to_act_dict(turn['useract_nodelx'])
            db_num = turn["match"]
            db_sample = turn["match_sample"]

            system_act_domains, activated_state_dict, unactivated_state_dict, present_turn = info_present_turn(activated_state_dict, unactivated_state_dict, pre_state_dict, state_dict, act_dict)
            new_act_dict, retelling_turn = create_act_turn(system_act_domains, act_dict, deepcopy(present_turn), db_num, db_sample)
            act_input = new_act_dict if retelling_turn == 'YES' else act_dict
            modified_sysact_nodelx_dict, name_slot, name_value = act_modify(act_input, state_dict, user_act_dict, act_history, db_sample)
            
            retelling_sysact_nodelx_dict = {}
            if modified_sysact_nodelx_dict != act_input or retelling_turn == 'YES':
                retelling_sysact_nodelx_dict = modified_sysact_nodelx_dict

            data[dial_idx]['log'][turn_idx]['sysact_nodelx_dict'] = deepcopy(act_dict)
            data[dial_idx]['log'][turn_idx]['activated_constraint_dict'] = deepcopy(activated_state_dict)
            data[dial_idx]['log'][turn_idx]['present_turn_dict'] = deepcopy(present_turn)
            data[dial_idx]['log'][turn_idx]['retelling_sysact_nodelx_dict'] = deepcopy(retelling_sysact_nodelx_dict)

            act_history.append([modified_sysact_nodelx_dict, user_act_dict])
            pre_state_dict = state_dict

    with open(target_path, 'w') as f:
        json.dump(data, f, indent=4)
    return 1


if __name__ == "__main__":
    dataset = 'MultiWOZ_2.0'
    if dataset == 'MultiWOZ_2.0':
        data_path = 'data/MultiWOZ_2.0_processed/data_for_damd.json'
        target_path = 'data/MultiWOZ_2.0_processed/data_for_our.json'
    elif dataset == 'MultiWOZ_2.1':
        data_path = 'data/MultiWOZ_2.1_processed/data_for_damd.json'
        target_path = 'data/MultiWOZ_2.1_processed/data_for_our.json'
    else:
        print('select dataset in MultiWOZ_2.0 and MultiWOZ_2.1')
        exit()
    preprocess_mwoz(data_path, target_path)
