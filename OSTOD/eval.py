import math, logging, copy, json, re
from collections import Counter, OrderedDict
from nltk.util import ngrams
from transformers import (AdamW, T5Tokenizer, T5ForConditionalGeneration)
import spacy
from clean_dataset import clean_slot_values
from db_ops import MultiWozDB

use_true_bspn_for_ctr_eval = True        
use_true_domain_for_ctr_eval = True
same_eval_as_cambridge = True
use_true_curr_bspn = False

error_info_file = 'new_error_info.json'

nlp = spacy.load('en_core_web_sm')

all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']

special_tokens = ['domain', 'choice', 'price', 'type', 'parking', 'day', 'stay', 'people', 'reference', 'name',
                'address', 'phone', 'postcode', 'destination', 'arrive', 'departure', 'leave', 'stars', 'internet',
                'area', 'time', 'food', 'department', 'id', 'car']

all_domain_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}

requestables = ['phone', 'address', 'postcode', 'reference', 'id']

dbs = {'attraction': '../data/db/attraction_db_processed.json',
        'hospital': '../data/db/hospital_db_processed.json',
        'hotel': '../data/db/hotel_db_processed.json',
        'police': '../data/db/police_db_processed.json',
        'restaurant': '../data/db/restaurant_db_processed.json',
        'taxi': '../data/db/taxi_db_processed.json',
        'train': '../data/db/train_db_processed.json',
        }

db = MultiWozDB(dbs)

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


def same_resp_as_ubar(text):
    for token in special_tokens:
        sub = r'\[s_{}\][^\[]*\[e_{}\]'.format(token, token)
        match_strs = re.findall(sub, text)
        for ms in match_strs:
            if token == 'domain':
                text = text.replace(ms, '')
            else:
                text = text.replace(ms, '[value_%s]'%token)
    return text


def bleu_metric(processed, pred, tokenizer, same_as_ubar=True):
    bleu_scorer = BLEUScorer()
    gen, truth = [],[]
    for dial_idx, dial_dict in processed.items():
        for ti, turn in enumerate(dial_dict["log"]):
            turn_gold = turn['system_delex']
            turn_pred = pred[dial_idx][turn["turn_idx"]]['pred_utter']
            turn_gold = tokenizer.decode(tokenizer.encode(turn_gold)).replace(' [eos]', '')
            turn_pred = tokenizer.decode(tokenizer.encode(turn_pred)).replace(' [eos]', '')
            if same_as_ubar:
                turn_gold = same_resp_as_ubar(turn_gold)
                turn_pred = same_resp_as_ubar(turn_pred)
            truth.append(turn_gold)
            gen.append(turn_pred)
    wrap_generated = [[_] for _ in gen]
    wrap_truth = [[_] for _ in truth]
    if gen and truth:
        sc = bleu_scorer.score(zip(wrap_generated, wrap_truth))
    else:
        sc = 0.0
    return sc

def pointerBack(vector, domain):
    if domain.endswith(']'):
        domain = domain[1:-1]
    if domain != 'train':
        nummap = {
            0: '0',
            1: '1',
            2: '2-3',
            3: '>3'
        }
    else:
        nummap = {
            0: '0',
            1: '1-5',
            2: '6-10',
            3: '>10'
        }
    if vector[:4] == [0,0,0,0]:
        report = ''
    else:
        num = vector.index(1)
        report = domain+': '+nummap[num] + '; '

    if vector[-2] == 0 and vector[-1] == 1:
        report += 'booking: ok'
    if vector[-2] == 1 and vector[-1] == 0:
        report += 'booking: unable'

    return report


def extract_label(text):
    result = {}
    domain_temp = r'\[s_{}\][^\[]*\[e_{}\]'.format('domain', 'domain')
    match_strs = re.findall(domain_temp, text)
    pred_domain = []
    for mi, ms in enumerate(match_strs):
        domain = ms.replace('[s_domain]','').replace('[e_domain]','').strip()
        if domain not in all_domain_slots:
            continue
        if domain not in pred_domain:
            pred_domain.append(domain)
        if mi == len(match_strs)-1:
            domain_text = text[text.index(ms)+len(ms):]
        else:
            domain_text = text[text.index(ms)+len(ms):text.index(match_strs[mi+1])]
        for slot in all_domain_slots[domain]:
            slot_token = slot if slot != 'pricerange' else 'price'
            slot_temp = r'\[s_{}\][^\[]*\[e_{}\]'.format(slot_token, slot_token)
            slot_match_strs = re.findall(slot_temp, domain_text)
            for sms in slot_match_strs:
                value = sms.replace('[s_%s]'%slot_token,'').replace('[e_%s]'%slot_token,'').strip()
                if value != '' and domain+'-'+slot not in result:
                    if slot in ['internet', 'parking']:
                        value = 'yes'
                    result[domain+'-'+slot]=value
    return result, pred_domain


def parseGoal(goal, true_goal, domain):
    """Parses user goal into dictionary format."""
    goal[domain] = {}
    goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
    if 'info' in true_goal[domain]:
        if domain == 'train':
            # we consider dialogues only where train had to be booked!
            if 'book' in true_goal[domain]:
                goal[domain]['requestable'].append('reference')
            if 'reqt' in true_goal[domain]:
                if 'id' in true_goal[domain]['reqt']:
                    goal[domain]['requestable'].append('id')
        else:
            if 'reqt' in true_goal[domain]:
                for s in true_goal[domain]['reqt']:  # addtional requests:
                    if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                        # ones that can be easily delexicalized
                        goal[domain]['requestable'].append(s)
            if 'book' in true_goal[domain]:
                goal[domain]['requestable'].append("reference")

        for s, v in true_goal[domain]['info'].items():
            s_,v_ = clean_slot_values(domain, s,v)
            if len(v_.split())>1:
                v_ = ' '.join([token.text for token in nlp(v_)]).strip()
            goal[domain]["informable"][s_] = v_

        if 'book' in true_goal[domain]:
            goal[domain]["booking"] = true_goal[domain]['book']
    return goal


def evaluateGeneratedDialogue(dialog, goal, real_requestables, counts, soft_acc=False):
    """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""

    # CHECK IF MATCH HAPPENED
    provided_requestables = {}
    venue_offered = {}
    domains_in_goal = []
    bspans = {}

    for domain in goal.keys():
        venue_offered[domain] = []
        provided_requestables[domain] = []
        domains_in_goal.append(domain)

    for t, turn in enumerate(dialog):
        sent_t = turn['resp_gen']
        # sent_t = turn['resp']
        for domain in goal.keys():
            # for computing success
            if same_eval_as_cambridge:
                # [restaurant_name], [hotel_name] instead of [value_name]
                if use_true_domain_for_ctr_eval:
                    dom_pred = turn["gold_domain"]
                else:
                    dom_pred = turn["pred_domain"]
                if domain not in dom_pred:  # fail
                    continue

            if '[s_name]' in sent_t or '[s_id]' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    if not use_true_curr_bspn and not use_true_bspn_for_ctr_eval:
                        bspn = turn['pred_state']
                    else:
                        bspn = turn['gold_state']
                    constraint_dict = bspn
                    if constraint_dict.get(domain):
                        venues = db.queryJsons(domain, constraint_dict[domain], return_name=True)
                    else:
                        venues = []

                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = venues
                        bspans[domain] = constraint_dict[domain]
                    else:
                        flag = False
                        for ven in venues:
                            if  ven not in venue_offered[domain]:
                                flag = True
                                break
                        if flag and venues:  # sometimes there are no results so sample won't work
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                else:  # not limited so we can provide one
                    venue_offered[domain] = '[value_name]'

            # ATTENTION: assumption here - we didn't provide phone or address twice! etc
            for requestable in requestables:
                if requestable == 'reference':
                    if '[s_reference]' in sent_t:
                        if 'ok' in turn['pointer'] or '[s_reference]' in turn['resp']:  # if pointer was allowing for that?
                            provided_requestables[domain].append('reference')
                        # provided_requestables[domain].append('reference')
                else:
                    if '[s_' + requestable + ']' in sent_t:
                        provided_requestables[domain].append(requestable)

    # if name was given in the task
    for domain in goal.keys():
        # if name was provided for the user, the match is being done automatically
        if 'name' in goal[domain]['informable']:
            venue_offered[domain] = '[value_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[value_name]'

        if domain == 'train':
            if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                venue_offered[domain] = '[value_name]'

    """
    Given all inform and requestable slots
    we go through each domain from the user goal
    and check whether right entity was provided and
    all requestable slots were given to the user.
    The dialogue is successful if that's the case for all domains.
    """
    # HARD EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                'taxi': [0, 0, 0],
                'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match = 0
    success = 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            goal_venues = db.queryJsons(domain, goal[domain]['informable'], return_name=True)
            if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
            elif len(venue_offered[domain]) > 0 and len(set(venue_offered[domain])& set(goal_venues))>0:
                match += 1
                match_stat = 1
        else:
            if '_name]' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if soft_acc:
        match = float(match)/len(goal.keys())
    else:
        if match == len(goal.keys()):
            match = 1.0
        else:
            match = 0.0

    for domain in domains_in_goal:
        for request in real_requestables[domain]:
            counts[request+'_total'] += 1
            if request in provided_requestables[domain]:
                counts[request+'_offer'] += 1
    
    # SUCCESS
    if match == 1.0:
        for domain in domains_in_goal:
            success_stat = 0
            domain_success = 0
            if len(real_requestables[domain]) == 0:
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            # for request in set(provided_requestables[domain]):
            #     if request in real_requestables[domain]:
            #         domain_success += 1
            for request in real_requestables[domain]:
                if request in provided_requestables[domain]:
                    domain_success += 1

            # if domain_success >= len(real_requestables[domain]):
            if domain_success == len(real_requestables[domain]):
                success += 1
                success_stat = 1

            stats[domain][1] = success_stat

        # final eval
        if soft_acc:
            success = float(success)/len(real_requestables)
        else:
            if success >= len(real_requestables):
                success = 1
            else:
                success = 0

    return success, match, stats, counts


def state_trans(state):
    state_domain = {}
    for slot_info, value in state.items():
        # if slot_info == 'hotel-type' and value == 'hotel':
        #     continue
        domain, slot = slot_info.split('-')
        if domain not in state_domain:
            state_domain[domain] = {}
        state_domain[domain][slot] = value
    return state_domain


def compute_scores(processed_path, pred_path, tokenizer_path):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    processed_data = json.load(open(processed_path))
    pred_data = json.load(open(pred_path))
    bleu = bleu_metric(processed_data, pred_data, tokenizer)
    all_turn = 0
    ja = 0
    counts = {}
    for req in requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    dial_num, successes, matches = 0, 0, 0
    error_dial = []
    for dial_idx, dial_dict in processed_data.items():
        reqs = {}
        goal = {}
        for domain in all_domains:
            if dial_dict['goal'].get(domain):
                true_goal = dial_dict['goal']
                goal = parseGoal(goal, true_goal, domain)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']

        dial_info = []
        pred_state = {}
        for ti, turn in enumerate(dial_dict["log"]):
            all_turn += 1
            pred_text = pred_data[dial_idx][turn["turn_idx"]]['pred_utter']
            turn_label, pred_domain = extract_label(pred_text)
            pred_state.update(turn_label)
            gold_state = turn['activated_state']
            gold_state = {k:v for k,v in gold_state.items() if v != 'dontcare'}
            if set(pred_state) == set(gold_state):
                ja += 1
            turn_domain = turn["turn_domain"]
            pointer = pointerBack(turn["pointer"], turn_domain[-1])
            dial_info.append({
                "dial_idx": dial_idx,
                "turn_idx": turn["turn_idx"],
                "resp": turn["system_delex"],
                "resp_gen": pred_text,
                "gold_state": state_trans(gold_state),
                "pred_state": state_trans(pred_state),
                "pointer": pointer,
                "gold_domain": turn_domain,
                "pred_domain": pred_domain
            })

        success, match, stats, counts = evaluateGeneratedDialogue(dial_info, goal, reqs, counts)

        if match != 1:
            error_dial.append(dial_idx)

        successes += success
        matches += match
        dial_num += 1
    
    succ_rate = successes/( float(dial_num) + 1e-10) * 100
    match_rate = matches/(float(dial_num) + 1e-10) * 100

    print('joint acc: ',ja/all_turn*1.0)
    print('success: ',succ_rate)
    print('inform: ', match_rate)
    print('bleu: ', bleu)
    score = 0.5 * (succ_rate + match_rate) + bleu
    print('Combined score: ', score)
    
    return 1


if __name__ == "__main__":
    processed_path = 'processed_data/MultiWOZ_2.0/test.json'
    pred_path = 'save/MultiWOZ_2.0/t5_lr_0.0005_epoch_20_seed_42_scale_1.0/results/pred_result.json'
    tokenizer_path = 'save/MultiWOZ_2.0/t5_lr_0.0005_epoch_20_seed_42_scale_1.0'
    compute_scores(processed_path, pred_path, tokenizer_path)