import os, json, re
import torch
from tqdm import tqdm
from domain_split_config import get_args
from domain_split_data_loader import prepare_data, match_value, match_candidate
from T5_generator.T5_generate import prepare_test as prepare_gen
from T5_filter.T5_generate import prepare_test as prepare_fil

os.environ['CUDA_VISIBLE_DEVICES']="2"

def test(args):
    args = vars(args)

    save_path = os.path.join(args["saving_dir"], args["dataset"], args["set"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    GEN_MODEL_DIR = args["gen_dir"]
    FIL_MODEL_DIR = args["fil_dir"]
    
    os.chdir(GEN_MODEL_DIR)
    gen_args, gen_tokenizer, gen_model = prepare_gen(args["dataset"], args["gen_history"])
    os.chdir('../')
    os.chdir(FIL_MODEL_DIR)
    fil_args, fil_tokenizer, fil_model = prepare_fil(args["dataset"], args["fil_history"])
    os.chdir('../')

    gen_loader, fil_loader = prepare_data(args, gen_tokenizer, fil_tokenizer)

    print("processing start...")

    predictions = {}
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:0")
    gen_model.to(device1)
    gen_model.eval()
    fil_model.to(device2)
    fil_model.eval()

    wrong_result = {}

    pattern = re.compile(r'\((.*)\)')

    count = 0
    correct = 0

    for batch in tqdm(gen_loader):

        dst_outputs = gen_model.generate(input_ids=batch["encoder_input"].to(device1),
                                attention_mask=batch["attention_mask"].to(device1),
                                eos_token_id=gen_tokenizer.eos_token_id,
                                max_length=gen_args["length"],
                                num_beams=5,
                                num_return_sequences=5
                                )
        value_batch = gen_tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        value_batch_list = [value_batch[i:i + 5] for i in range(0, len(value_batch), 5)]
        for di, ti, si, sd, sa, usr, sys, outp, inp, fil_dsv, fil_inp, act_v, pred in zip(batch["dialogue_idx"], batch["turn_idx"], batch["split_idx"], batch["split_domain"], batch["system_act"], batch["user"], batch["system"], batch["output_text"], batch["input_text"], batch["fil_act_str"], batch["fil_input_text"], batch["act_values"], value_batch_list):
            matched_pred = match_value(act_v, pred)
            if matched_pred == []:
                candidate_pred = pred
            else:
                candidate_pred = matched_pred
            candidate_input = [fil_inp.replace(sys, cp) for cp in candidate_pred]
            candidate_input_batch = fil_tokenizer(candidate_input, padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
            candidate_dst_outputs = fil_model.generate(input_ids=candidate_input_batch["input_ids"].to(device2),
                                attention_mask=candidate_input_batch["attention_mask"].to(device2),
                                eos_token_id=fil_tokenizer.eos_token_id,
                                max_length=fil_args["length"],
                                )
            candidate_value_batch = fil_tokenizer.batch_decode(candidate_dst_outputs, skip_special_tokens=True)
            chosen_candidate = match_candidate(' '.join(fil_dsv.split()), [cv.replace(',', ' ,') for cv in candidate_value_batch], candidate_pred)
            if di not in predictions:
                predictions[di] = {}
            if ti not in predictions[di]:
                predictions[di][ti] = {}
            if si not in predictions[di][ti]:
                predictions[di][ti][si] = {}
            predictions[di][ti][si]['user_utter'] = usr
            predictions[di][ti][si]['gold_utter'] = outp.replace(' [eos]', '')
            predictions[di][ti][si]['pred_utter'] = chosen_candidate
            predictions[di][ti][si]['input_utter'] = inp
            predictions[di][ti][si]['split_domain'] = sd
            predictions[di][ti][si]['split_act'] = sa
    
    for batch in tqdm(fil_loader):
        batch["input_text"] = [tx.replace(sy, predictions[di][ti][si]['pred_utter']) for di, ti, si, sy, tx in zip(batch["dialogue_idx"], batch["turn_idx"], batch["split_idx"], batch["system"], batch["input_text"])]
        input_batch = fil_tokenizer(batch["input_text"], padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
        batch["encoder_input"] = input_batch["input_ids"]
        batch["attention_mask"] = input_batch["attention_mask"]
        dst_outputs = fil_model.generate(input_ids=batch["encoder_input"].to(device2),
                                attention_mask=batch["attention_mask"].to(device2),
                                eos_token_id=fil_tokenizer.eos_token_id,
                                max_length=fil_args["length"],
                                )
        value_batch = fil_tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        for di, ti, si, dsv, pred in zip(batch["dialogue_idx"], batch["turn_idx"], batch["split_idx"], batch["act_str"], value_batch):
            predictions[di][ti][si]['gold_act'] = ' '.join(dsv.split())
            predictions[di][ti][si]['pred_act'] = pred.replace(',', ' ,')

            list_gold = predictions[di][ti][si]['gold_act'].split(' , ')
            new_list_gold = []
            for l in list_gold:
                if "(" in l and ")" in l:
                    act = ' '.join(l.split()[:2])
                    slots = pattern.findall(l)[0].strip().split()
                    for slot in slots:
                        new_list_gold.append(act + ' ' + slot)
                else:
                    new_list_gold.append(l)

            list_pred = predictions[di][ti][si]['pred_act'].split(' , ')
            new_list_pred = []
            for l in list_pred:
                if "(" in l and ")" in l:
                    act = ' '.join(l.split()[:2])
                    slots = pattern.findall(l)[0].strip().split()
                    for slot in slots:
                        new_list_pred.append(act + ' ' + slot)
                else:
                    new_list_pred.append(l)

            count += 1
            if set(new_list_gold) == set(new_list_pred):
                correct += 1
            else:
                if di not in wrong_result:
                    wrong_result[di] = {}
                if ti not in wrong_result[di]:
                    wrong_result[di][ti] = {}
                wrong_result[di][ti][si] = predictions[di][ti][si]

    print(count)
    print(correct)
    print(correct*1.0/count)
    
    acc_result = {'all_sample':count, 'correct_sample':correct, 'acc':correct*1.0/count}

    with open(os.path.join(save_path, "pred_result.json"), 'w') as f:
        json.dump(predictions, f, indent=4)
    with open(os.path.join(save_path, "acc.json"), 'w') as f:
        json.dump(acc_result,f, indent=4)
    with open(os.path.join(save_path, "wrong_result.json"), 'w') as f:
        json.dump(wrong_result,f, indent=4)

if __name__ == "__main__":
    args = get_args()
    test(args)
