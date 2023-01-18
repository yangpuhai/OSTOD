import os
os.environ['CUDA_VISIBLE_DEVICES']="4,5,6,7"

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, T5ForConditionalGeneration)
from data_loader import prepare_data, add_sepcial_tokens, match_candidate
from config import get_args
import json
from tqdm import tqdm
from db_ops import db_tokens

class T5_Seq2Seq(pl.LightningModule):

    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]

    def training_step(self, batch, batch_idx):
        loss = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]).loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]).loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)



def train(args, *more):
    args = vars(args)
    args["model_name"] = "t5" + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_scale_" + str(args["scale"])
    # train!
    seed_everything(args["seed"])

    model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
    tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], eos_token="[eos]", sep_token="[sep]")
    add_sepcial_tokens(tokenizer)
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    task = T5_Seq2Seq(args, tokenizer, model)

    train_loader, val_loader, test_loader = prepare_data(args, task.tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"],args["dataset"],args["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=20, verbose=False, mode='min')],
                    gpus=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    strategy="ddp"
                    )

    trainer.fit(task, train_loader, val_loader)

    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    return 1

def test(args, *more):
    args = vars(args)
    args["model_name"] = "t5" + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_scale_" + str(args["scale"])
    #save model path
    save_path = os.path.join(args["saving_dir"],args["dataset"],args["model_name"])

    model = T5ForConditionalGeneration.from_pretrained(save_path)
    tokenizer = T5Tokenizer.from_pretrained(save_path)

    task = T5_Seq2Seq(args, tokenizer, model)

    if args["mode"] == "test_sequential":
        test_data, dbs = prepare_data(args, task.tokenizer)
        print("test start...")
        test_model_sequential(args, task.tokenizer, task.model, test_data, dbs, save_path)
    else:
        _, _, test_loader = prepare_data(args, task.tokenizer)
        print("test start...")
        test_model(args, task.tokenizer, task.model, test_loader, save_path)

def test_model(args, tokenizer, model, test_loader, save_path):
    
    save_path = os.path.join(save_path,"results")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    batch_num = 0
    for batch in tqdm(test_loader):
        batch_num += 1

        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=args["length"],
                                num_beams=10,
                                num_return_sequences=1
                                )
        value_batch = tokenizer.batch_decode(dst_outputs)
        for di, ti, usr, sys, sysd, present, pred in zip(batch["dialogue_idx"], batch["turn_idx"], batch["user"], batch["system"], batch["system_delex"], batch["present_turn"], value_batch):
            if di not in predictions:
                predictions[di] = {}
            if ti not in predictions[di]:
                predictions[di][ti] = {}
            predictions[di][ti]['user'] = usr
            predictions[di][ti]['gold_system'] = sys
            predictions[di][ti]['gold_system_delex'] = sysd
            predictions[di][ti]['present_turn'] = present
            predictions[di][ti]['pred_utter'] = ' '.join(pred.replace(tokenizer.sep_token,'').replace(tokenizer.eos_token,'').replace(tokenizer.pad_token,'').strip().split())
        
    # if args["mode"] == 'train':
    with open(os.path.join(save_path, "pred_result.json"), 'w') as f:
        json.dump(predictions, f, indent=4)


def test_model_sequential(args, tokenizer, model, test_data, dbs, save_path):
    
    save_path = os.path.join(save_path,"results_sequential")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data = {}
    for turn in test_data:
        if turn["dialogue_idx"] not in data:
            data[turn["dialogue_idx"]] = {}
        data[turn["dialogue_idx"]][turn["turn_idx"]] = turn

    predictions = {}
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    count = 0
    for dial_idx, dial_dict in tqdm(data.items()):
        turn_num = len(dial_dict)
        pre_state = {}
        # print(dial_idx)
        for turn_idx in range(turn_num):
            # print(turn_idx)
            turn_dict = dial_dict[str(turn_idx)]
            input_text = turn_dict["input_text"]
            db_pointer = turn_dict["db_pointer"]
            inputs = [input_text.replace(db_pointer, t) for t in db_tokens]
            inputs_batch = tokenizer(inputs, padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
            outputs = model.generate(input_ids=inputs_batch["input_ids"].to(device),
                                attention_mask=inputs_batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=args["length"],
                                num_beams=10,
                                num_return_sequences=1
                                )
            output_texts = tokenizer.batch_decode(outputs)
            chosen_text, turn_label = match_candidate(dbs, output_texts, pre_state, tokenizer, turn_dict["turn_domain"])
            pre_state.update(turn_label)
            di = dial_idx
            ti = str(turn_idx)
            if di not in predictions:
                predictions[di] = {}
            if ti not in predictions[di]:
                predictions[di][ti] = {}
            predictions[di][ti]['user'] = turn_dict["user"]
            predictions[di][ti]['gold_system'] = turn_dict["system"]
            predictions[di][ti]['gold_system_delex'] = turn_dict["system_delex"]
            predictions[di][ti]['present_turn'] = turn_dict["present_turn"]
            predictions[di][ti]['pred_utter'] = chosen_text
            predictions[di][ti]['gold_db'] = turn_dict["db_pointer"]

    # if args["mode"] == 'train':
    with open(os.path.join(save_path, "pred_result.json"), 'w') as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    if args.mode=="train":
        train(args)
    if args.mode=="test" or args.mode=="test_sequential" :
        test(args)
