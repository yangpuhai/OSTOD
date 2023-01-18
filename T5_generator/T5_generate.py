import os
os.environ['CUDA_VISIBLE_DEVICES']="2"

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, T5ForConditionalGeneration)

try:
    from .data_loader import prepare_data, add_sepcial_tokens
    from .config import get_args
except ImportError:
    from data_loader import prepare_data, add_sepcial_tokens
    from config import get_args
import json
from tqdm import tqdm


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
    args["model_name"] = "t5" + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_history_" + str(args["history"])
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
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5, verbose=False, mode='min')],
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
    args["model_name"] = "t5" + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_history_" + str(args["history"])
    #save model path
    save_path = os.path.join(args["saving_dir"],args["dataset"],args["model_name"])

    model = T5ForConditionalGeneration.from_pretrained(save_path)
    tokenizer = T5Tokenizer.from_pretrained(save_path)

    task = T5_Seq2Seq(args, tokenizer, model)

    _, _, test_loader = prepare_data(args, task.tokenizer)

    print("test start...")
    #evaluate model
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

    for batch in tqdm(test_loader):

        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=args["length"],
                                )
        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)
        for di, ti, dsv, sys, sysa, pred in zip(batch["dialogue_idx"], batch["turn_idx"], batch["act_str"], batch["system"], batch["system_act"], value_batch):
            if di not in predictions:
                predictions[di] = {}
            if ti not in predictions[di]:
                predictions[di][ti] = {}
            predictions[di][ti]['gold_act'] = ' '.join(dsv.split())
            predictions[di][ti]['gold_utter'] = sys
            predictions[di][ti]['system_act'] = sysa
            predictions[di][ti]['pred_utter'] = pred
    
    # if args["mode"] == 'train':
    with open(os.path.join(save_path, "pred_result.json"), 'w') as f:
        json.dump(predictions, f, indent=4)

def prepare_test(dataset, history):
    args = get_args()
    args = vars(args)
    args["dataset"] = dataset
    args["history"] = history
    args["model_name"] = "t5" + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"]) + "_history_" + str(args["history"])
    #save model path
    save_path = os.path.join(args["saving_dir"],args["dataset"],args["model_name"])

    model = T5ForConditionalGeneration.from_pretrained(save_path)
    tokenizer = T5Tokenizer.from_pretrained(save_path)

    return args, tokenizer, model

if __name__ == "__main__":
    args = get_args()
    if args.mode=="train":
        train(args)
    if args.mode=="retelling_test":
        test(args)
