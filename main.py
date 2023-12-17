import logging
import random
import time

import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from datetime import datetime
from data_loader import APIdataset
import config
from utils import get_cosine_schedule_with_warmup, load_dict
from Seq2Seq import Seq2Seq
from Metrics import Metric
from Evaluate import evaluate

def train():
    print(config.emb_dim)
    print(config.n_hidden)
    print(config.n_layers)
    print(config.batch_size)
    modelName = 'RNNEncDec'
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG,format="%(message)s")
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    seed = 1111
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    vocab_api = load_dict(config.vocab_api_file)
    vocab_desc = load_dict(config.vocab_desc_file)

    train_set = APIdataset(config.train_desc_file, config.train_api_file, config.vocab_desc_file, config.vocab_api_file, config.max_desc_length, config.max_api_length, config.max_longtail_length, config.vocab_general_file, config.vocab_longtail_file)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True,
                                               num_workers=1, drop_last=False)

    valid_set = APIdataset(config.test_desc_file, config.test_api_file, config.vocab_desc_file, config.vocab_api_file,
                           config.max_desc_length, config.max_api_length, config.max_longtail_length,
                           config.vocab_general_file, config.vocab_longtail_file)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=config.valid_batch_size, shuffle=True,
                                               num_workers=1, drop_last=False)

    print("Loaded data!")
    if config.pretrain_type=="codet5":
        tokenizer = RobertaTokenizer.from_pretrained(config.pretrain_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.pretrain_file)

    #model = Seq2Seq()
    model = Seq2SeqModel(config.max_desc_len, tokenizer, api_vocab_size, config.max_api_len, config.d_model, config.n_heads, config.d_ff, config.n_layers, config.dropout)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_loader) * config.epochs)

    logger.info("Training...")
    itr_global = 1
    start_epoch = 1
    max_map = 0
    max_iter = 0
    max_time = datetime.now().strftime('%Y%m%d%H%M')
    for epoch in range(start_epoch, config.epochs + 1):
        itr_start_time = time.time()
        logger.info(itr_start_time)
        for batch in train_loader:
            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch]
            loss = model(*batch_gpu)
            loss = torch.mean(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            '''
            if itr_global % config.log_every == 0:
                elapsed = time.time() - itr_start_time
                log = 'epo:[%d/%d] iter:%d steptime: %ds loss:%f' %(epoch, config.epochs, itr_global, elapsed, loss)
                logger.info(log)
                itr_start_time = time.time()
            '''
            if itr_global % config.valid_every == 0:
                model.eval()
                loss_records = {}
                for batch in valid_loader:
                    batch_gpu = [tensor.to(device) for tensor in batch]
                    with torch.no_grad():
                        valid_loss = model.valid(*batch_gpu)
                    for loss_name,loss_value in valid_loss.items():
                        v = loss_records.get(loss_name, [])
                        v.append(loss_value)
                        loss_records[loss_name] = v

                log = 'Valid'
                for loss_name, loss_values in loss_records.items():
                    log = log + loss_name + ':%.4f' %(np.mean(loss_values))
                logger.info(log)
            itr_global += 1
            if itr_global % config.eval_every == 0:
                model.eval()
                test_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size = config.valid_batch_size, shuffle = False, num_workers = 1)

                metrics = Metric()
                os.makedirs(f'./output/{modelName}/{timestamp}/results', exist_ok=True)
                f_eval = open(f"./output/{modelName}/{timestamp}/results/iter{itr_global}.txt",'w')

                topk = config.topk
                max_bleu, avg_bleu, map, ndcg = evaluate(model, metrics, test_loader, vocab_desc, vocab_api, topk, f_eval)

                if max_map == 0 and max_iter == 0:
                    save_model(model, modelName, itr_global, timestamp)
                    max_iter = itr_global
                    max_map = map
                    max_time = timestamp
                else:
                    if map > max_map:
                        delete_model(modelName, max_iter, max_time)
                        max_iter = itr_global
                        max_map = map
                        max_time = timestamp
                        save_model(model, modelName, itr_global, timestamp)








def delete_model(modelName, iter, timestamp):
    os.remove(f'./output/{modelName}/{timestamp}/models/model_iter{iter}.pkl')



def save_model(model, modelName, iter, timestamp):
    os.makedirs(f'./output/{modelName}/{timestamp}/models', exist_ok=True)
    ckpt_path = f'./output/{modelName}/{timestamp}/models/model_iter{iter}.pkl'
    print(f"save_model to {iter}")
    torch.save(model.state_dict(), ckpt_path)







if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True  # fix the random seed in cudnn

    train()
