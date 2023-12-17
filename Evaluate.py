import numpy as np
import torch
from utils import index2str, index2strlist, index2apirefstr

def evaluate(model, metric, test_loder, vocab_desc, vocab_api, top_k, f_eval):
    #vocab_desc_i = {v: k for k, v  in vocab_desc}
    vocab_desc_i = {vocab_desc[k]:k for k in vocab_desc.keys()}
    vocab_api_i = {vocab_api[k]:k for k in vocab_api.keys()}
    device = next(model.parameters()).device
    map = []
    mrr = []
    ndcg = []
    max_all_bleu = []
    avg_all_bleu = []
    local_iter = 0
    for descs, apis, long_tails in test_loder:
        descs, _ = [tensor.to(device) for tensor in [descs, descs]]
        with torch.no_grad():
            sample_words,_ = model.sample(descs, top_k)
        desc_ref = [i for i in descs[0].cpu().numpy() if i>0]
        api_ref = [i for i in apis[0].cpu().numpy() if i>0]
        api_ref = api_ref[1:]
        long_tail_ref = long_tails[0].cpu().numpy()
        long_tail_ref = long_tail_ref[1:]
        results_api = sample_words
        map.append(metric.MAP(results_api, api_ref, long_tail_ref))
        ndcg.append(metric.NDCG(results_api, api_ref, long_tail_ref))
        max_bleu, avg_bleu = metric.bleu(results_api, api_ref, long_tail_ref)
        max_all_bleu.append(max_bleu)
        avg_all_bleu.append(avg_bleu)

        local_iter += 1
        descs_str = index2str(desc_ref, vocab_desc_i)
        apis_str = index2apirefstr(api_ref, long_tail_ref, vocab_api_i)
        api_str_list = index2strlist(results_api, vocab_api_i)

        f_eval.write("local_iter %d \n" % (local_iter))
        f_eval.write(f"query: {descs_str} \n")
        f_eval.write("target: %s \n" % (apis_str))
        for r_id, api_result in enumerate(api_str_list):
            f_eval.write("result %d >> %s \n" % (r_id, api_result))
        f_eval.write("\n")

    max_bleu_mean = float(np.mean(max_all_bleu))
    avg_bleu_mean = float(np.mean(avg_all_bleu))
    map_mean = float(np.mean(map))
    ndcg_mean = float(np.mean(ndcg))

    report = "Avg maxBLEU %f, avg BLEU %f, avg MAP %f, avg NDCG %f" % (max_bleu_mean, avg_bleu_mean, map_mean, ndcg_mean)
    print(report)
    f_eval.write(report + '\n')

    return max_bleu_mean, avg_bleu_mean, map_mean, ndcg_mean



