import numpy as np
from collections import OrderedDict

class Evaluation_IR:
    def __init__(self):
        self.ks = ks
        self.max_k = max(self.ks)

    def evaluate(self, model, dataset, mode = 'valid'):
        if mode == 'valid':
            eval_mat = dataset.valid_matrix.toarray()
            eval_id = dataset.valid_id
            qid2pid = dataset.qid2pid_valid
            doc_id = dataset.doc_id_valid
        elif mode == 'test':
            eval_mat = dataset.test_matrix.toarray()
            eval_id = dataset.test_id
            qid2pid = dataset.qid2pid
            doc_id = dataset.doc_id

        score_dict = OrderDict()
        topns = self.ks
        # make predictions and compute scores

        close_docs_topk = closeset_docs(eval_matrix.astype('float32'), model, method='dot', mode = mode) ## method: doc, cosine, gating

        ir_score = 

def closest_docs(query, model, method, mode):
    doc_length = 8841823
    query_length = query_vectors.shape[0]
    topk = 1000 
    topk_docs = np.zeros((query_length, topk), dtype=np.int32)

    if method == 'dot':
        document_output = model.get_sparse_output(mode)
    
    return topk_docs

def retrieve_eval_valid(closest_docs, document_ids, query_ids, query2doc, valid_measure, valid_k):
    # closest_docs = [num_query, TOPN]
    # document_ids = [num_document]  : pid 저장
    # query_ids = [num_query]  : qid 저장
    # query2doc = {qid : pid_list}

    tmp_closest = closest_docs[:, :valid_k]
    measures = []

    # For each query
    for i, qid in enumerate(query_ids):
        # predict_pid
        pred_pid = [document_ids[p] for p in tmp_closest[i]]
        # answer_pid
        true_pid = query2doc[qid]
        if len(true_pid) == 0:
            continue

        if 'prec' in valid_measure:
            measures.append(precision_at_k(true_pid, pred_pid, valid_k))
        elif 'recall' in valid_measure:
            measures.append(recall_at_k(true_pid, pred_pid, valid_k))
        elif 'mrr' in valid_measure:
            measures.append(mrr(true_pid, pred_pid, valid_k))
        elif 'map' in valid_measure:
            measures.append(ap_at_k(true_pid, pred_pid, valid_k))
        else:
            raise("Incorrect earlystop measure")

    return np.mean(measures)

def retrieve_eval(closest_docs, doc_ids, query_ids, query2doc, topn_list =[1,3,5]):
    # closest_docs = [num_query, 1000]
    # doc_ids = [num_document]  : pid 저장
    # query_ids = [num_query]  : qid 저장
    # query2doc = {qid : pid_list} 

    precisions = []
    recalls = []
    MRRs = []

    ## Get Precisions & Recall & MRR
    for n_docs in topn_list:
        tmp_closest = closest_docs[:, :n_docs]

        precisions_k = []
        recalls_k = []
        MRRs_k = []
        # For each query
        for i, qid in enumerate(query_ids):
            if not query2doc.get(qid):
                continue
            # predict_pid
            pred_pid = [str(document_ids[p]) for p in tmp_closest[i]] ## [10, 1, 2, 3] document_ids[10]:
            # answer_pid
            true_pid = query2doc[qid] ## 7067032
            if len(true_pid) == 0:
                continue

            precisions_k.append(precision_at_k(true_pid, pred_pid, n_docs))
            recalls_k.append(recall_at_k(true_pid, pred_pid, n_docs))
            MRRs_k.append(mrr(true_pid, pred_pid, n_docs))
            
        precisions.append(np.mean(precisions_k))
        recalls.append(np.mean(recalls_k))
        MRRs.append(np.mean(MRRs_k))

    ## Get MAP
    # For each query
    MAP_k = []
    for i, qid in enumerate(query_ids):
        if not query2doc.get(qid):
            continue
        # predict_pid
        pred_pid = [str(doc_ids[p]) for p in closest_docs[i]]
        # answer_pid
        true_pid = query2doc[qid]
        if len(true_pid) == 0:
            continue

        MAP_k.append(ap_at_k(true_pid, pred_pid, 1000))
    MAP = np.mean(MAP_k)

    return precisions, recalls, MRRs, MAP

def precision_at_k(actual, predicted, k):
    pred_top_k = predicted[:k]

    true_positives = sum(1 for item in pred_top_k if item in set(actual))

    # precision@k
    precision = true_positives / k
    
    return precision
    
def recall_at_k(actual, predicted, k):
    pred_top_k = predicted[:k]

    true_positives = sum(1 for item in pred_top_k if item in set(actual))

    total_actual = len(set(actual))

    # recall@k
    recall = true_positives / total_actual if total_actual > 0 else 0 
    
    return recall

def ap_at_k(actual, predicted, k):
    num_correct = 0
    precision_at_i = []
    for i,p in enumerate(predicted[:k}):
        if p in actual:
            num_correct += 1
            precision_at_i.append(num_correct / (i+1))
    if not precision_at_i:
        return 0
    ap = sum(precision_at_i) / min(len(actual), k)  
    return ap

def mrr(actual, predicted):
    rr_sum = 0
    for ground_truth in actual:
        if ground_truth in predicted:
            rr_sum += 1 / (predicted.index(ground_truth) + 1)
    mrr = rr_sum / len(actual) if actual else 0
    return mrr
