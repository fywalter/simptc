from datetime import date
import logging
import os
from typing import OrderedDict
import numpy as np
# from openprompt.utils.metrics import classification_metrics
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import *
import csv
import configs
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
datasets = ["agnews", 'dbpedia', "yahoo", "amazon", "imdb"]
base_data_dir = "./datasets/TextClassification"

def set_file_handler(logger, file_name):
    today = date.today().strftime("%b-%d-%Y")
    log_base = "./log"
    if not os.path.exists(log_base):
        os.mkdir(log_base)
    log_dir = os.path.join(log_base, today)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    log_path = os.path.join(log_dir, file_name)
            
    fh = logging.FileHandler(filename=log_path, mode='a')
    formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def classification_metrics(preds: Sequence[int],
                           labels: Sequence[int],
                           metric: Optional[str] = "micro-f1",
                          ) -> float:
    """evaluation metrics for classification task.

    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        metric (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".

    Returns:
        score (float): evaluation score
    """
    
    if metric == "micro-f1":
        score = f1_score(labels, preds, average='micro')
    elif metric == "macro-f1":
        score = f1_score(labels, preds, average='macro')
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "recall":
        score = recall_score(labels, preds)
    elif metric == "f1-all":
        score = f1_score(labels, preds, average=None)
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    return score

def log_performance(pred_list, label_list, logger):
    scores = OrderedDict()
    scores_str = ""
    for metric in ["micro-f1", "macro-f1", "accuracy"]:
        score = classification_metrics(pred_list, label_list, metric)
        scores[metric] = score
        scores_str += "{}: {}\n".format(metric, score)
    scores_total = scores.copy()
    logger.info("{} Performance: {}".format("test", scores_str.strip()))
    preds = np.array(pred_list)
    labels = np.array(label_list)
    for class_id in range(np.max(labels)+1):
        preds_class = preds[labels==class_id]
        labels_class = labels[labels==class_id]
        scores_str = ""
        for metric in ["micro-f1", "macro-f1", "accuracy"]:
            score = classification_metrics(preds_class, labels_class, metric)
            scores[metric] = score
            scores_str += "{}: {}\n".format(metric, score)
        logger.info("{} Performance for class {}: {}".format("test", class_id, scores_str.strip()))
    return scores_total["accuracy"]

def preprocess_word(word):
    # preprocessing for dbpedia word format
    word_list = word.split()
    word_list[0]=word_list[0].capitalize()
    word = "_".join(word_list)
    word = word.replace(" ", "_")
    word = word.replace("'", r'''\'''')
    word = word.replace(",", "")
    word = word.replace(".", "")
    word = word.replace("/", "_")
    word = word.replace("*", "")
    word = word.replace("+", r'''\+''')
    word = word.replace("!", r'''\!''')
    return word

def load_agnews(data_path, label_path=None):
    label_list = []
    text_list = []
    with open(data_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for row in csv_reader:
            label, title, description = row
            label = int(label) - 1
            text = " ".join((title, description))
            label_list.append(label)
            text_list.append(text)
    return text_list, label_list

def load_dbpedia(label_path, data_path):
    label_file  = open(label_path,'r')
    label_list  = [int(x.strip()) for x in label_file.readlines()]
    text_list = []
    with open(data_path,'r') as text_file:
        for text in text_file:
            text_list.append(text)
    return text_list, label_list

def load_yahoo(data_path, label_path=None):
    label_list = []
    text_list = []
    with open(data_path, encoding="utf-8") as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            label, title, content, answer = row
            label = int(label) - 1
            text = " ".join((title, content, answer))
            label_list.append(label)
            text_list.append(text)
    return text_list, label_list

def load_sentiment(label_path, data_path):
    label_file  = open(label_path,'r')
    label_list  = [int(x.strip()) for x in label_file.readlines()]
    text_list = []
    with open(data_path,'r') as text_file:
        for text in text_file:
            text_list.append(text)
    return text_list, label_list

def log_bias_and_performance(preds, labels, logger, log_bias=True):
    confusion_mat = confusion_matrix(labels, preds)
    class_num = confusion_mat.shape[0]
    if log_bias:
        logger.info(f"confusion matrix: \n {confusion_mat}")
        logger.info(f"bias in confusion matrix: {np.sum(confusion_mat, axis=0)}")
        logger.info(f"cluster anchor paired number precision: {np.sum(np.argmax(confusion_mat, axis=0)==np.arange(class_num))}/{class_num}")
        logger.info(f"cluster anchor paired number recall: {np.sum(np.argmax(confusion_mat, axis=1)==np.arange(class_num))}/{class_num}")
    return log_performance(preds, labels, logger)

def get_prompt(logger, dataset, template_id=1, label_name_file=None, model_name='bert_large', big_template=False):
    simcse = model_name in configs.model_names
    if simcse:
        model_name_with_prefix = f"simcse_{True}_" + model_name
    else:
        model_name_with_prefix = model_name
    if big_template:
        file_prefix = "prompt_big"
    else:
        file_prefix = "prompt"
    if label_name_file is not None:
        prompt_path = os.path.join(base_data_dir, dataset, f"embeddings/{file_prefix}_{template_id}_{label_name_file}_{model_name_with_prefix}.npz")
    else:
        prompt_path = os.path.join(base_data_dir, dataset, f"embeddings/{file_prefix}_{template_id}_{model_name_with_prefix}.npz")
    logger.info(f"prompt path: {prompt_path}")
    data_prompt = np.load(prompt_path)
    prompt_embeddings = data_prompt["embedding_list"]
    prompt_labels = data_prompt["label_list"]
    return prompt_embeddings, prompt_labels

def get_train_and_test(logger, dataset, split="train" ,model_name='bert_large'):
    simcse = model_name in configs.model_names
    if simcse:
        model_name_with_prefix = f"simcse_{True}_" + model_name
    else:
        model_name_with_prefix = model_name
    path = os.path.join(base_data_dir, dataset, f"embeddings/{split}_{model_name_with_prefix}.npz")
    logger.info(f"embedding path: {path}")
    data = np.load(path)
    embeddings = data["embedding_list"]
    labels = data["label_list"]
    return embeddings, labels

def topk_similar(anchors, embeddings, k=100):
    dist_mat = distance.cdist(embeddings, anchors, 'cosine')

    # normalize per row
    sum_of_rows = dist_mat.sum(axis=1)
    dist_mat = dist_mat / sum_of_rows[:, np.newaxis]

    class_num = dist_mat.shape[1]

    pseudo_labels = np.argmin(dist_mat, axis=1)
    confusion_mat = confusion_matrix(pseudo_labels, pseudo_labels)
    bias = np.sum(confusion_mat, axis=0)
    assigning_order = np.argsort(-bias) # assign in a descenting order
    
    index_mat = np.zeros((class_num, k))

    for label in assigning_order:   
        sort_idxes = np.argsort(dist_mat[:, label])
        index_mat[label, :] = sort_idxes[:k]
    return index_mat.astype(int)

def match_labels(preds, cost_mat):
    """ assign predictions into distinct labels based on the provided cost matrix
    cost_mat: cost matrix, the (i, j)th element the is cost of assigning cluster i to label j 
    """
    pred_ids, label_ids = linear_sum_assignment(cost_mat)
    preds_copy = np.copy(preds)
    for i in pred_ids:
        preds[preds_copy==i] = label_ids[i]
    return preds