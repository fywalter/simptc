import argparse
import numpy as np
from numpy import linalg as LA
import os
import logging
from utils import set_file_handler, log_bias_and_performance, get_prompt, get_train_and_test
from sklearn.metrics import davies_bouldin_score
from scipy.spatial import distance
from sklearn.decomposition import PCA
import configs
from model import VariationalSimPTC

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="exploring unsupervised clustering")
    parser.add_argument(
        "--dataset", type=str, default="agnews", help="dataset name, from agnews, dbpedia, yahoo, imdb, amazon"
    )
    parser.add_argument(
        "--template_id", type=int, default=None, help="template_id"
    )
    args = parser.parse_args()
    return args

def variational_simptc(unlabeled_embeddings, anchors, test_embeddings, test_labels, cov_type="tied", max_iter=50, init_labels=None):
    embeddings = unlabeled_embeddings
    logger.info(f"unlabel data num: {unlabeled_embeddings.shape[0]}")
    n_samples, _ = embeddings.shape
    class_num = np.max(test_labels)+1
    if init_labels is None:
        logger.info("============encode and match=============")
        prompt_pred = np.argmin(distance.cdist(test_embeddings, anchors, 'cosine'), axis=1)
        pseudo_labels = np.argmin(distance.cdist(embeddings, anchors, 'cosine'), axis=1)
        log_bias_and_performance(prompt_pred, test_labels, logger)
    else:
        pseudo_labels = init_labels
    logger.info("davies_bouldin_score: ")
    logger.info(davies_bouldin_score(embeddings, pseudo_labels))
    
    # compute cov prior
    feature_dim = embeddings.shape[1]
    data_cov_list = np.zeros((class_num, feature_dim, feature_dim))
    for i in range(class_num):
        embeddings_class = embeddings[pseudo_labels==i]
        data_cov_list[i] = np.cov(np.transpose(embeddings_class))
    cov_mean = data_cov_list.mean(axis=0)

    model = VariationalSimPTC(
        n_components=class_num,
        covariance_type=cov_type,
        max_iter=1,
        weight_concentration_prior_type="dirichlet_distribution",
        weight_concentration_prior=int(len(embeddings)/class_num),    # set effective prior to N/K
        mean_precision_prior=1e-6,   # non-informative prior
        mean_prior=embeddings.mean(axis=0),
        degrees_of_freedom_prior=embeddings.shape[1],    # n_feature: non-informative prior
        covariance_prior=cov_mean,
    )
    
    model.initialize_parameters_with_data(embeddings, pseudo_labels)
    best_iter = 0
    best_acc = 0
    best_score = 10000
    best_score_iter = 0
    acc_list = []
    for i in range(max_iter):
        pseudo_labels_new = model.fit_predict(embeddings)
        if (pseudo_labels == pseudo_labels_new).all():
            break
        else:
            pseudo_labels = pseudo_labels_new
        y_test_pred = model.predict(test_embeddings)
        logger.info(f"============Variational SimPTC performance iter {i}=============")
        logger.info("davies_bouldin_score: ")
        new_score = davies_bouldin_score(embeddings, pseudo_labels)
        logger.info(new_score)
        acc = log_bias_and_performance(y_test_pred, test_labels, logger) 
        acc_list.append(acc)
        if acc>best_acc:
            best_acc = acc
            best_iter = i
            logger.info(f"New best acc: {best_acc} at iter {best_iter}")
        if new_score<best_score:
            best_score = new_score
            best_score_iter = i
            logger.info(f"New best score: {best_score} at iter {best_score_iter}")
    logger.info(f"final best acc: {best_acc} at iter {best_iter}")
    logger.info(f"final best score: {best_score} at iter {best_score_iter} with acc {acc_list[best_score_iter]}")

def run(dataset, model_name='roberta_large', template_id=1, label_name_file="conceptnet_1000"):
    cov_type = 'tied' if dataset in ['imdb', 'amazon'] else 'full'
    max_iter = configs.max_iters[dataset]
    logger.info(f"==============Setting================")
    logger.info(f"dataset={dataset}")
    logger.info(f"model name={model_name}")
    logger.info(f"label_name_file={label_name_file}")
    logger.info(f"cov_type={cov_type}")
    logger.info(f"max iter={max_iter}")
    logger.info(f"template id = {template_id}, template = {configs.TEMPLATES[dataset][template_id]}")

    prompt_embeddings, prompt_labels = get_prompt(logger, dataset, template_id=template_id, label_name_file=label_name_file, model_name=model_name)
    test_embeddings, test_labels = get_train_and_test(logger, dataset, split="test", model_name=model_name)
    train_embeddings, train_labels = get_train_and_test(logger, dataset, split="train", model_name=model_name)
    class_num = np.max(test_labels) + 1
    prompt_cluster_centers = np.array([prompt_embeddings[prompt_labels == i].mean(axis=0) for i in range(class_num)])

    unlabeled_embeddings = np.concatenate((train_embeddings, test_embeddings))

    variational_simptc(unlabeled_embeddings=unlabeled_embeddings, anchors=prompt_cluster_centers, test_embeddings=test_embeddings,
            test_labels=test_labels, cov_type=cov_type, max_iter=max_iter)

def get_embeddings_TC14(logger, dataset):
    base_data_dir = f'./datasets/TextClassification/'
    prompt_path = os.path.join(base_data_dir, dataset, f"embeddings/prompt_0_simcse_True_roberta_large.npz")
    text_path = os.path.join(base_data_dir, dataset, f"embeddings/simcse_roberta_large.npz")
    logger.info(f"prompt path: {prompt_path}")
    logger.info(f"text path: {text_path}")
    data_prompt = np.load(prompt_path)
    prompt_embeddings = data_prompt["embedding_list"]
    prompt_labels = data_prompt["label_list"]
    data_text = np.load(text_path)
    text_embeddings = data_text['embedding_list']
    text_labels = data_text['label_list']
    return text_embeddings, text_labels, prompt_embeddings, prompt_labels

def run_TC14(dataset, model_name='roberta_large', cov_type='tied', max_iter=50, pca_percentage=0.97):
    pca = True if dataset in ['banking77', 'bbc-news'] else False
    logger.info(f"==============Setting================")
    logger.info(f"dataset={dataset}")
    logger.info(f"model name={model_name}")
    logger.info(f"cov_type={cov_type}")
    logger.info(f"max iter={max_iter}")
    logger.info(f"pca={pca}")
    logger.info(f"pca percentage={pca_percentage}")
    logger.info(f"template = <mask>")
    text_embeddings, text_labels, prompt_embeddings, prompt_labels = get_embeddings_TC14(logger, dataset)
    class_num = np.max(text_labels) + 1
    prompt_cluster_centers = np.array([prompt_embeddings[prompt_labels == i].mean(axis=0) for i in range(class_num)])
    cluster_centers = np.array([text_embeddings[text_labels == i].mean(axis=0) for i in range(class_num)])
    unlabeled_embeddings = text_embeddings

    if pca:
        # keep the reconstruction error as 3%
        cov_mat = np.cov(np.transpose(unlabeled_embeddings))
        e_values, _ = LA.eig(cov_mat)
        e_values[::-1].sort()
        cumulative_sum = np.cumsum(e_values)
        total_sum = e_values.sum()
        percentages = cumulative_sum/total_sum
        pca_dim = np.sum(percentages < pca_percentage)
        logger.info(f"pca dim = {pca_dim}")
        model_pca = PCA(n_components=pca_dim)
        unlabeled_embeddings = model_pca.fit_transform(unlabeled_embeddings)
        text_embeddings = model_pca.transform(text_embeddings)
        prompt_cluster_centers = model_pca.transform(prompt_cluster_centers)

    logger.info("============encode and match=============")
    prompt_pred = np.argmin(distance.cdist(text_embeddings, prompt_cluster_centers, 'cosine'), axis=1)
    log_bias_and_performance(prompt_pred, text_labels, logger)

    variational_simptc(unlabeled_embeddings=unlabeled_embeddings, anchors=prompt_cluster_centers, test_embeddings=text_embeddings,
            test_labels=text_labels, cov_type=cov_type, max_iter=max_iter)

def main():
    args = parse_args()
    dataset = args.dataset
    model_name = 'roberta_large'
    template_id = args.template_id
    log_file_name = f"run_{dataset}_{template_id}.log"
    set_file_handler(logger, log_file_name)

    if dataset in configs.benchmarks:
        run(dataset, model_name, template_id)
    else:
        run_TC14(dataset, model_name)

if __name__ == "__main__":
    main()