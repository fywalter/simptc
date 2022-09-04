import logging
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import numpy as np
from utils import set_file_handler
import argparse
import configs
import json

logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_file_name = "encode_dataset.log"
set_file_handler(logger, log_file_name)
logger.info('Using {} device'.format(device))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='imdb', help="dataset name"
    )
    args = parser.parse_args()
    return args

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_sentence(text, model, tokenizer, simcse=True):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    encoded_input.to(device)
    with torch.no_grad():
        if simcse:
            sent_embedding = model(**encoded_input, output_hidden_states=True, return_dict=True).pooler_output.cpu().detach().numpy()
        else:
            model_output = model(**encoded_input)
            sent_embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().detach().numpy()
    if not isinstance(text, list):
        sent_embedding = sent_embedding.flatten()
    return sent_embedding

def load_texts_and_labels(dataset):
    data_path = f"./datasets/TextClassification/{dataset}/data.json"
    f = open(data_path)
    data = json.load(f)
    return data['texts'], data['labels']

def load_label_names(dataset):
    label_name_file = f"./datasets/TextClassification/{dataset}/label_names.txt"
    labels = []
    with open(label_name_file, "r") as f:
        for line in f:
            words = line.split("\n")[0].split(",")
            labels.append(words)
    return labels

def load_and_encode_agnews(data_path, model, tokenizer, label_path=None, simcse=True):
    label_list = []
    embedding_list = []
    with open(data_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for row in tqdm(csv_reader):
            label, title, description = row
            label = int(label) - 1
            text = " ".join((title, description))
            label_list.append(label)
            sent_embedding = encode_sentence(text, model, tokenizer, simcse)
            embedding_list.append(sent_embedding)
    return embedding_list, label_list

def load_and_encode_dbpedia(label_path, data_path, model, tokenizer, simcse=True):
    label_file  = open(label_path,'r')
    label_list  = [int(x.strip()) for x in label_file.readlines()]
    embedding_list = []
    with open(data_path,'r') as text_file:
        for text in tqdm(text_file):
            sent_embedding = encode_sentence(text, model, tokenizer, simcse)
            embedding_list.append(sent_embedding)
    return embedding_list, label_list

def load_and_encode_yahoo(data_path, model, tokenizer, label_path=None, simcse=True):
    label_list = []
    embedding_list = []
    with open(data_path, encoding="utf-8") as csv_file:
        rows = csv.reader(csv_file)
        for row in tqdm(rows):
            label, title, content, answer = row
            label = int(label) - 1
            text = " ".join((title, content, answer))
            label_list.append(label)
            sent_embedding = encode_sentence(text, model, tokenizer, simcse)
            embedding_list.append(sent_embedding)
    return embedding_list, label_list

def load_and_encode_sentiment(label_path, data_path, model, tokenizer, simcse=True):
    label_file  = open(label_path,'r')
    label_list  = [int(x.strip()) for x in label_file.readlines()]
    embedding_list = []
    with open(data_path,'r') as text_file:
        for text in tqdm(text_file):
            sent_embedding = encode_sentence(text, model, tokenizer, simcse)
            embedding_list.append(sent_embedding)
    return embedding_list, label_list

def encode_label_names(dataset, label_name_file, model, tokenizer, model_name):
    logger.info("encoding class anchor sentences...")
    embedding_dir = f'./datasets/TextClassification/{dataset}/embeddings'
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    simcse = model_name in configs.model_names
    labels = []
    with open(label_name_file, "r") as f:
        for line in f:
            words = line.split("\n")[0].split(",")
            labels.append(words)

    template_number = len(configs.TEMPLATES[dataset])
    for template_id in range(template_number):
        if simcse:
            model_name_with_prefix = f"simcse_{True}_" + model_name
        else:
            model_name_with_prefix = model_name
        if label_name_file.split('/')[-1] == 'label_names.txt':
            prompt_path = os.path.join(embedding_dir, f"prompt_{template_id}_{model_name_with_prefix}.npz")
        else:
            label_names_name = label_name_file.split('/')[-1][12:-4]   # extract label names name
            prompt_path = os.path.join(embedding_dir, f"prompt_{template_id}_{label_names_name}_{model_name_with_prefix}.npz")
        template = configs.TEMPLATES[dataset][template_id]
        # create prompt embeddings: 
        if not os.path.exists(prompt_path):
            logger.info(f"generating prompt embeddings using {model_name} from {label_name_file}, template: {template}")
            logger.info(f"embedding path = {prompt_path}")
            label_list = []
            prompt_embeddings = []
            for label_id, label_words in enumerate(labels):
                show_example = True
                for label_word in label_words:
                    prompted_text = template.replace("<mask>", label_word)
                    if show_example:
                        show_example = False
                        logger.info(f"prompt exampe: {prompted_text}")
                    prompt_embeddings.append(encode_sentence(prompted_text, model, tokenizer, simcse=simcse))
                    label_list.append(label_id)
            prompt_embeddings = np.array(prompt_embeddings)
            label_list = np.array(label_list)
            logger.info(f"embedding shape: {prompt_embeddings.shape}")
            np.savez(prompt_path, embedding_list=prompt_embeddings, label_list=label_list)
        else:
            logger.info(f"prompt embeddings already exists at {prompt_path}")

def encode_and_save_sentences(dataset, suffix, model, tokenizer, model_name, split):
    logger.info("encoding train and test set...")
    label_path = os.path.join(f"./datasets/TextClassification/{dataset}", f"{split}_labels.{suffix}")
    embedding_dir = os.path.join(f"./datasets/TextClassification/{dataset}", "embeddings")
    
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    if model_name in ["roberta_large_vanilla", "roberta_large_sentence"]:
        path = os.path.join(embedding_dir, f"{split}_{model_name}.npz")
        simcse = False
    else:
        path = os.path.join(embedding_dir, f"{split}_simcse_True_{model_name}.npz")
        simcse = True
    
    # create testing embeddings:
    embedding_path = path
    data_path = os.path.join(f"./datasets/TextClassification/{dataset}", f"{split}.{suffix}")
    if not os.path.exists(embedding_path):
        logger.info(f"generating train/test embeddings using {model_name} from {data_path}")
        encode_functions = {
            "agnews": load_and_encode_agnews,
            "dbpedia": load_and_encode_dbpedia,
            "yahoo": load_and_encode_yahoo,
            "imdb": load_and_encode_sentiment,
            "amazon": load_and_encode_sentiment,
        }
        embedding_list, label_list = encode_functions[dataset](label_path=label_path, data_path=data_path, model=model, tokenizer=tokenizer, simcse=simcse)
        label_list = np.array(label_list)
        embedding_list = np.array(embedding_list)
        logger.info(f"embedding size: {embedding_list.shape}, label shape: {label_list.shape}")
        logger.info(f"embeddings saved at {embedding_path}")
        np.savez(embedding_path, embedding_list=embedding_list, label_list=label_list)
    else:
        logger.info(f"embeddings already exists at {embedding_path}")
    return path

def encode_and_save_5_benchmarks(dataset, suffix, model, tokenizer, model_name):
    label_name_file = f"./datasets/TextClassification/{dataset}/label_names_conceptnet_1000.txt"

    encode_label_names(dataset, label_name_file, model, tokenizer, model_name)
    for split in ['train', 'test']:
        encode_and_save_sentences(dataset, suffix, model, tokenizer, model_name, split)

def encode_texts_and_labels_TC14(dataset, model, tokenizer, model_name, embedding_path, simcse):
    logger.info("encoding train and test set...")
    if not os.path.exists(embedding_path):
        texts, labels = load_texts_and_labels(dataset)
        logger.info(f"generating train/test embeddings using {model_name} for {dataset} dataset")
        embedding_list = []
        for text in tqdm(texts):
            embedding_list.append(encode_sentence(text, model, tokenizer, simcse))
        label_list = np.array(labels)
        embedding_list = np.array(embedding_list)

        logger.info(f"embedding size: {embedding_list.shape}, label shape: {label_list.shape}")
        logger.info(f"embeddings saved at {embedding_path}")
        np.savez(embedding_path, embedding_list=embedding_list, label_list=label_list)
    else:
        logger.info(f"embeddings already exists at {embedding_path}")

def encode_label_names_TC14(dataset, model, tokenizer, model_name, prompt_path, simcse, template = "<mask>"):
    logger.info("encoding anchor sentences...")
    label_names = load_label_names(dataset)

    # create prompt embeddings: 
    if not os.path.exists(prompt_path):
        logger.info(f"generating prompt embeddings using {model_name} for {dataset} dataset, template: {template}")
        logger.info(f"embedding path = {prompt_path}")
        label_list = []
        prompt_embeddings = []
        for label_id, label_words in enumerate(label_names):
            show_example = True
            for label_word in label_words:
                prompted_text = template.replace("<mask>", label_word)
                if show_example:
                    show_example = False
                    logger.info(f"prompt exampe: {prompted_text}")
                prompt_embeddings.append(encode_sentence(prompted_text, model, tokenizer, simcse=simcse))
                label_list.append(label_id)
        prompt_embeddings = np.array(prompt_embeddings)
        label_list = np.array(label_list)
        logger.info(f"embedding shape: {prompt_embeddings.shape}")
        np.savez(prompt_path, embedding_list=prompt_embeddings, label_list=label_list)
    else:
        logger.info(f"prompt embeddings already exists at {prompt_path}")

def encode_and_save_TC14(dataset, model, tokenizer, model_name):
    embedding_dir = os.path.join(f"./datasets/TextClassification/{dataset}", "embeddings")
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    
    if model_name in ["roberta_large_vanilla", "roberta_large_sentence"]:
        embedding_path = os.path.join(embedding_dir, f"{model_name}.npz")
        prompt_path = os.path.join(embedding_dir, f"prompt_0_{model_name}.npz")
        simcse = False
    else:
        embedding_path = os.path.join(embedding_dir, f"simcse_{model_name}.npz")
        prompt_path = os.path.join(embedding_dir, f"prompt_0_simcse_True_{model_name}.npz")
        simcse = True

    encode_label_names_TC14(dataset, model, tokenizer, model_name, prompt_path, simcse)
    encode_texts_and_labels_TC14(dataset, model, tokenizer, model_name, embedding_path, simcse)

def main():
    args = parse_args()
    model_name = 'roberta_large'
    dataset = args.dataset
    suffix_dict = {
        'agnews': "csv",
        'yahoo': "csv",
        'dbpedia': "txt",
        'imdb': "txt", 
        'amazon': "txt"
    }

    tokenizer = AutoTokenizer.from_pretrained(configs.model_names[model_name])
    model = AutoModel.from_pretrained(configs.model_names[model_name])
    logger.info(f"model name={configs.model_names[model_name]}")
    logger.info(f"dataset={dataset}")
    model.to(device)

    if dataset in configs.benchmarks:
        encode_and_save_5_benchmarks(dataset, suffix_dict[dataset], model, tokenizer, model_name)
    else:   # TC14
        encode_and_save_TC14(dataset, model, tokenizer, model_name)

if __name__ == "__main__":
    main()