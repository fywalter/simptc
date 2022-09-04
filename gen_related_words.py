import os
import numpy as np
from scipy.spatial import distance
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='imdb', help="dataset name"
    )
    args = parser.parse_args()
    return args

def get_labels(label_file_path="./datasets/agnews/label_names.txt"):
    labels = {}
    with open(label_file_path,'r') as fin:
        L = fin.readlines()
        for idx, line in enumerate(L):
            category_words = line.strip().replace(","," ").split()
            labels[idx] = category_words
    return labels

def delete_duplicated_words(words):
    return list(set(words))

def delete_common_words(expanded_labels):
    labels_copy = expanded_labels.copy()
    for label_id in expanded_labels:
        old_words = set(expanded_labels[label_id])
        for label_id_comp in expanded_labels:
            if label_id_comp != label_id:
                comp_words = set(labels_copy[label_id_comp])
                old_words = old_words - comp_words
        expanded_labels[label_id] = list(old_words)
    return expanded_labels

embedding_path = "./embeddings/numberbatch-en-19.08.txt"

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    idx_2_word = []
    word_2_idx = {}
    word_vectors = []
    with open(path, encoding='utf-8', mode = 'r') as f:
        is_first = True
        for i, line in enumerate(f):
            if is_first:
                is_first=False
            else:
                word, vector = get_coefs(*line.strip().split(' '))
                idx_2_word.append(word)
                word_2_idx[word]=i-1
                word_vectors.append(vector)
    word_vectors = np.array(word_vectors)
    return idx_2_word, word_2_idx, word_vectors

def get_related_words_conceptnet(word, n_words, idx_2_word, word_2_idx, word_vectors):
    vector = word_vectors[word_2_idx[word]]
    distances = distance.cdist([vector], word_vectors, "cosine")[0]
    sorted_index = np.argsort(distances)
    word_idx_list = sorted_index[:n_words]
    related_words = []
    for idx in word_idx_list:
        related_words.append(idx_2_word[idx].replace("_", " "))
        # print(idx_2_word[idx], distances[idx])
    return related_words
    

def expand_labels_concetpnet(labels, word_number, idx_2_word=None, word_2_idx=None, word_vectors=None):
    expanded_labels = labels.copy()
    for label_id in labels:
        # print(label_id)
        old_words = expanded_labels[label_id]
        new_words = old_words.copy()
        for word in old_words:
            new_words = new_words + get_related_words_conceptnet(word, int(word_number/len(old_words)), idx_2_word, word_2_idx, word_vectors)
        new_words = delete_duplicated_words(new_words)
        expanded_labels[label_id]=new_words
    return delete_common_words(expanded_labels)

def gen_label_names_conceptnet(dataset, idx_2_word, word_2_idx, word_vectors):
    dataset_path = os.path.join("./datasets/TextClassification/", dataset)
    labels = get_labels(f"./datasets/TextClassification/{dataset}/label_names.txt")
    word_number_list = [1000]

    for word_number in word_number_list:
        out_path = os.path.join(dataset_path, f"label_names_conceptnet_{word_number}.txt")
        print(f"saved at {out_path}")
        related_words = expand_labels_concetpnet(labels, word_number, idx_2_word, word_2_idx, word_vectors)
        for key in related_words:
            print(f"Number of related words of class {key} : {len(related_words[key])}")
        with open(out_path, 'w', encoding='utf-8') as f:
            lines = [",".join(related_words[key]) for key in related_words]
            line = "\n".join(lines)
            f.write(line)

def main():
    args = parse_args()
    dataset = args.dataset
    print("loading concetpnet embeddings....")
    idx_2_word, word_2_idx, word_vectors = load_embeddings(embedding_path)
    print(f"expanding {dataset} label names...")
    print('with conceptnet')
    gen_label_names_conceptnet(dataset, idx_2_word, word_2_idx, word_vectors)

if __name__ == "__main__":
    main()