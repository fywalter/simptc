TEMPLATES = {
    "agnews": [
    "The news is about <mask>.",
    "The news is related to <mask>.",
    "<mask> is the topic of the news.",
    "This week's news is about <mask>.",
    "<mask>",
    ],
    "dbpedia": [
    "The object is about <mask>.",
    "The object is related to <mask>.",
    "<mask> is the topic of the object.",
    "<mask> is the subject of the object.",
    "<mask>",
    ],
    "yahoo": [
    "The answer is about <mask>.",
    "The answer is related to <mask>.",
    "<mask> is the topic of the answer.",
    "<mask> is involved in the answer.",
    "<mask>",
    ],
    "imdb": [
    "A <mask> movie review.",
    "The movie review is <mask>.",
    "The reviewer found the movie <mask>.",
    "The movie is <mask>.",
    "<mask>",
    ],
    "amazon": [
    "A <mask> product review.",
    "The product review is <mask>.",
    "The reviewer found the product <mask>.",
    "The product is <mask>.",
    "<mask>",
    ]
}

benchmarks = ['agnews', 'imdb', 'amazon', 'dbpedia', 'yahoo']

max_iters = {
    'agnews': 100,
    'imdb': 150,
    'amazon': 100,
    'yahoo': 20,
    'dbpedia': 50,
}

model_names = {
    'roberta_base': "princeton-nlp/sup-simcse-roberta-base",
    'roberta_large': "princeton-nlp/sup-simcse-roberta-large",
    'bert_large': "princeton-nlp/sup-simcse-bert-large-uncased",
}

model_names_other = {
    'roberta_large_vanilla': "roberta-large",
    "roberta_large_sentence": "sentence-transformers/nli-roberta-large", 
}