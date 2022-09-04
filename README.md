# SimPTC

This is a codebase to perform zero-shot text classification with SimPTC. 

## Installation

The easiest way to install the code is to create a fresh anaconda environment:

```bash
conda create -n simptc python=3.7
source activate simptc
pip install -r requirements.txt
```

## Download the data

All the datasets we used are publicly available text classification datasets. One can download all datasets using the following anonymous link: https://drive.google.com/file/d/1MDO8LDRUNC-T8OCtHDk8yfuSnjo84EMX/view?usp=sharing

- One should put the `datasets` and `embeddings` folders at the same directory of the codes.
- `datasets` contains texts and labels for all datasets.
- `embeddings`contains the [ConcepNet NumberBatch](https://github.com/commonsense/conceptnet-numberbatch) embeddings used for expanding class names.

We provide more data source information below.

### 5 benchmark datasets

AG's News, DBpedia, Yahoo, IMDB, Amazon are 5 benchmark datasets commonly used to evaluate zero-shot text classification methods. We collect the datasets from: https://github.com/thunlp/OpenPrompt/blob/main/datasets/download_text_classification.sh

### TC14

To study the performance of SimPTC on a wider range of text classification tasks we did a literature search in zero-shot text classification and collected datasets that best fit our text classification setting with label names that have class-info. Here are the resources of the datasets in TC14.

20 News: https://huggingface.co/datasets/SetFit/20_newsgroups

NYT topic and NTY location: https://github.com/yumeng5/CatE/tree/master/datasets/nyt

BBC News: https://huggingface.co/datasets/SetFit/bbc-news

Yelp: https://huggingface.co/datasets/yelp_polarity

Emotion: https://huggingface.co/datasets/emotion

Banking77: https://huggingface.co/datasets/banking77

SST-2: https://huggingface.co/datasets/gpt3mix/sst2

SST-5: https://huggingface.co/datasets/SetFit/sst5

MPQA: https://github.com/princeton-nlp/LM-BFF/blob/main/data/download_dataset.sh

Subj: https://huggingface.co/datasets/SetFit/subj

TREC: https://huggingface.co/datasets/trec

Biomedical: https://github.com/rashadulrakib/short-text-clustering-enhancement/tree/master/data

StackOverflow: https://github.com/rashadulrakib/short-text-clustering-enhancement/tree/master/data

## Reproducing our results

After downloading and unzipping the two folders under the code directory, one can reproduce our result. 

We have already include the expanded class names and encoded datasets for IMDb, one can reproduce the results on IMDb with template #0 by

```bash
python run.py --dataset imdb --template 0
```
Averaging the final results of template 0 to 3 gives the number in Table 1 of the paper.

To reproduce the results on other datasets:

1. Expand class names using ConceptNet (**can be skipped** as we have already include the expanded class names in the downloaded datasets):

```bash
python gen_related_words.py --dataset agnews
```

2. Encode texts and class anchor texts (GPU required) using [SimCSE](https://github.com/princeton-nlp/SimCSE) 

```bash
CUDA_VISIBLE_DEVICES=0 python encode_dataset.py --dataset agnews
```

3. Run SimPTC.

   1. To reproduce the results on AG's News, DBpedia, Yahoo, IMDB, Amazon: (Averaging the final results of template 0 to 3 gives the number in Table 1 of the paper.)

   ```bash
   python run.py --dataset agnews --template 0
   ```

   2. To reproduce the results on TC14 (we only use the naive template "<mask>" on TC14, running the following code gives the numbers in Figure 4)

   ```bash
   python run.py --dataset sst2
   ```

