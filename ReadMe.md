# Phonetic Word Similarity

A novel method to compare the phonetic similarity between words based on phonetic features.

Our paper: [Phonetic Word Embeddings](https://arxiv.org/abs/2109.14796)

- Table of content
  - [Preparing dataset and environment](#preparing-dataset-and-environment)
  - [Algorithm results](#algorithm-results)
  - [Train embedding](#train-embedding)
  - [Embedding results](#embedding-results)
  - [Docker](#docker)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## Preparing dataset and environment

### Downloading

Download [The CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) in the data directory.

```
wget -P data http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b
```

Download SOTA model vocab from [NLP for Hindi git repo](https://github.com/goru001/nlp-for-hindi).

```
wget -O data/hindi_lm_large.vocab https://drive.google.com/uc?export=download&id=1P6r8UBcegvVmr1kBDjqcYppmt_WgnbNt
```

### Preparing

Add missing words to cmu dictionary

```
cat data/cmudict-0.7b res/cmudict_missing_words >> data/cmudict-0.7b-with-vitz-nonce
```

Install all the dependencies.

```
pip install -r src/requirements.txt
```

Generate hindi dictionary from LM vocab

```
python src/preprocess/vocab2dict.py res/hindi_phones.csv data/hindi_lm_large.vocab data/dict_hindi
```

---

## Algorithm results

[results_method.ipynb](src/results_method.ipynb) contains results for the algorithm. The result includes:

Comparision between unigram, bigram, bigram with penalty and bigram with penalty & vowel weight.
![01](docs/img/01_methods.png)

How we obtained the penalty of 2.5.
![02](docs/img/02_penalty.png)

Comparision between [Vitz and Winkler (1973)](https://www.researchgate.net/publication/232418589_Predicting_the_Judged_Similarity_of_Sound_of_English_words), [Parrish's Embeddings (2017)](https://aaai.org/ocs/index.php/AIIDE/AIIDE17/paper/view/15879), and our methods (with and without vowel weights).

![03](docs/img/03_comparison.png)

^ The Parrish's Embeddings (PSSVec) results are generated from the author's provided git [code](https://github.com/aparrish/phonetic-similarity-vectors) using `numpy.seed(0)` in `generate.py`. We can not use author provided pretrained vectors because the dictionary used by them misses a word `BELATION` used in the `RELATION` dataset by [Vitz and Winkler (1973)](https://www.researchgate.net/publication/232418589_Predicting_the_Judged_Similarity_of_Sound_of_English_words).

The similarity vectors used by us for calculating PSSVec can be downloaded using

```
wget -O data/cmudict-0.7b-simvecs https://drive.google.com/uc?export=download&id=1gCvwI8ldxGM52vCoN70wUKmJfFMdapNl
```

---

## Train embedding

- [English Embedding](embedding_english/)
- [Hindi Embedding](embedding_hindi/)

Embedding scores can be re-generated using [src/embedding.py](src/embedding.py) by providing the learned embedding file and the output file.

```
python src/embedding.py data/cmudict-0.7b-simvecs res/PSSVec_results.csv
python src/embedding.py embedding_english/simvecs res/embedding_score.csv
```

^ These files are used to generate scores in the result section using [results_method.ipynb](../src/results_method.ipynb).

---

## Embedding results

![Embeddings](docs/img/04_embedding.png)

TSNE Plot for some English words

![TSNE_ENGLISH](docs/img/05_tsne_english.png)

TSNE Plot for some Hindi words

![TSNE_ENGLISH](docs/img/06_tsne_hindi.png)

Pun Dataset (see [docs/puns.md](docs/puns.md) for more details)

![Density](docs/img/08_puns_not_same.png)

---

## Docker

Docker supported for development and training.

### Building docker image

```
make build
```

### Running an interactive docker container.

```
make develop
```

This will give you a command prompt inside the docker. Current directory will be mounted at `/workspace`.
The container will be destroyed on exit but all the files and changes done in the directly will persist.

You can also start it with GPU support:

```
make develop_gpu
```

### Removing the image.

```
make clean
```

Remember this will not delete the base image. To clean the base image run:

```
make clean_base
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- Hat tip to anyone whose code was used
