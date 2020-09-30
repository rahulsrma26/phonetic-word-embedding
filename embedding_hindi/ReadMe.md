# Hindi Phonetic Embeddings

## Preparing environment

```
cd embedding_hindi
pip install -r requirements.txt
```

## Creating the cython package

Create it using the make.

```
make build
```

We can test whether it's build sucessfully or not.

```
make test
```

## Training Embedding

```
make train
```

This will generate the `simvecs_hindi` file. For running it manually see options by running

```
python train.py -h
```

You can also download the pre-trained vector embedding.

```
wget -O simvecs_hindi https://drive.google.com/uc?export=download&id=1nrcRiGpEW1KPIDkLF5c5LuHidhPNMu6b
```

## Results

To see the embedding results check [result.ipynb](result.ipynb)
