# Relation extraction with document level distant supervision on ChemDisGene dataset.

Create a new conda enviroment:

```
conda create -n chemdisgene python=3.7
conda activate chemdisgene
```

If you haven't already, clone the git repo:

```
git lfs clone https://github.com/chanzuckerberg/ChemDisGene.git
cd ChemDisGene/baseline/
```

Install all required dependencies by running the following in the root directory:

```
make install
```

Convert dataset from `../data` into the json format and save under `./data/` for dataloader:

```
python src/tsv2json.py ../data/ ./data/
```

---

Now it is ready for training. 

Modify proper enviroment variables `set_environment.sh`, and set up enviroment variable before training / testing:

```
source set_environment.sh
```

An example of training command to reimplement our paper's result:

```
python src/main.py --data_path data/ --learning_rate 1e-5 --mode train --encoder_type microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --model biaffine
```

log and results are saved under `saved_models/data/`

