# LAGCN SRL Pointer
This repository includes the code of the Semantic Role Labeling (SRL) Parser with label-aware graph convolutional network (LAGCN) based pointer networks
of the AAAI 2021 paper: [Encoder-Decoder Based Unified Semantic Role Labeling with Label-Aware Syntax](https://ojs.aaai.org/index.php/AAAI/article/view/17514). 


-------------------

# Requirement Install
  

```bash
pip install -r requirements.txt
```

# Datasets

### Two popular dependency-based SRL datasets.
Download them and put at `./data` folds. 

- [CoNLL09](https://ufal.mff.cuni.cz/conll2009-st/train-dev-data.html)
- [UPB](https://universalpropositions.github.io/)


### Syntax annotation parsing

To prepare the syntactic dependency features, deploy the CoreNLP:

```bash
wget https://nlp.stanford.edu/software/stanford-corenlp-latest.zip

nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8083 -timeout 15000 > 1.log 2>&1 &
```

See the example data in [./data/demo](data%2Fdemo).


# Experiments

Step 1. To train the parser, you need to include the pre-trained word embeddings in the ``embs`` folder and run the following script:

```bash
./scripts/run_parser.sh <model> <data>
```

To evaluate the best trained model on the test set, just use the official script to compute the F1 scores:

```bash
./scripts/eval.sh <best epoch> <data> <model>
```


# Citation

```
@inproceedings{FeiGraphSynAAAI21,
  author    = {Hao Fei and Fei Li and Bobo Li and Donghong Ji},
  title     = {Encoder-Decoder Based Unified Semantic Role Labeling with Label-Aware Syntax},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {12794--12802},
  year      = {2021},
}
```


# License

The code is released under Apache License 2.0 for Noncommercial use only. 
