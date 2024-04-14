# Syntactic Multi-view Learning for Open Information Extraction

## Introduction
SMiLe-OIE is the first neural OpenIE system that incorporates heterogeneous syntactic information through GCN encoders and multi-view learning. The details of this work are elaborated in our paper published in Main Conference of [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.272/).

## Syntactic Graphs Construction
To be uploaded soon in 
```data\graph_construction```.

## SMiLe-OIE Model
### Installation Instructions
Credits given to [LSOIE](https://github.com/Jacobsolawetz/large-scale-oie), we build our system based on it.

Use a python-3.7 environment and install the dependencies using (we will update the requirements file soon),
```
pip install -r requirements.txt
```

### Running the code

```
python allennlp_run.py --div 0.024 --c1 0.012 --c2 0.012 --config config/wiki_multi_view.json  --epoch 5 --batch 16  --model trained_model/multi_view
```
Arguments:
- config: configuration file containing all the parameters for the model
- model:  path of the directory where the model will be saved
- div:  the loss weightage for inter-node intra-view multi-view loss
- c1:  the loss weightage for intra-node inter-view multi-view loss
- c2:  the loss weightage for inter-node inter-view multi-view loss


## Citing
If you use this code in your research, please cite:

```
@inproceedings{dong-etal-2022-syntactic,
    title = "Syntactic Multi-view Learning for Open Information Extraction",
    author = "Dong, Kuicai  and
      Sun, Aixin  and
      Kim, Jung-Jae  and
      Li, Xiaoli",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.272",
    pages = "4072--4083",
    abstract = "Open Information Extraction (OpenIE) aims to extract relational tuples from open-domain sentences. Traditional rule-based or statistical models were developed based on syntactic structure of sentence, identified by syntactic parsers. However, previous neural OpenIE models under-explored the useful syntactic information. In this paper, we model both constituency and dependency trees into word-level graphs, and enable neural OpenIE to learn from the syntactic structures. To better fuse heterogeneous information from the two graphs, we adopt multi-view learning to capture multiple relationships from them. Finally, the finetuned constituency and dependency representations are aggregated with sentential semantic representations for tuple generation. Experiments show that both constituency and dependency information, and the multi-view learning are effective.",
}


```

## Contact
In case of any issues, please send a mail to
```kuicai001 (at) e (dot) ntu (dot) edu (dot) sg```
