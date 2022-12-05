# Syntactic Multi-view Learning for Open Information Extraction

## Introduction
SMiLe-OIE is the first neural OpenIE system that incorporates heterogeneous syntactic information through GCN encoders and multi-view learning. The details of this work are elaborated in our paper published in Main Conference of EMNLP 2022.

## Syntactic Graphs Construction
To be uploaded in 
```data\graph_construction```.

## SMiLe-OIE Model
### Installation Instructions
Credits given to [LSOIE](https://github.com/Jacobsolawetz/large-scale-oie), we build our system based on it.

Use a python-3.7 environment and install the dependencies using,
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
@inproceedings{dong-etal-2021-docoie,
    title = "{D}oc{OIE}: A Document-level Context-Aware Dataset for {O}pen{IE}",
    author = "Dong, Kuicai  and
      Yilin, Zhao  and
      Sun, Aixin  and
      Kim, Jung-Jae  and
      Li, Xiaoli",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.210",
    doi = "10.18653/v1/2021.findings-acl.210",
    pages = "2377--2389",
}

```

## Contact
In case of any issues, please send a mail to
```kuicai001 (at) e (dot) ntu (dot) edu (dot) sg```