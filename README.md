# Syntactic Multi-view Learning for Open Information Extraction

## Introduction
SMiLe-OIE is the first neural OpenIE system that incorporates heterogeneous syntactic information through GCN encoders and multi-view learning. The details of this work are elaborated in our paper published in Main Conference of [EMNLP 2022](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.272.pdf).

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
@article{https://doi.org/10.48550/arxiv.2212.02068,
  doi = {10.48550/ARXIV.2212.02068},
  url = {https://arxiv.org/abs/2212.02068},
  author = {Dong, Kuicai and Sun, Aixin and Kim, Jung-Jae and Li, Xiaoli},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Syntactic Multi-view Learning for Open Information Extraction},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}


```

## Contact
In case of any issues, please send a mail to
```kuicai001 (at) e (dot) ntu (dot) edu (dot) sg```