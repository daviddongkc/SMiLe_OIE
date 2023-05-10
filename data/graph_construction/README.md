# Dataset conversion
## This is used to construct word-level syntactic graphs (constituency and dependency graphs) as needed by SMiLe-OIE.

The original LSOIE dataset can be downloaded from [LSOIE](https://github.com/Jacobsolawetz/large-scale-oie).
You can use the following command to create the train/test dataset from the original LSOIE dataset:
```
bash dataset_create.sh
```
The train/test dataset will be saved in ```SMiLe-OIE\data```.
Note that the process will depend on spaCy(en_core_web_trf) and StanfordCoreNLP. Please install the dependency prior to the dataset creation.