# TimeVis
Official source code for IJCAI 2022 Paper: **Temporality Spatialization: A Scalable and Faithful Time-Travelling Visualization for Deep Classifier Training**

## Usage
### Download dependencies
Please run the following commands to install all dependencies:
```console
~$ pip install -r requirements.txt
```
### Run 
1. Prepare data
> 1.1 Save subject models and data following our format (see [here](https://github.com/xianglinyang/DeepVisualInsight/wiki/design#data-directory)).

> 1.2 Set training hyperparameters in ~/TimeVis/singleVis/config.json

2. Train a visualization model
```console
~$ python main.py ---content_path /path/to/subject_models --dataset dataset_name -g gpu_id
```
3. Evaluate visualization model
```console
~$ python test.py ---content_path /path/to/subject_models --dataset dataset_name -g gpu_id
```

### Training hyperparameters
| Hyperparameters | Meaning | Example |
|---|---|---|
|```Config Name``` |The config name for one training process|"cifar10"|
|```NET```|The subject model name to be called|"resnet18"|
|```TRAINING_LEN```|Training data len|50000|
|```TESTING_LEN```|Testing data len|10000|
|```LAMBDA```|The trade-off between umap loss and reconstruction loss. It depends on dataset.|10.|
|```L_BOUND```|The |.5|
|```MAX_HAUSDORFF```|r0||
|```ALPHA```|\alpha||
|```BETA```|\beta||
|```HIDDEN_LAYER```|The number of hidden layers for our visualization model.|3|
|```INIT_NUM```|||
|```EPOCH_START```|||
|```EPOCH_END```|||
|```EPOCH_PERIOD```|||
|```N_NEIGHBORS```|The |15|
|```MAX_EPOCH```|||
|```S_N_EPOCHS```|||
|```B_N_EPOCHS```|||
|```T_N_EPOCHS```|||
|```PATIENT```|Early stopping patient.|3|

## Reference

If you find our tool helpful, please cite the following paper:
```
@inproceedings{yang2022temporality,
  title={Temporality Spatialization: A Scalable and Faithful Time-Travelling Visualization for Deep Classifier Training},
  author={Yang, Xianglin and Lin, Yun and Liu, Ruofan and Dong, Jin Song},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  year={2022}
},
@inproceedings{yang2022deepvisualinsight,
  title={DeepVisualInsight: Time-Travelling Visualization for Spatio-Temporal Causality of Deep Classification Training},
  author={Yang, Xianglin and Lin, Yun and Liu, Ruofan and He, Zhenfeng and Wang, Chao and Dong, Jin Song and Mei, Hong},
  booktitle = {The Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
} 
```
