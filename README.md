# Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning

This is the official code release of the following paper: 

Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. [Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning](https://arxiv.org/abs/2104.10353). SIGIR 2021 Full Paper.

<img src="https://github.com/Lee-zix/RE-GCN/blob/master/img/regcn.png" alt="regcn_architecture" width="700" class="center">

## Quick Start

### Environment variables & dependencies
```
conda create -n regcn python=3.7

conda activate regcn

pip install -r requirement.txt
```

### Process data
First, unzip and unpack the data files 
```
tar -zxvf data-release.tar.gz
```
For the three ICEWS datasets `ICEWS18`, `ICEWS14`, `ICEWS05-15`, go into the dataset folder in the `./data` directory and run the following command to construct the static graph.
```
cd ./data/<dataset>
python ent2word.py
```

### Train models
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

1. Train models
```
cd src
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0
```

### Evaluate models
To generate the evaluation results of a pre-trained model, simply add the `--test` flag in the commands above. 

For example, the following command performs single-step inference and prints the evaluation results (with ground truth history).
```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test
```

The following command performs multi-step inference and prints the evaluation results (without ground truth history).
```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test --multi-step --topk 0
```


### Change the hyperparameters
To get the optimal result reported in the paper, change the hyperparameters and other experiment set up according to Section 5.1.4 in the paper (https://arxiv.org/abs/2104.10353). 


## Citation
If you find the resource in this repository helpful, please cite
```
@article{li2021temporal,
  title={Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning},
  author={Li, Zixuan and Jin, Xiaolong and Li, Wei and Guan, Saiping and Guo, Jiafeng and Shen, Huawei and Wang, Yuanzhuo and Cheng, Xueqi},
  booktitle={SIGIR},
  year={2021}
}
```
