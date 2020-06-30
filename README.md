# Estimating Generalization under Distribution Shifts via Domain-Invariant Representations

<p align='center'>
<img src='https://github.com/chingyaoc/estimating-generalization/blob/master/misc/fig1.png?raw=true' width='400'/>
</p>

When the test distribution differs from the training distribution, machine learning models can perform poorly and wrongly overestimate their performance. In this work, we aim to better estimate the model's performance under distribution shift, without supervision. To do so, we use a set of domain-invariant predictors as a proxy for the unknown, true target labels. The error of this performance estimation is bounded by the target risk of the proxy model. 

**Estimating Generalization under Distribution Shifts via Domain-Invariant Representations**
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/), 
[Antonio Torralba](http://web.mit.edu/torralba/www/), and
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>
In International Conference on Machine Learning (ICML), 2020.


## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- PIL


## Risk Estimation


### Dataset
We will examine our method on two datasets: **MNIST** (source) and **MNIST-M** (target) where we assume that the labels of MNIST-M are not acceesible while estimating. The goal is to estimate the generalization of models trained on MNIST on MNIST-M.

Download the MNIST-M dataset from [Google Drive](https://drive.google.com/open?id=1iij6oj3akjJtaVe9eV-6UnRPJSO4GpdH) and unzip it. 
```
mkdir dataset
cd dataset
tar -zvxf mnist_m.tar.gz
```

### Estimate proxy risk
The main idea of this work is to use domain adaptation models as a proxy to unknown labels. In particular, we first train a domain adversarial neural network (DANN) with the following command: 
```
python pretrain.py
```
After training, the check model will be saved as `checkpoints/model_check.pth`. Equipped with the pretrained check model, we can estimate the proxy risk of itself or other hypotheses by maximizing the disagreement (Algorithm 1 in the paper). 

Flags: 

 - `--model_path`: specify the path to candidate model.
 - `--check_model_path`: specify the path to pretrained check model.
 - `--eps`: constraint for the domain-invariant loss of check models.
 - `--lam`: Tradeoff parameter for maximizing disgreement.

#### Proxy risk for DANN
For instance, to estimate the proxy risk of the check model itself (DANN) with default setting, run
```
python proxy_risk.py --model_path checkpoints/model_check.pth --check_model_path checkpoints/model_check.pth
```
#### Proxy risk for supervised models
Next, we examine our method by estimating the proxy risk for non-adaptive models that are trained only on the source, i.e., standard supervised learning. Pretrain the supervised model on MNIST:
```
python suptrain.py
```
Estimate proxy risk:
```
python proxy_risk.py --model_path checkpoints/model_source.pth --check_model_path checkpoints/model_check.pth
```


## Embedding Complexity Tradeoff
<p align='center'>
<img src='https://github.com/chingyaoc/estimating-generalization/blob/master/misc/fig2.png?raw=true' width='800'/>
</p>

Train DANN with designated number of layers in the encoder
```
python pretrain.py --g_num = 2
```

