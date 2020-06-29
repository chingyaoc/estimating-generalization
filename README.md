# Estimating Generalization under Distribution Shifts via Domain-Invariant Representations

<p align='center'>
<img src='https://github.com/chingyaoc/risk-estimation-dip/blob/master/misc/fig1.png?raw=true' width='400'/>
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


#### Dataset

Download the MNIST-M dataset from [Google Drive](https://drive.google.com/open?id=1iij6oj3akjJtaVe9eV-6UnRPJSO4GpdH) and unzip it. 
```
mkdir dataset
cd dataset
tar -zvxf mnist_m.tar.gz
```

#### Estimate proxy risk

Pretrain the domain adversarial neural network (DANN) as check model
```
python pretrain.py
```

Approximate proxy risk
```
python proxy_err.py
```

## Embedding Complexity Tradeoff
<p align='center'>
<img src='https://github.com/chingyaoc/risk-estimation-dip/blob/master/misc/fig2.png?raw=true' width='800'/>
</p>

Train DANN with designated number of layers in the encoder
```
python pretrain.py --g_num = 2
```
