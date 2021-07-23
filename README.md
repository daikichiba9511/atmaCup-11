# atma cup #11

this repository has code of [#11 [初心者歓迎! / 画像編] atmaCup](https://www.guruguru.science/competitions/17/)

this repository contains code which is used atmacup #11

we can't use pretrained weight to train DL model (e.g resnet, vit etc). many participants uses self-supervised leraning by using [lightly](https://github.com/lightly-ai/lightly) when training.

I also tried self-supervised learning when training Resnet18d , Vit.

I and many others tried Simsiam to get pretrained weight, but I guess that we almost cant do well, me too.

## Summary

I choosed two sub that are result of exp013 , result which is ensemble of exp013, exp016, exp024 as last submision

I tried below.

* resnet18d
* vit
* pretrained by self-supervised learning
    * simsiam (lightly & lightning bolts)
* custom head (MLP etc)

I think key point in this competetion is 'How long to let them train.'. e.g. when pretraining, epoch range is 800 ~ 1600 etc.

### direcotry

I do all experiments on colab-pro.

```shell
tree -L 2
.
├── eda
│   ├── eda001.py
│   ├── eda002.py
│   ├── eda003.py
├── exp
│   ├── exp000.py
│   ├── exp001.py
│   ├── # other exp is also like above.
│   ・・・・
│
├── input
│   └── atmacup-11 # data is under this repsiotry, images is in `image` dir under this directory
├── Makefile
├── output
│   ├── exp000
│   ├── # other exp results is also saved like above.
│   ・・・・
│
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   └── utils.py # this module to load my credentials, so not uploaded
```

## prerequire

```shell
# register name and email in git, and copy credentials under this repository like ssh credentials
make config

# install package to run exp files
make develop
```