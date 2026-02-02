# DMG: Disentangled Multiplex Graph Representation Learning 

This repository contains the reference code for the manuscript ``Disentangled Multiplex Graph Representation Learning" 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)
0. [Testing](#test)

## Installation
* pip install -r requirements.txt 
* Download the datasets
* Download the trained models

## Preparation
Important args:
* `--use_pretrain` Test checkpoints in [checkpoints](checkpoints)
* `--dataset` ukb hcp-d hcp-ya hcp-a abide
* `--custom_key` Node: node classification

## Training
1. python prepare_data.py 
2. python main.py

## Testing
use_pretrain == 'True'
