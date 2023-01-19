# OSTOD

This code is the official pytorch implementation of our paper: **OSTOD: One Step Task-Oriented Dialogue with Activated State and Retelling Response**

## Abstract
Currently, task-oriented dialogue systems have gotten tons of attention due to their great potential in helping people accomplish goals such as booking hotels. In general, previous task-oriented dialogue was often deconstructed into a multi-step system, which consists of three subtasks: understanding user utterances, determining system actions and generating responses. However, the error propagation problem is unavoidable in a multi-step system, which severely limits the ability of task-oriented dialogue. To solve the problem, in this paper, we propose One Step Task-Oriented Dialogue (OSTOD), which models task-oriented dialogue by synchronously generating activated states and retelling responses. Specifically, first, we introduce the definitions and design automatic methods to build data that contains both activated states and retelling responses. Then, we propose a joint generation model to predict the activated state and retelling response in a single step. Empirical results show that, on MultiWOZ 2.0 and MultiWOZ 2.1 datasets, our OSTOD model is better than the state-of-the-art models, and OSTOD has demonstrate outstanding capabilities in few-shot learning and domain transfer.

## Requirements
* python >= 3.7
* pytorch >= 1.0

## Description
### 1. Folders
```data```: Data folder and contains data preprocessing

```T5_generator```: The generator to build data that contains retelling response

```T5_filter```: The filter to build data that contains retelling response

```OSTOD```: Our one step task-oriented dialogue model

### 2. Files
```domain_split_config.py```, ```domain_split_data_loader.py``` and ```domain_split_generate_and_filtering.py```: Retelling response generation and filtering using T5_generator and T5_filter

## Usage
### 1. Data preprocessing
In the ```data``` folder, run ```preprocess_from_ubar.py``` and ```preprocess_for_our.py``` in turn to preprocess the dataset.

### 2. Dataset construction
In the ```T5_generator``` folder, run ```T5_generate.py``` to train the generator model.

In the ```T5_filter``` folder, run ```T5_generate.py``` to train the filter model.

In the main folder, run ```domain_split_generate_and_filtering.py``` to build the data containing the retelling responses using T5_generator and T5_filter.

### 3. Model eveluation
In the ```OSTOD``` folder, run ```train.py``` to train or test our OSTOD model, then run ```eval.py``` to calculate all metrics.
