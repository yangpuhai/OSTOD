# OSTOD

This code is the official pytorch implementation of our paper: **OSTOD: One-Step Task-Oriented Dialogue with Activated State and Retelling Response**

## Abstract
Currently, with the advancement of large-scale pre-trained language models, the potential of conversational AI is gradually being realized, resulting in rapid development in related research. In particular, task-oriented dialogue systems have garnered significant attention due to their enormous potential for assisting people in achieving various goals, such as booking hotels, making restaurant reservations, and purchasing train tickets. Traditionally, task-oriented dialogue systems have been decomposed into a multi-step system consisting of four steps: spoken language understanding, dialogue state tracking, dialogue policy learning, and natural language generation. Recently, with the help of large-scale pre-trained language models, end-to-end task-oriented dialogue systems have integrated these steps into a single model, enabling joint optimization and avoiding error propagation. However, almost all previous end-to-end methods inevitably require predicting the intermediate result of the dialogue state for interaction with the database, and the dialogue state is often domain or task specific, which poses a significant challenge for generalization in previous task-oriented dialogue systems. To solve the problem, in this paper, we propose One-Step Task-Oriented Dialogue (OSTOD), which models task-oriented dialogue by synchronously generating activated states and retelling responses, where activated states refer to slot values that contribute to database access, and retelling responses are system responses that contain activated state information. Specifically, first, automatic methods are designed to build data containing activated states and retelling responses. Then, a joint generation model that synchronously predicts activated states and retelling responses in a single step is proposed for task-oriented dialogue modeling. Empirical results show that, on MultiWOZ 2.0 and MultiWOZ 2.1 datasets, our OSTOD model is comparable to the state-of-the-art baselines, but also has excellent generalization ability in few-shot learning and domain transfer. 

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
