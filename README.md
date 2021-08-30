# Precomputation of Image Features for the Classification of Dynamic Properties in Waves

Repositiory for code to reproduce the results of the pre-print

[Precomputation of Image Features for the Classification of Dynamic Properties in Waves](#precomputation-of-image-features-for-the-classification-of-dynamic-properties-in-waves)

## Contents

Precomputation of Image Features for the Classification of Dynamic Properties in Waves

- [1. Environment](#1-environment)
- [2. Data](#2-data)
    - [2.1. Sources](#21-sources)
    - [2.2. Folder Structure](#22-folder-structure)
- [3. Models](#3-models)
- [4. Training](#4-training)
- [5. Evaluation](#5-evaluation)
    - [5.1. Evaluating](#51-evaluating)
    - [5.2. Results](#52-results)

## 1. Environment

### Using [`conda`](https://docs.conda.io/en/latest/)

```bash
conda env create -f environment.yml
conda activate dynamictexture
```

## 2. Data

### 2.1. Sources

Data is available to download from the [IR_Waveclass](https://github.com/dbuscombe-usgs/IR_waveclass) repository on github with the original train/test split. It is also supplied with this repositiory and split using the method described in our respective paper. ~~The bash script is also provided for grouping together waves.  We then hand split the waves into train and test groups to get the desired split ratio.~~

### 2.2. Folder Structure
<pre>
<details>
    <summary>conf : the config files for each model, each comes with the following parameters</summary>
    - model name of pretrained CNN to be used
    - weights to be used on the CNN model
    - include top - to use the fully connected layer on the CNN (set to false)
    - train path of images for training
    - test path of images for testing
    - features path where to save features extracted from train images
    - lables path where to save labels of train images
    - test features where to save features of test images
    - test labels where to save labels of test images
    - results where to results report
    - classifier path where to save the classifier model
    - imaug whether to use image augmentation
    - extraction func single, two IR or IR and OF image inputs
</details>
<details>
    <summary>data</summary>
</details>
<details>
    <summary>figures</summary>
</details>
<details>
    <summary>notebook out</summary>
</details>
<details>
    <summary>out</summary>
</details>
</pre>

## 3. Models

## 4. Training
```bash
python extract_CNN_features.py -c CONFIG_FILE
python train_model.py -c CONFIG_FILE
```

```bash
run_all_configs
```

## 5. Evaluation

### 5.1. Evaluating
```bash
python test.py -c CONFIG_FILE
```

```bash
python test.py -c CONFIG_FILE -e n
```
### 5.2. Results
