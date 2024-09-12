# Vision Transformer FineTuned Classifier
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19ySYPRIMFlyPO0hrEfOmucgz_xbM0YOD?authuser=1#scrollTo=6xZnCvBEHcbo)

## Overview 
Used pretrained VIT model from pytorch, the VIT model is used as a feature extractor. In the code you can either use pretrained model or finetune using the train data and then test it on test data. Then after extracting features from VIT (last hidden layer vector). Then use this as feature to train any standard classifer.

## Setup
Install miniconda to create virtual environment.
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

```
for more details regarding installing [miniconda on linux.](https://waylonwalker.com/install-miniconda/)

Create environment name `vit` from `requirements.txt`
```bash
conda create -n vit -r requirements.text
```
## DataFolder
The format for data shoulf be `classname_train` for train data and `classname_test` for test data. Inside the data folder keep the folder structure. You can use you own custom data here with the following format and it will take the images and associate labels.
For more understanding look at the data folder in repository.


## Run
To run VIT use below command
```zsh
python run.py
```
You will be asked to chose: if you want to finetune VIT or use pretrain model only.
Enter: y or n
```zsh
Do you want to fine-tune the model? (y/n): y
```
Next you will get option to pick a classifer.
select the classifier by entering number associated with it.
```zsh
pick a classifier:-
 1: KNN
 2: SVM
 3: RandomForest
 4: Logistic regression
 Enter choice: 1
 ```
 Next option ask if you would finetune all weights (y) or  add a hidden layer and freeze other layers (n).
 ```bash
 Do you want to fully fine-tune the model? (y/n): y
 ```

## Evaluation

### Pretrained
these are the best score: 
| Classifer | Acurracy |  F1-score |
| --------- | -------- | --------- | 
| K-Nearest Neighbors | 0.95 |0.95 |
| Support Vector Machine | 0.98 | 0.98|
| Random Forest | 0.98 |  0.98 | 
| Logistic Regression | 0.98 | 0.98 |

### FineTuned
| Classifer | Acurracy | F1-score |
| --------- | -------- |  ---------|
| K-Nearest Neighbors| 0.97 |  0.97 |
| Support Vector Machine | 0.99 |  0.99 |
| Randome Forest | 0.95 |  0.95 |
| Logistic Regression | 0.98 | 0.98 | 

### Full FineTuned
| Classifer | Acurracy |  F1-score |
| --------- | -------- | ---------|
| K-Nearest Neighbors | 0.96 | 0.96 |
| Support Vector Machine | 0.96 | 0.96 |
| Random Forest |0.94 | 0.94 | 
| Logistic Regression | 0.96 | 0.95 | 

### Quality of Feature space
