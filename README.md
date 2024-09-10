# Vision Transformer FineTuned Classifier
[google colab](https://colab.research.google.com/drive/19ySYPRIMFlyPO0hrEfOmucgz_xbM0YOD?authuser=1#scrollTo=6xZnCvBEHcbo)

## Overview 
Used pretrained VIT model from pytorch, the VIT model is used as a feature extractor. In the code you can either use pretrained model or alose finetune using the train data and then test it on test data.

## DataFolder
The format for data will be `classname_train` for train data and `classname_test` for test data. Inside the data folder keep the folder structure. You can use you own custom data here with the following format and it will take the images and associate labels.


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

 