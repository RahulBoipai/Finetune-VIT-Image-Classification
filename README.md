# Vision Transformer FineTune Classifier
[google colab](https://colab.research.google.com/drive/19ySYPRIMFlyPO0hrEfOmucgz_xbM0YOD?authuser=1#scrollTo=6xZnCvBEHcbo)

## Run
To run VIT use below command
```zsh
python run.py
```
You will be asked to chose: if you want to finetune VIT or use pretrain model only.
Enter y or n
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
 Next option ask if you would finetune all weights of add a hidden layer and freeze other layers.
 ```zsh
 Do you want to fully fine-tune the model? (y/n): y
 ```

 