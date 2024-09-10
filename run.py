
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from backbone import VIT
from classifier import Classifier
from data_process import CreateDataset

from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def plot(X_test, y_test, id_label):
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=6).fit_transform(X_test)
    y = y_test.detach()
    label = [id_label[label.item()] for label in y]
    df = pd.DataFrame(dict(x=X_embedded[:,0], y=X_embedded[:,1], label= label))
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x="x", y="y", hue=df.label.tolist(),
                    palette=sns.color_palette("hls", 5),
                    data=df).set(title="Extracted feature T-SNE projection")
                    
    plt.savefig('tsne.png')

def main():
    finetune = input("Do you want to fine-tune the model? (y/n): ")
    if finetune.lower() == 'y':
        finetune = True
    else:
        finetune = False
        
    classifer = input("pick a classifier:-\n 1: KNN\n 2: SVM\n 3: RandomForest\n 4: Logistic regression\n Enter choice: ")
    if classifer == '1':
        model = 'KNN'
    elif classifer == '2':
        model = 'SVM'
    elif classifer == '3':
        model = 'RandomForest'
    elif classifer == '4':
        model = 'LogisticRegression'
    else:
        raise ValueError('Invalid choice')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    #Dataset
    print('Creating dataset...')
    train_dataset = CreateDataset(root_dir='./data', transform=transform, train=True)
    label_id, id_label = train_dataset.get_mapping() #train and test should have same mapping
    test_dataset = CreateDataset(root_dir='./data', transform=transform, train=False,
                                label_id=label_id, id_label=id_label)
    #DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    #Model
    print('Creating model...')
    vit = VIT(device=device)

    # if Fine-tuning the model
    if finetune:
        print('Fine-tuning the model...')
        full_tune = input("Do you want to fully fine-tune the model? (y/n): ")
        if full_tune.lower() == 'y':
            full_finetune = True
        else:
            full_finetune = False
        vit.finetune(train_dataloader, full_finetune=full_finetune, num_epochs=10)
    
    #Extract features from the model
    print('Extracting features...')
    X_train, y_train = vit.feature_extractor(train_dataloader)
    X_test, y_test = vit.feature_extractor(test_dataloader)
    #Classifier
    print('Creating classifier...')
    classifier = Classifier(model=model, k=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy, precision, recall, f1, cm = classifier.metric(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=classifier.model.classes_)
    disp.plot()
    plt.show()
    
    id_label = train_dataset.id_label
    plot(X_test, y_test, id_label)


if __name__ == '__main__':
    main()