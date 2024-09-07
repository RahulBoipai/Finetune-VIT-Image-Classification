
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from backbone import VIT, Trainer
from classifier import Classifier
from data_process import CreateDataset

from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(X_test, y_test, id_label):
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=6).fit_transform(X_test)
    y = torch.tensor(y_test).detach()
    label = [id_label[label.item()] for label in y]
    df = pd.DataFrame(dict(x=X_embedded[:,0], y=X_embedded[:,1], label= label))
    sns.scatterplot(x="x", y="y", hue=df.label.tolist(),
                    palette=sns.color_palette("hls", 5),
                    data=df).set(title="Extracted feature T-SNE projection")
    plt.show()

def main(finetune=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_dataset = CreateDataset(root_dir='./data', transform=transform, train=True)
    test_dataset = CreateDataset(root_dir='./data', transform=transform, train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    vit = VIT(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": vit.heads.head.parameters()},  # Parameters of the existing last fully connected layer
    ],
    lr=0.001,
    momentum=0.9)
    
    # if Fine-tuning the model
    vit.finetune(train_dataloader, full_finetune=False, criterion=criterion, optimizer=optimizer, num_epochs=10)
    
    #Extract features from the model
    features = vit.feature_extractor(test_dataloader)
    
    classifier = Classifier(model='KNN', k=3)
    classifier.fit(features, test_dataset.labels)
    y_pred = classifier.predict(features)
    accuracy, precision, recall, f1, cm = classifier.metric(test_dataset.labels, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=classifier.classes_)
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main(finetune=False)