import torch
import torch.nn as nn   
import torchvision.models as models
import torch.optim as optim
from tqdm.auto import tqdm
class Trainer:
    def __init__(self, model, dataloader, device, criterion, optimizer, num_epochs=10):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
    
    def train_model(self):
        """Train the model."""
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_corrects = 0
            for i, data in enumerate(self.dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #zero the parameter gradients
                self.optimizer.zero_grad()
                #forward
                outputs = self.model(inputs)
                #backward + optimize
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                #statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)
                
            epoch_loss = running_loss / len(self.dataloader.dataset)
            epoch_acc = running_corrects.double() / len(self.dataloader.dataset)

            print(f'Epoch {epoch}/{self.num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        return self.model

class VIT:
    """Vision Transformer (ViT) backbone."""
    
    def __init__(self, device, num_classes=5):
        self.model = models.vit_b_16(weights='DEFAULT')
        self.num_classes = num_classes
        self.device = device
        
    def feature_extractor(self, dataloader):
        """Extract features from the input image."""
        self.model.to(self.device)
        extracted_features = []
        labels = []
        with torch.inference_mode():
            self.model.eval()
            for X, y in tqdm(dataloader):
                X = X.to(self.device)
                features = self.model(X)
                extracted_features.append(features.cpu())
                labels.append(y)
        return torch.cat(extracted_features).squeeze(), torch.cat(labels).squeeze()
    
    def forward(self, x):
        return self.model(x)
    
    def finetune(self, dataloader, full_finetune=False, num_epochs=10):
        """Fine-tune the model."""
        num_features = self.model.heads.head.out_features
        add_fc = nn.Linear(num_features, self.num_classes)
        self.model.heads.head = nn.Sequential(
            self.model.heads.head, #existing last layer
            nn.ReLU(),
            add_fc #new last layer
        )
        # decide to train all layers or just the last layer
        if full_finetune:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.heads.head.parameters():
                param.requires_grad = True  
        #train the model     
        criterion = nn.CrossEntropyLoss()
        if full_finetune:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            optimizer = optim.Adam([
                {"params": self.model.heads.head.parameters()},  # Parameters of the existing last fully connected layer
            ],
            lr=0.001)     
        trainer = Trainer(self.model, dataloader, self.device, criterion, optimizer, num_epochs)
        self.model = trainer.train_model()
        return self.model