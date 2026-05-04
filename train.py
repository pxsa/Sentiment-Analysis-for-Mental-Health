import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# ==================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
    

# ==================================================
# ==================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


# ==================================================
# ==================================================

def train_epoch():
    pass

def evaluate():
    pass

def train_model(model, train_loader, val_loader, criterion,
                optimizer, scheduler, device, num_epochs,
                early_stopping=None, model_name='model'):
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'*' * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save metrics
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}')
        print(f'Val Loss: {val_loss:.3f}')
        print(f'Val Precision: {val_precision:.3f} | Val Recall: {val_recall:.3f} | Val F1: {val_f1:.3f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f'{model_name}_best.pt')
            print(f'Best model saved with validation accuracy: {best_val_acc:.3f}')

        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss, model):
                print(f'Early stopping triggered at epoch {epoch+1}')
                model.load_state_dict(early_stopping.best_model_state)
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, train_accs, val_losses, val_accs
        
def plot_training_history():
    pass