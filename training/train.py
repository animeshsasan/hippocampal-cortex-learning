
from training.loss import orthogonality_loss
import torch.nn as nn
import torch.optim as optim
from metrics.accuracy import get_accuracy

class TrainUtil():

    def __init__(self, X, y, split = None, X_val = None, y_val = None):
        if split is not None:
            self.X_train, self.y_train = X[:split], y[:split]
            self.X_val, self.y_val = X[split:], y[split:]
            self.train_size = split
        else:
            assert X_val is not None and y_val is not None, "Either provide a split or validation data"
            self.X_train, self.y_train = X, y
            self.X_val, self.y_val = X_val, y_val
            self.train_size = len(self.X_train)
        
        self.granularity = 10

    def train(self, model, criterion, optimizer, epochs=100, print_msg=False, ortho_lambda = None):
        loss_over_epochs = []
        epoch_number = []
        accuracy_over_epochs = []
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            n_data = epoch % self.train_size
            if ortho_lambda:
                outputs, activations = model(self.X_train[n_data].unsqueeze(0), return_acts = True)
            else:
                outputs = model(self.X_train[n_data].unsqueeze(0))
            
            loss = criterion(outputs, self.y_train[n_data].unsqueeze(0))
            
            if ortho_lambda:
                ortho_loss = 0
                for layer in activations:
                    ortho_loss += orthogonality_loss(activations[layer])
                loss = loss + ortho_lambda * ortho_loss

            loss.backward()
            optimizer.step()

            if (epoch) % self.granularity == 0:
                if print_msg:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                loss_over_epochs.append(loss.item())
                epoch_number.append(epoch)
                accuracy_over_epochs.append(get_accuracy(model, self.X_train, self.y_train)*100)
        return loss_over_epochs, accuracy_over_epochs, epoch_number
    
    def set_data_granularity(self, granularity):
        self.granularity = granularity



    def train_and_evaluate(self, model,  name = "Model", epochs=100, ortho_lambda = None, print_msg = False):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_over_epochs, accuracy_over_epochs, epoch_number = self.train(model, criterion, optimizer, epochs=epochs, ortho_lambda=ortho_lambda, print_msg=print_msg)
        train_accuracy = get_accuracy(model, self.X_train, self.y_train)*100
        val_accuracy = get_accuracy(model, self.X_val, self.y_val)*100

        if print_msg:
            print(f"{name} Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return loss_over_epochs, accuracy_over_epochs, epoch_number, train_accuracy, val_accuracy