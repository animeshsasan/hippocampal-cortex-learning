
from training.loss import orthogonality_loss
import torch.nn as nn
import torch.optim as optim
from metrics.accuracy import get_accuracy
import torch

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
    
    def _get_data_index(self, epoch, random = True):
        position_in_pass = epoch % self.train_size
        
        if not random:
            return position_in_pass
        
        # Shuffle indices at the start of each new pass
        if position_in_pass == 0:
            self.shuffled_indices = torch.randperm(self.train_size)
        
        n_data = self.shuffled_indices[position_in_pass]
        return n_data

    def train(self, 
              model, 
              criterion, 
              optimizer, 
              epochs=100, 
              print_msg=False, 
              ortho_lambda = None, 
              random_sequencing = True, 
              batch_train = False,
              return_acts = False):
        loss_over_epochs = []
        epoch_number = []
        train_accuracy_over_epochs = []
        test_accuracy_over_epochs = []
        train_activations = {}
        test_activations = {}
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            if batch_train:
                outputs, activations = self.forward_batch(model, ortho_lambda)
                loss = criterion(outputs, self.y_train)
            else:
                n_data = self._get_data_index(epoch, random_sequencing)
                outputs, activations = self.forward_single(model, ortho_lambda, n_data)
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
                train_accuracy_over_epochs.append(self.get_train_accuracy(model))
                test_accuracy_over_epochs.append(self.get_test_accuracy(model))
                if return_acts:
                    with torch.no_grad():
                        _, activationsTest = model.forward(self.X_val, return_acts = True)
                        _, activationsTrain = model.forward(self.X_train, return_acts = True)
                    train_activations[epoch] = activationsTrain
                    test_activations[epoch] = activationsTest
        return loss_over_epochs, train_accuracy_over_epochs, epoch_number, test_accuracy_over_epochs, train_activations, test_activations
    
    def forward_single(self, model, get_activations, n_data):
        activations = None
        if get_activations:
            outputs, activations = model(self.X_train[n_data].unsqueeze(0), return_acts = True)
        else:
            outputs = model(self.X_train[n_data].unsqueeze(0))
        return outputs, activations
    
    def forward_batch(self, model, get_activations):
        activations = None
        if get_activations:
            outputs, activations = model(self.X_train, return_acts = True)
        else:
            outputs = model(self.X_train)
        return outputs, activations
    
    def set_data_granularity(self, granularity):
        self.granularity = granularity
    
    def get_test_accuracy(self, model):
        return get_accuracy(model, self.X_val, self.y_val)*100
    
    def get_train_accuracy(self, model):
        return get_accuracy(model, self.X_train, self.y_train)*100
    
    def get_mismatched_indices(self, model):
        with torch.no_grad():
            outputs = model(self.X_val)
            predicted = torch.argmax(outputs, 1)
            mismatched_indices = (predicted != self.y_val).nonzero(as_tuple=True)[0]
        return mismatched_indices

    def train_and_evaluate(self, 
                           model,  
                           name = "Model", 
                           epochs=100, 
                           ortho_lambda = None, 
                           print_msg = False, 
                           batch_train = False,
                           random_sequencing = True,
                           return_train_acts = False
                           ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_over_epochs, \
            accuracy_over_epochs, \
                epoch_number, \
                    test_accuracy_over_epochs, \
                        train_activations, \
                            test_activations \
                                = self.train(model, 
                                       criterion, 
                                       optimizer, 
                                       epochs=epochs, 
                                       ortho_lambda=ortho_lambda, 
                                       print_msg=print_msg, 
                                       random_sequencing=random_sequencing, 
                                       batch_train=batch_train,
                                       return_acts=return_train_acts)
        
        train_accuracy = get_accuracy(model, self.X_train, self.y_train)*100
        val_accuracy = get_accuracy(model, self.X_val, self.y_val)*100

        if print_msg:
            print(f"{name} Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        return loss_over_epochs, accuracy_over_epochs, epoch_number, train_accuracy, val_accuracy, test_accuracy_over_epochs, train_activations, test_activations