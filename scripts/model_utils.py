import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from sklearn import neighbors

def train_knn_regressor(x_train: pd.Series, 
                        y_train: pd.Series, 
                        n_neighbors: int, 
                        weights: str)->neighbors.KNeighborsRegressor:
    """
    A function wrapper to the scikit-learn API that performs fit and predict
        
        Args:
            x_train (pd.Series): predictor data
            y_train (pd.Series): response variable
            n_neighbors (int): number of neighbors to use in KNN
            weights (str): uniform or distance

        Returns:
            A scikitlearn k nearest neightbour regressor fitted
    """
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    knn.fit(x_train, y_train)
    return knn


class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """
    Training data object for our nyc taxi data
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = torch.from_numpy(self._one_hot_X().toarray()) # potentially smarter ways to deal with sparse here
        self.y = torch.from_numpy(self.y_train.values)
        self.X_enc_shape = self.X.shape[-1]
        print(f"encoded shape is {self.X_enc_shape}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
        
    def _one_hot_X(self):
        return self.one_hot_encoder.fit_transform(self.X_train)
    
class MLP(nn.Module):
    """Multilayer Perceptron for regression. """
    def __init__(self, encoded_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
    
    def forward(self, x):
        return self.layers(x)