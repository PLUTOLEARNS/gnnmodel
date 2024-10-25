import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import torchvision.transforms as transforms

class LeukemiaGNN(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super(LeukemiaGNN, self).__init__()
        
        # Graph Convolutional Layers
        self.conv1 = ChebConv(num_features, 64, K=3)
        self.conv2 = ChebConv(64, 128, K=3)
        self.conv3 = ChebConv(128, 256, K=3)
        self.conv4 = ChebConv(256, 512, K=3)
        self.conv5 = ChebConv(512, 256, K=3)
        self.conv6 = ChebConv(256, 128, K=3)
        
        # Fully Connected Layer
        self.fc = nn.Linear(128 * 100, num_classes)  # 100 nodes * 128 features
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # Apply Graph Convolutions with ReLU activation
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        
        # Flatten and apply fully connected layer
        x = x.view(-1, 128 * 100)
        x = self.fc(x)
        
        # Apply softmax
        return F.softmax(x, dim=1)

def preprocess_image(image_path, num_segments=100):
    """
    Preprocess the blood smear image by segmenting it into regions
    and extracting features.
    """
    # Load and convert image to numpy array
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).numpy()[0]
    
    # Reshape image for clustering
    pixels = image.reshape(-1, 1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_segments, random_state=42)
    clusters = kmeans.fit_predict(pixels)
    clusters = clusters.reshape(image.shape)
    
    # Extract features for each segment
    features = []
    for i in range(num_segments):
        mask = (clusters == i)
        avg_intensity = np.mean(image[mask]) if mask.any() else 0  # Handle empty mask
        features.append([avg_intensity])
    
    return torch.FloatTensor(features)

def create_graph(features, k=5):
    """
    Create a graph from the extracted features using k-nearest neighbors.
    """
    # Calculate pairwise distances
    dist_matrix = torch.cdist(features, features)
    
    # Get k nearest neighbors for each node
    _, indices = torch.topk(dist_matrix, k + 1, largest=False)
    indices = indices[:, 1:]  # Remove self-loops
    
    # Create edge index
    rows = torch.arange(features.size(0)).view(-1, 1).repeat(1, k).view(-1)
    cols = indices.view(-1)
    edge_index = torch.stack([rows, cols])
    
    # Calculate edge weights (inverse of distances)
    edge_weights = 1.0 / (dist_matrix[rows, cols] + 1e-6)
    
    return edge_index, edge_weights
