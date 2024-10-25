import os
import torch
from torch_geometric.data import Dataset, DataLoader
from leukemia_gnn import LeukemiaGNN, preprocess_image, create_graph
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
from pathlib import Path

class LeukemiaDataset(Dataset):
    def __init__(self, root, image_paths, labels, transform=None):
        super().__init__(root, transform)
        self.image_paths = image_paths
        self.labels = labels
        
    def len(self):
        return len(self.image_paths)
    
    def get(self, idx):
        # Load and preprocess image
        features = preprocess_image(self.image_paths[idx])
        
        # Create graph
        edge_index, edge_weights = create_graph(features)
        
        # Create data object
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=torch.tensor([self.labels[idx]], dtype=torch.long)
        )
        
        return data

def load_dataset(data_dir):
    """
    Load dataset from directory structure:
    data_dir/
        ALL/
            image1.jpg
            image2.jpg
            ...
        AML/
            image1.jpg
            image2.jpg
            ...
    """
    image_paths = []
    labels = []
    
    # Load ALL images (label 0)
    all_images = glob.glob(os.path.join(data_dir, 'ALL', '*.[jp][pn][g]'))
    image_paths.extend(all_images)
    labels.extend([0] * len(all_images))
    
    # Load AML images (label 1)
    aml_images = glob.glob(os.path.join(data_dir, 'AML', '*.[jp][pn][g]'))
    image_paths.extend(aml_images)
    labels.extend([1] * len(aml_images))
    
    return image_paths, labels

def train_model(data_dir, num_epochs=100, batch_size=32, learning_rate=0.001):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels = load_dataset(data_dir)
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = LeukemiaDataset(root=None, image_paths=train_paths, labels=train_labels)
    val_dataset = LeukemiaDataset(root=None, image_paths=val_paths, labels=val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LeukemiaGNN(num_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                pred = output.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        val_acc = val_correct / val_total
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print("Saved new best model")
        
        print("-" * 50)
    
    return model

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/dataset"  # Update this path
    model = train_model(data_dir)