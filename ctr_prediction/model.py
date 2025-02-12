import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from huggingface_hub._snapshot_download import snapshot_download
from transformers import BertTokenizer, BertModel, AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
from PIL import Image
import requests
import pickle
import csv
import os
import argparse
from io import BytesIO

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")

# load pretrained model

# Chinese text encoder: RoBERTa-wwm-ext
model_name = 'hfl/chinese-roberta-wwm-ext'
revision = None
roberta_url = snapshot_download(model_name, revision=revision, local_files_only=True)
chinese_tokenizer = BertTokenizer.from_pretrained(roberta_url)
chinese_bert_model = BertModel.from_pretrained(roberta_url)


model_name = 'openai/clip-vit-base-patch32'
revision = None
clip_url = snapshot_download(model_name, revision=revision, local_files_only=True)

clip_model = CLIPModel.from_pretrained(clip_url)
image_processor = CLIPProcessor.from_pretrained(clip_url)

def get_chinese_text_embeddings(batch_text, expected_dim=768):
    chinese_bert_model.to(device)
    chinese_bert_model.eval()

    # Tokenize the batch of text
    inputs = chinese_tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the appropriate device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = chinese_bert_model(**inputs)
    
    # Calculate the cls embedding of the last hidden state
    embeddings = outputs.last_hidden_state[:, 0, :]
    
    # Apply padding for each embeddings in the batch as needed
    padded_embeddings = []
    for embedding in embeddings:
        padding_size = expected_dim - embedding.size(0)
        if padding_size > 0:
            padding = torch.zeros(padding_size, device=device)
            embedding = torch.cat([embedding, padding])
        padded_embeddings.append(embedding)
    
    # Stack all padded embeddings to form a batch tensor
    batch_embeddings = torch.stack(padded_embeddings)
    
    return batch_embeddings

def get_image_embeddings(image_urls, expected_dim=512):
    clip_model.to(device)
    clip_model.eval()

    # List for storing processed images
    processed_images = []

    # Download and preprocess each image
    for url in image_urls:
        image = requests.get(url)
        tempIm = BytesIO(image.content)
        img = Image.open(tempIm).convert('RGB')
        processed_images.append(img)

    # Process all images into tensors
    inputs = image_processor(images=processed_images, return_tensors="pt", padding=True)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Forward pass through the model
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)

    # Apply padding to ensure each embedding meets the expected dimension size
    padded_embeddings = []
    for embedding in embeddings:
        padding_size = expected_dim - embedding.size(0)  # Assuming embedding is 1D
        if padding_size > 0:
            padding = torch.zeros(padding_size, device=device)
            embedding = torch.cat([embedding, padding])
        padded_embeddings.append(embedding)

    # Stack all padded embeddings to form a batch tensor
    batch_embeddings = torch.stack(padded_embeddings)

    return batch_embeddings


class CTRDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.text_columns = ["item_name", "item_category", "shop_name", "shop_main_category_name", "bu_flag", "city_name"]
        self.image_column = "vertical_image_url"
        self.labels = torch.tensor(dataframe["ctr"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = [self.data.iloc[idx][col] for col in self.text_columns]
        image = self.data.iloc[idx][self.image_column]
        label = self.labels[idx]
        item = text + [image]

        # Return raw text and image URL, they will be processed in the model
        return item, label

def get_dataloader(data, batch_size, address, seed):
    # Create Dataset and DataLoader
    dataset = CTRDataset(data)

    # Define lengths for each dataset split based on dataset size
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = total_size - train_size - valid_size  # remainder to test size

    # Use a generator to ensure repeatable splits
    generator = torch.Generator().manual_seed(seed)

    # Split the dataset into train, validation, and test sets
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size], generator=generator)

    train_indices = train_dataset.indices
    valid_indices = valid_dataset.indices
    test_indices = test_dataset.indices
    
    with open(address, 'wb') as f:
        pickle.dump((train_indices, valid_indices, test_indices), f)

    # Create DataLoaders for each set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, valid_dataloader, test_dataloader

class CTRPredictionModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, hidden_dim, num_head, num_tranformer_layers, device):
        super(CTRPredictionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_tranformer_layers)

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * num_embeddings, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1),
            nn.Sigmoid()
        )

    def forward(self, text_image):
        text = text_image[:6]
        image = text_image[6]

        # Create text embeddings
        text_embs = [get_chinese_text_embeddings(col, expected_dim=self.embedding_dim).to(self.device) for col in text]

        # Stack embeddings; shape: [len(text), batch_size, embedding_dim]
        text_embs = torch.stack(text_embs, dim=0)
        text_embs = text_embs.permute(1, 0, 2)  # Shape: [batch_size, seq_length, embedding_dim]

        # Process image to get embeddings
        image_embs = get_image_embeddings(image, expected_dim=self.embedding_dim).to(self.device)
        if image_embs is None:
            return None

        # Add a sequence dimension for image: originally might be [batch_size, embedding_dim]
        image_embs = image_embs.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        # Concatenate text embeddings and image embeddings; shape: [batch_size, seq_length + 1, embedding_dim]
        combined_emb = torch.cat((text_embs, image_embs), dim=1)

        # Pass combined embeddings through the transformer; expect [batch_size, seq_length + 1, embedding_dim]
        combined_emb_t = self.transformer(combined_emb)
        
        # Flatten the output for the fully connected layers
        combined_flattened = combined_emb_t.reshape(combined_emb_t.size(0), -1)  # [batch_size, seq_length+1 * embedding_dim]

        # Forward through fully connected layers
        output = self.fc(combined_flattened)
        return output


def train_model(model, train_dataloader, criterion, optimizer, device, model_save_path):
    model.train()
    total_train_loss = []

    # Training loop
    for X_batch, y_batch in tqdm(train_dataloader, desc=f"Training"):
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_train_loss.append(loss.item())
    
    avg_train_loss = np.mean(total_train_loss)
    print(f"Training Loss: {avg_train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return avg_train_loss

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_eval_loss = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient calculation
        for X_batch, y_batch in tqdm(dataloader, desc="Evaluating"):
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            total_eval_loss.append(loss.item())

            # Collect predictions and targets for reporting metrics later
            all_predictions.append(outputs.cpu())
            all_targets.append(y_batch.cpu())

    avg_eval_loss = np.mean(total_eval_loss)
    
    # Flatten lists to calculate additional metrics if necessary
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    print(f"Test Loss: {avg_eval_loss:.4f}")

    # Additional Metrics can be calculated here if needed

    return avg_eval_loss

def load_trained_model(device, model_save_path):
    hidden_dim = (1024, 256)
    num_tranformer_layers = 4
    num_head = 4

    # Initialize the model architecture
    ctr_model = CTRPredictionModel(
        embedding_dim = 768, 
        num_embeddings = 7, 
        hidden_dim = hidden_dim, 
        num_head = num_head, 
        num_tranformer_layers = num_tranformer_layers, 
        device = device)
    ctr_model.to(device)
    
    # Load the saved state dictionary
    model_state_dict = torch.load(model_save_path, map_location=device)
    ctr_model.load_state_dict(model_state_dict)

    return ctr_model

def load_splits(data, batch_size, file_path):
    dataset = CTRDataset(data)
    with open(file_path, 'rb') as f:
        train_indices, valid_indices, test_indices = pickle.load(f)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader