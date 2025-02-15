from transformers import BertTokenizer, BertModel, AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, AutoPipelineForInpainting
import chromadb
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import argparse

# Initialize the image processor and model
image_processor = AutoImageProcessor.from_pretrained(args.swin_url)
swin_model = AutoModelForImageClassification.from_pretrained(swin_url).to('cuda')

# Initialize the ChromaDB client and collection
client = chromadb.PersistentClient(path="")
collection = client.get_or_create_collection(name="")

def get_image_paths(folder_path):
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    metadatas = []
    file_names = []
    for ext in image_extensions:
        image_paths = glob.glob(os.path.join(folder_path, ext))
        for path in image_paths:
            metadata = {"source": os.path.abspath(path)}
            metadatas.append(metadata)
            file_name = os.path.splitext(os.path.basename(path))[0]
            file_names.append(file_name)
    return metadatas, file_names

def get_feature(metadatas, batch_size=32):
    result = []
    total_batches = len(metadatas) // batch_size if len(metadatas) % batch_size == 0 else len(metadatas) // batch_size + 1
    pbar = tqdm(total=total_batches, desc="Processing images")
    for i in range(0, len(metadatas), batch_size):
        batch_metadatas = metadatas[i:i + batch_size]
        images = []
        for metadata in batch_metadatas:
            url = metadata['source']
            img = Image.open(url)
            images.append(img)
        # Batch process the list of images
        inputs = image_processor(images=images, return_tensors="pt").to(swin_model.device)
        with torch.no_grad():
            outputs = swin_model.swin(**inputs)  # Use the backbone model (swin) directly
            features = outputs.pooler_output.cpu()
        result.extend(np.array(features))
        pbar.update(1)
    pbar.close()
    return result

# If you need to add new images to the retrieval library, you need to use .add
folder_path = ''
metadatas, ids = get_image_paths(folder_path)
collection.add(
    metadatas=metadatas,
    embeddings=get_feature(metadatas),
    ids=ids
)

from generator import *

# Initialize the inpainting pipeline
pipe = AutoPipelineForInpainting.from_pretrained(
    args.sd_pretrain, custom_pipeline=sd_pretrain
).to('cuda')

# Create ArgumentParser object
parser = argparse.ArgumentParser()

# Add positional arguments
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--ref_url', type=str)
parser.add_argument('--sd_pretrain', type=str)
parser.add_argument('--swin_url', type=str)

args = parser.parse_args()

metadatas, ids = get_image_paths(args.input_file)
for i in range(len(metadatas)):
    result = collection.query(
        query_embeddings=get_feature([metadatas[i]]),
        n_results=10,
    )['metadatas'][0]
    ref = []
    for name in result:
        ref.append(args.ref_url + name['source'].split('/')[-1])
    origin_image_path = metadatas[i]['source']
    file_name = origin_image_path.split('/')[-1].split('.')[0]
    for j in range(len(ref)):
        full_image_fusion_pipeline(origin_image_path, ref[j], pipe, f'{file_name}_{j}', args.output_path)


