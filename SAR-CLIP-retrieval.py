import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import os
import numpy as np
import pandas as pd
import open_clip
from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify, dataloader_with_indices
from clip_benchmark.datasets.builder import get_dataset_collate_fn
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str,
        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
        help="Name of backbone. In open_clip.list_models() or hugging face transformers",
    )
    parser.add_argument(
        "--retrieval-csv-path",
        type=str,
        default=None,
        help="Path to retrieval CSV dataset",
    )
    parser.add_argument(
        "--sarclip-path",
        default=None,
        type=str,
        help="Path to sarclip weight",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers per GPU."
    )

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(f'[Unknown args]: {unknown}')
    return args

def get_model(args):
    CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained='openai',
        device=args.device,
        cache_dir='cache/weights/open_clip'
    )
    tokenize = open_clip.tokenize
    checkpoint = torch.load(args.sarclip_path, map_location="cuda")
    
    print("1111111111111------------", checkpoint.keys())
    msg = CLIP_model.load_state_dict(checkpoint['state_dict']) 

    return CLIP_model, preprocess_train, preprocess_val, preprocess_val, tokenize

class CsvDataset(Dataset):
    def __init__(self, csv_path, transforms):
        self.csv_data = pd.read_csv(csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        img_path = row['filepath']
        caption = row['caption']
        image = Image.open(img_path)
        image = self.transforms(image)
        return image, caption

# modified from clip_benchmark.metrics.zeroshot_retrieval
def retrieval_evaluation(args, model, preprocess, tokenize, recall_k_list=[1, 5, 10]):
    dataset = CsvDataset(
        args.retrieval_csv_path,
        preprocess
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=get_dataset_collate_fn('mscoco_captions')
    )
    n_batches = len(dataloader)

    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    for batch_images, batch_texts, inds in tqdm(dataloader, total=n_batches):
        batch_images = batch_images.to(args.device)
        batch_texts_image_index = [ind for ind, text in zip(inds, batch_texts)]
        batch_texts = tokenize(batch_texts).to(args.device)

        with torch.no_grad():
            batch_image_features = model.encode_image(batch_images)
            batch_text_features = model.encode_text(batch_texts)
            batch_images_emb = F.normalize(batch_image_features, dim=-1)
            batch_texts_emb = F.normalize(batch_text_features, dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])

    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    scores = texts_emb @ images_emb.t()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        metrics[f"retrieval-image2text-R@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, args.device, k=recall_k) > 0).float().mean().item() * 100

    for recall_k in recall_k_list:
        metrics[f"retrieval-text2image-R@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, args.device, k=recall_k) > 0).float().mean().item() * 100

    metrics[f"retrieval-mean-recall"] = np.mean(list(metrics.values()))

    for key, item in metrics.items():
        metrics[key] = round(float(item), 2)

    return metrics

if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda"
    model, preprocess_train, preprocess_val, preprocess_aug, tokenize = get_model(args)

    # Image-text retrieval
    all_metrics = {}
    metrics = {}
    retrieval_metrics = retrieval_evaluation(args, model, preprocess_aug, tokenize)
    metrics.update(retrieval_metrics)
    all_metrics.update(retrieval_metrics)

    for name, val in metrics.items():
        print(name, round(val, 2))
