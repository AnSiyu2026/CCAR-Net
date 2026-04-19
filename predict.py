import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


MODEL_PATH = "./checkpoint.pth"
IMG_INFER_DIR = "./crater_img"
CSV_DATA_PATH = "./lunar_crater_database_robbins.csv"
OUTPUT_DIR = "./result"

CONVNEXT_TYPE = "convnext_small"
IMG_SIZE = 224
BATCH_SIZE = 128
# MC Dropout Sampling Count
MC_SAMPLES = 20 

class CraterAgeConvNeXtRegressor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        base_model = models.convnext_small(pretrained=pretrained)
        feature_dim = 768

        self.backbone = base_model

        self.backbone.classifier = nn.Identity()
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, 1) 
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        features = self.global_avg_pool(features)
        out = self.regressor(features)
        
        return torch.sigmoid(out).view(-1) * 4.4

# ===================== 3. Core Utility Functions =====================

def enable_dropout(model):

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def load_model(model_path, convnext_type, device):
    print(f"Loading model from {model_path}...")
    model = CraterAgeConvNeXtRegressor(convnext_type=convnext_type, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def get_infer_transform(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def build_crater_metadata_dict(csv_path):
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    crater_dict = {}
    for _, row in df.iterrows():
        crater_dict[str(row['CRATER_ID'])] = {
            'LAT': row['LAT_CIRC_IMG'],
            'LON': row['LON_CIRC_IMG'],
            'DIAM': row['DIAM_CIRC_IMG']
        }
    return crater_dict

def infer_batch_uncertainty(model, img_dir, crater_dict, transform, device, batch_size, mc_samples=20):
    img_dir_path = Path(img_dir)
    all_imgs = list(img_dir_path.glob("*.png"))
    
    # Filter valid images based on CSV metadata
    valid_tasks = []
    for p in all_imgs:
        cid = p.stem
        if cid in crater_dict:
            valid_tasks.append((p, cid))
            
    total_imgs = len(valid_tasks)
    print(f"Starting Uncertainty Inference on {total_imgs} images (Samples={mc_samples})...")
    
    results = []
    
    model.eval()

    enable_dropout(model) 
    
    for i in range(0, total_imgs, batch_size):
        batch_tasks = valid_tasks[i : i + batch_size]
        batch_tensors = []
        batch_ids = []
        
        for p, cid in batch_tasks:
            try:
                img = Image.open(p).convert('RGB')
                batch_tensors.append(transform(img))
                batch_ids.append(cid)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                continue
        
        if not batch_tensors: continue
        
        input_tensor = torch.stack(batch_tensors).to(device) # [B, C, H, W]
        
        batch_predictions = []
        
        with torch.no_grad():
            for _ in range(mc_samples):
                preds = model(input_tensor) # [B]
                batch_predictions.append(preds.cpu().numpy())
        
        batch_predictions = np.stack(batch_predictions, axis=0)
        
        mean_preds = np.mean(batch_predictions, axis=0) # Final predicted age
        std_preds = np.std(batch_predictions, axis=0)   # Uncertainty score
        
        for j, cid in enumerate(batch_ids):
            meta = crater_dict[cid]
            mu = float(mean_preds[j])
            sigma = float(std_preds[j])
            
            lower_bound = max(0.0, mu - 1.96 * sigma)
            upper_bound = min(4.4, mu + 1.96 * sigma)
            
            results.append({
                'CRATER_ID': cid,
                'LAT': meta['LAT'],
                'LON': meta['LON'],
                'DIAM': meta['DIAM'],
                'AGE_MEAN': round(mu, 4),        # Predicted Mean Age
                'AGE_STD': round(sigma, 4),      # Uncertainty
                'CI_LOWER': round(lower_bound, 4), # CI Lower Bound
                'CI_UPPER': round(upper_bound, 4)  # CI Upper Bound
            })
            
        if (i + 1) % (batch_size * 2) == 0:
            print(f"Processed {min(i + batch_size, total_imgs)}/{total_imgs}")
            
    return results

if __name__ == "__main__":
    # Load Model
    model = load_model(MODEL_PATH, device)
    
    # Load Metadata
    try:
        meta_dict = build_crater_metadata_dict(CSV_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_DATA_PATH}")
        exit()
        
    tf = get_infer_transform(IMG_SIZE)
    
    # Run Inference
    final_results = infer_batch_uncertainty(
        model, IMG_INFER_DIR, meta_dict, tf, device, BATCH_SIZE, MC_SAMPLES
    )
    
    # Save Results
    if final_results:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(final_results)
        
        save_path = os.path.join(OUTPUT_DIR, "crater_ages_with_uncertainty.csv")
        df.to_csv(save_path, index=False)
        
        print(f"\nDone! Results saved to: {save_path}")
        print("Columns explained:")
        print("- AGE_MEAN: The predicted age (Ga).")
        print("- AGE_STD: The uncertainty score (higher = less confident).")
        print("- CI_LOWER / CI_UPPER: The 95% confidence interval range.")
    else:
        print("No valid results generated. Please check image paths and CSV IDs.")