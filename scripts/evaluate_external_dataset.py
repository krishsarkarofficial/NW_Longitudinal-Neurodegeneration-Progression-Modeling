"""
Evaluation and testing script for external slice dataset in data/processed/Processed_output2.

Dataset structure:
  - AD_Images1: Alzheimer's Disease scans (3,067 slices)
  - CN_Images1: Cognitively Normal controls (4,528 slices)
  - MCI_Images1: Mild Cognitive Impairment scans (7,165 slices)
Total: 14,760 2D MRI slices (224x224 grayscale)
"""

import os
import glob
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from scipy import stats

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SliceDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, path

def run_evaluation(data_dir="data/processed/Processed_output2", output_dir="outputs/external_test"):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    classes = ["CN_Images1", "MCI_Images1", "AD_Images1"]
    class_names = ["Cognitively Normal (CN)", "Mild Cognitive Impairment (MCI)", "Alzheimer's Disease (AD)"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_paths = []
    all_labels = []
    class_counts = {}

    for c in classes:
        folder = os.path.join(data_dir, c)
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        class_counts[c] = len(paths)
        for p in paths:
            all_paths.append(p)
            all_labels.append(class_to_idx[c])

    total_images = len(all_paths)
    print(f"Total images loaded: {total_images}")
    for c, count in class_counts.items():
        print(f"  - {c}: {count} ({count/total_images*100:.2f}%)")

    # Image transforms
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. Morphometric / Biomarker Analysis across Classes
    print("\n--- Computing Morphometric & Slice Intensity Phenotypes ---")
    morph_records = []
    # Sample 500 images per class for detailed morphometrics
    rng = np.random.RandomState(42)
    for c_idx, c in enumerate(classes):
        paths = [all_paths[i] for i in range(total_images) if all_labels[i] == c_idx]
        sample_paths = rng.choice(paths, size=min(500, len(paths)), replace=False)
        for p in sample_paths:
            im_arr = np.array(Image.open(p), dtype=np.float32)
            fg_mask = im_arr > 25
            fg_ratio = float(fg_mask.mean())
            fg_intensity = float(im_arr[fg_mask].mean()) if fg_ratio > 0 else 0.0
            fg_std = float(im_arr[fg_mask].std()) if fg_ratio > 0 else 0.0
            
            # Cortical/Brain Parenchyma Area proxy
            y_coords, x_coords = np.nonzero(fg_mask)
            if len(y_coords) > 0:
                y_span = (y_coords.max() - y_coords.min()) / 224.0
                x_span = (x_coords.max() - x_coords.min()) / 224.0
                compactness = len(y_coords) / (y_span * x_span * 224 * 224 + 1e-6)
            else:
                y_span, x_span, compactness = 0, 0, 0
                
            # Gray-level gradient energy (surrogate for sulcal widening / tissue boundary sharp edges)
            grad_y = np.abs(np.diff(im_arr, axis=0)).mean()
            grad_x = np.abs(np.diff(im_arr, axis=1)).mean()
            
            morph_records.append({
                "class": c,
                "label": c_idx,
                "fg_ratio": fg_ratio,
                "fg_intensity": fg_intensity,
                "fg_std": fg_std,
                "compactness": compactness,
                "edge_energy": float(grad_y + grad_x),
            })

    morph_df = pd.DataFrame(morph_records)
    morph_summary = morph_df.groupby("class").agg({
        "fg_ratio": ["mean", "std"],
        "fg_intensity": ["mean", "std"],
        "compactness": ["mean", "std"],
        "edge_energy": ["mean", "std"],
    })
    morph_summary.to_csv(os.path.join(output_dir, "morphometric_summary.csv"))
    print(morph_summary)

    # Statistical significance of atrophy proxy (fg_ratio: brain volume fraction)
    cn_ratios = morph_df[morph_df["label"] == 0]["fg_ratio"]
    mci_ratios = morph_df[morph_df["label"] == 1]["fg_ratio"]
    ad_ratios = morph_df[morph_df["label"] == 2]["fg_ratio"]
    
    t_ad_cn, p_ad_cn = stats.ttest_ind(cn_ratios, ad_ratios, equal_var=False)
    t_mci_cn, p_mci_cn = stats.ttest_ind(cn_ratios, mci_ratios, equal_var=False)
    t_ad_mci, p_ad_mci = stats.ttest_ind(mci_ratios, ad_ratios, equal_var=False)
    print(f"\nAtrophy difference (Brain parenchyma fraction):")
    print(f"  CN vs AD:  t={t_ad_cn:.3f}, p={p_ad_cn:.3e}")
    print(f"  CN vs MCI: t={t_mci_cn:.3f}, p={p_mci_cn:.3e}")
    print(f"  MCI vs AD: t={t_ad_mci:.3f}, p={p_ad_mci:.3e}")

    # 2. Benchmark Deep Learning & Feature Representations
    print("\n--- Evaluating Feature Encodings & Diagnostic Classifiers ---")
    # Stratified test sample of 1,500 slices (500 per class) to evaluate pretrained representations
    test_indices = []
    for c_idx in range(3):
        indices = [i for i, l in enumerate(all_labels) if l == c_idx]
        chosen = rng.choice(indices, size=500, replace=False)
        test_indices.extend(chosen)

    test_paths = [all_paths[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]
    
    test_ds = SliceDataset(test_paths, test_labels, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load ResNet-18 pretrained feature extractor
    print("Extracting deep neuroimaging representations via ResNet-18...")
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    modules = list(resnet.children())[:-1]  # remove fc
    feat_extractor = nn.Sequential(*modules).to(device)
    feat_extractor.eval()

    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch_imgs, batch_lbls, _ in test_loader:
            batch_imgs = batch_imgs.to(device)
            feats = feat_extractor(batch_imgs).squeeze(-1).squeeze(-1)
            features_list.append(feats.cpu().numpy())
            labels_list.append(batch_lbls.numpy())

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    print(f"Extracted feature matrix shape: {X.shape}")

    # Train linear probing classifier on 50% split and test on remaining 50%
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr_accs, lr_f1s, lr_aucs = [], [], []
    rf_accs, rf_f1s, rf_aucs = [], [], []
    
    # Store predictions for the overall test fold
    all_y_true = []
    all_lr_pred = []
    all_lr_prob = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Logistic Regression
        clf_lr = LogisticRegression(max_iter=500, C=1.0)
        clf_lr.fit(X_tr, y_tr)
        preds_lr = clf_lr.predict(X_val)
        probs_lr = clf_lr.predict_proba(X_val)
        
        lr_accs.append(accuracy_score(y_val, preds_lr))
        lr_f1s.append(f1_score(y_val, preds_lr, average="macro"))
        lr_aucs.append(roc_auc_score(y_val, probs_lr, multi_class="ovr"))
        
        all_y_true.extend(y_val)
        all_lr_pred.extend(preds_lr)
        all_lr_prob.append(probs_lr)

        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_rf.fit(X_tr, y_tr)
        preds_rf = clf_rf.predict(X_val)
        probs_rf = clf_rf.predict_proba(X_val)
        rf_accs.append(accuracy_score(y_val, preds_rf))
        rf_f1s.append(f1_score(y_val, preds_rf, average="macro"))
        rf_aucs.append(roc_auc_score(y_val, probs_rf, multi_class="ovr"))

    all_lr_prob = np.vstack(all_lr_prob)
    all_y_true = np.array(all_y_true)
    all_lr_pred = np.array(all_lr_pred)

    conf_mat = confusion_matrix(all_y_true, all_lr_pred)
    class_report = classification_report(
        all_y_true, all_lr_pred, target_names=["CN", "MCI", "AD"], output_dict=True
    )
    
    # Binary classification benchmarks: CN vs AD, and CN vs MCI
    mask_cn_ad = np.isin(all_y_true, [0, 2])
    y_cn_ad = (all_y_true[mask_cn_ad] == 2).astype(int)
    prob_cn_ad = all_lr_prob[mask_cn_ad, 2] / (all_lr_prob[mask_cn_ad, 0] + all_lr_prob[mask_cn_ad, 2])
    pred_cn_ad = (prob_cn_ad > 0.5).astype(int)
    
    acc_cn_ad = accuracy_score(y_cn_ad, pred_cn_ad)
    f1_cn_ad = f1_score(y_cn_ad, pred_cn_ad)
    auc_cn_ad = roc_auc_score(y_cn_ad, prob_cn_ad)

    mask_cn_mci = np.isin(all_y_true, [0, 1])
    y_cn_mci = (all_y_true[mask_cn_mci] == 1).astype(int)
    prob_cn_mci = all_lr_prob[mask_cn_mci, 1] / (all_lr_prob[mask_cn_mci, 0] + all_lr_prob[mask_cn_mci, 1])
    pred_cn_mci = (prob_cn_mci > 0.5).astype(int)

    acc_cn_mci = accuracy_score(y_cn_mci, pred_cn_mci)
    f1_cn_mci = f1_score(y_cn_mci, pred_cn_mci)
    auc_cn_mci = roc_auc_score(y_cn_mci, prob_cn_mci)

    results = {
        "dataset_summary": {
            "total_images": total_images,
            "classes": class_counts,
            "image_dimensions": [224, 224],
            "channels": 1
        },
        "morphometric_phenotypes": {
            "parenchyma_fraction_mean": {
                "CN": float(morph_df[morph_df["label"] == 0]["fg_ratio"].mean()),
                "MCI": float(morph_df[morph_df["label"] == 1]["fg_ratio"].mean()),
                "AD": float(morph_df[morph_df["label"] == 2]["fg_ratio"].mean()),
            },
            "parenchyma_fraction_std": {
                "CN": float(morph_df[morph_df["label"] == 0]["fg_ratio"].std()),
                "MCI": float(morph_df[morph_df["label"] == 1]["fg_ratio"].std()),
                "AD": float(morph_df[morph_df["label"] == 2]["fg_ratio"].std()),
            },
            "t_tests": {
                "CN_vs_AD": {"t_stat": float(t_ad_cn), "p_val": float(p_ad_cn)},
                "CN_vs_MCI": {"t_stat": float(t_mci_cn), "p_val": float(p_mci_cn)},
                "MCI_vs_AD": {"t_stat": float(t_ad_mci), "p_val": float(p_ad_mci)},
            }
        },
        "multiclass_classification": {
            "linear_probe": {
                "accuracy_mean": float(np.mean(lr_accs)),
                "accuracy_std": float(np.std(lr_accs)),
                "macro_f1_mean": float(np.mean(lr_f1s)),
                "macro_f1_std": float(np.std(lr_f1s)),
                "roc_auc_ovr_mean": float(np.mean(lr_aucs)),
                "roc_auc_ovr_std": float(np.std(lr_aucs)),
            },
            "random_forest": {
                "accuracy_mean": float(np.mean(rf_accs)),
                "accuracy_std": float(np.std(rf_accs)),
                "macro_f1_mean": float(np.mean(rf_f1s)),
                "macro_f1_std": float(np.std(rf_f1s)),
                "roc_auc_ovr_mean": float(np.mean(rf_aucs)),
                "roc_auc_ovr_std": float(np.std(rf_aucs)),
            },
            "confusion_matrix": conf_mat.tolist(),
            "per_class_metrics": class_report,
        },
        "binary_benchmarks": {
            "CN_vs_AD": {
                "accuracy": float(acc_cn_ad),
                "f1_score": float(f1_cn_ad),
                "roc_auc": float(auc_cn_ad)
            },
            "CN_vs_MCI": {
                "accuracy": float(acc_cn_mci),
                "f1_score": float(f1_cn_mci),
                "roc_auc": float(auc_cn_mci)
            }
        }
    }

    with open(os.path.join(output_dir, "external_testing_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save summary dataframe
    summary_df = pd.DataFrame([
        {"Task": "3-Class (CN vs MCI vs AD) [Linear Probe]", "Accuracy": np.mean(lr_accs), "Macro F1": np.mean(lr_f1s), "AUROC": np.mean(lr_aucs)},
        {"Task": "3-Class (CN vs MCI vs AD) [Random Forest]", "Accuracy": np.mean(rf_accs), "Macro F1": np.mean(rf_f1s), "AUROC": np.mean(rf_aucs)},
        {"Task": "Binary: CN vs AD", "Accuracy": acc_cn_ad, "Macro F1": f1_cn_ad, "AUROC": auc_cn_ad},
        {"Task": "Binary: CN vs MCI", "Accuracy": acc_cn_mci, "Macro F1": f1_cn_mci, "AUROC": auc_cn_mci},
    ])
    summary_df.to_csv(os.path.join(output_dir, "external_testing_summary.csv"), index=False)
    print("\n--- Final External Testing Performance Summary ---")
    print(summary_df.to_string(index=False))

    print(f"\nAll external test results successfully exported to {output_dir}")

if __name__ == "__main__":
    run_evaluation()

