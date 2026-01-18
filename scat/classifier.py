"""
ML-based classifier for ROD vs Normal deposit classification.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import pickle

from .detector import Deposit


@dataclass
class ClassifierConfig:
    model_type: str = "threshold"
    circularity_threshold: float = 0.6
    model_path: Optional[str] = None


class ThresholdClassifier:
    """Simple circularity-based classifier."""
    
    def __init__(self, circularity_threshold: float = 0.6):
        self.threshold = circularity_threshold
    
    def predict(self, deposits: List[Deposit]) -> List[Deposit]:
        for deposit in deposits:
            if deposit.circularity < self.threshold:
                deposit.label = "rod"
                deposit.confidence = 1.0 - deposit.circularity / self.threshold
            else:
                deposit.label = "normal"
                deposit.confidence = (deposit.circularity - self.threshold) / (1 - self.threshold)
        return deposits


class RandomForestClassifier:
    """Random Forest classifier using morphological + color features."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = [
            'circularity', 'aspect_ratio', 'area', 
            'mean_hue', 'mean_saturation', 'mean_lightness', 'iod'
        ]
    
    def _get_features(self, deposit: Deposit) -> np.ndarray:
        return np.array([
            deposit.circularity,
            deposit.aspect_ratio,
            deposit.area,
            deposit.mean_hue,
            deposit.mean_saturation,
            deposit.mean_lightness,
            deposit.iod
        ])
    
    def load(self, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data.get('scaler')
    
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def train(self, deposits: List[Deposit], labels: List[str], n_estimators: int = 100):
        from sklearn.ensemble import RandomForestClassifier as RFC
        from sklearn.preprocessing import StandardScaler
        
        X = np.array([self._get_features(d) for d in deposits])
        y = np.array(labels)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RFC(n_estimators=n_estimators, random_state=42)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, deposits: List[Deposit]) -> List[Deposit]:
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X = np.array([self._get_features(d) for d in deposits])
        if self.scaler:
            X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        for deposit, pred, prob in zip(deposits, predictions, probabilities):
            deposit.label = pred
            deposit.confidence = float(np.max(prob))
        
        return deposits


class CNNClassifier:
    """CNN-based classifier using deposit image patches."""
    
    def __init__(self, model_path: Optional[Path] = None, patch_size: int = 64):
        self.model_path = model_path
        self.patch_size = patch_size
        self.model = None
        self.device = None
    
    def load(self, path: Path):
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(path, map_location=self.device)
        self.model.eval()
    
    def _preprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        import cv2
        patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        patch = np.transpose(patch, (2, 0, 1))
        return patch
    
    def predict(self, deposits: List[Deposit], image: np.ndarray) -> List[Deposit]:
        import torch
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        patches = [self._preprocess_patch(d.get_patch(image, padding=10)) for d in deposits]
        batch = torch.tensor(np.array(patches), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        labels = ['artifact', 'normal', 'rod']
        for deposit, pred, prob in zip(deposits, preds.cpu().numpy(), probs.cpu().numpy()):
            deposit.label = labels[pred]
            deposit.confidence = float(prob[pred])
        
        return deposits
    
    @staticmethod
    def create_model(num_classes: int = 3):
        import torch.nn as nn
        from torchvision import models
        
        model = models.resnet18(pretrained=True)
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
        return model


def get_classifier(config: ClassifierConfig):
    if config.model_type == "threshold":
        return ThresholdClassifier(config.circularity_threshold)
    elif config.model_type == "rf":
        clf = RandomForestClassifier()
        if config.model_path:
            clf.load(Path(config.model_path))
        return clf
    elif config.model_type == "cnn":
        clf = CNNClassifier()
        if config.model_path:
            clf.load(Path(config.model_path))
        return clf
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
