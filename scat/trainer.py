"""
Training module for SCAT classifier.
Supports Random Forest and CNN (transfer learning) models.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

from .detector import Deposit
from .features import FeatureExtractor

# Lazy imports for sklearn (loaded on first use)
_sklearn_loaded = False
_RandomForestClassifier = None
_train_test_split = None
_cross_val_score = None
_classification_report = None
_confusion_matrix = None
_StandardScaler = None
_compute_class_weight = None


def _load_sklearn():
    """Lazy load sklearn modules."""
    global _sklearn_loaded, _RandomForestClassifier, _train_test_split
    global _cross_val_score, _classification_report, _confusion_matrix
    global _StandardScaler, _compute_class_weight
    
    if _sklearn_loaded:
        return
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    
    _RandomForestClassifier = RandomForestClassifier
    _train_test_split = train_test_split
    _cross_val_score = cross_val_score
    _classification_report = classification_report
    _confusion_matrix = confusion_matrix
    _StandardScaler = StandardScaler
    _compute_class_weight = compute_class_weight
    _sklearn_loaded = True


class DataLoader:
    """Load labeled data from JSON files and images."""
    
    def __init__(self, image_dir: Path, label_dir: Optional[Path] = None):
        """
        Args:
            image_dir: Directory containing image files
            label_dir: Directory containing label JSON files (default: same as image_dir)
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else self.image_dir
    
    def find_label_files(self) -> List[Path]:
        """Find all label JSON files."""
        return list(self.label_dir.glob("*.labels.json"))
    
    def load_labeled_data(self, label_files: Optional[List[Path]] = None) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
        """
        Load patches, features, and labels from label files.
        
        Returns:
            patches: List of image patches
            features: List of feature dicts
            labels: List of label strings
        """
        if label_files is None:
            label_files = self.find_label_files()
        
        patches = []
        features = []
        labels = []
        
        for label_file in label_files:
            with open(label_file) as f:
                data = json.load(f)
            
            # Find corresponding image (support both 'image_file' and 'filename' keys)
            image_ref = data.get('image_file', data.get('filename', ''))
            image_name = Path(image_ref).name
            image_path = self.image_dir / image_name
            
            # Try common extensions if not found
            if not image_path.exists():
                for ext in ['.tif', '.tiff', '.png', '.jpg']:
                    candidate = self.image_dir / (image_path.stem + ext)
                    if candidate.exists():
                        image_path = candidate
                        break
            
            if not image_path.exists():
                print(f"Warning: Image not found for {label_file.name}, skipping...")
                continue
            
            # Load image
            image = np.array(Image.open(image_path))
            dpi = Image.open(image_path).info.get('dpi', (600, 600))[0]
            extractor = FeatureExtractor(dpi=dpi)
            
            # Convert to HLS for feature extraction
            hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            
            for dep in data['deposits']:
                label = dep.get('label', 'unknown')
                if label not in ['normal', 'rod', 'artifact']:
                    continue
                
                # Reconstruct contour
                contour = np.array(dep['contour'])
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract patch with padding
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                patch = image[y1:y2, x1:x2]
                
                if patch.size == 0:
                    continue
                
                # Extract features
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                
                pixels_hls = hls[mask > 0]
                if len(pixels_hls) > 0:
                    mean_hue = np.mean(pixels_hls[:, 0]) * 2
                    mean_lightness = np.mean(pixels_hls[:, 1]) / 255.0
                    mean_saturation = np.mean(pixels_hls[:, 2]) / 255.0
                else:
                    mean_hue, mean_lightness, mean_saturation = 0, 0, 0
                
                iod = area * (1 - mean_lightness)
                
                feature_dict = {
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'mean_hue': mean_hue,
                    'mean_lightness': mean_lightness,
                    'mean_saturation': mean_saturation,
                    'iod': iod
                }
                
                patches.append(patch)
                features.append(feature_dict)
                labels.append(label)
        
        print(f"Loaded {len(labels)} samples from {len(label_files)} files")
        print(f"  Normal: {labels.count('normal')}")
        print(f"  ROD: {labels.count('rod')}")
        print(f"  Artifact: {labels.count('artifact')}")
        
        return patches, features, labels


class DataAugmenter:
    """Data augmentation for image patches."""
    
    @staticmethod
    def augment_patch(patch: np.ndarray, n_augments: int = 4) -> List[np.ndarray]:
        """
        Generate augmented versions of a patch.
        
        Args:
            patch: Original image patch
            n_augments: Number of augmented versions to generate
            
        Returns:
            List of augmented patches (including original)
        """
        augmented = [patch]
        
        for _ in range(n_augments):
            aug = patch.copy()
            
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            aug = np.rot90(aug, k)
            
            # Random flip
            if np.random.random() > 0.5:
                aug = np.fliplr(aug)
            if np.random.random() > 0.5:
                aug = np.flipud(aug)
            
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            aug = np.clip(aug * brightness, 0, 255).astype(np.uint8)
            
            augmented.append(aug)
        
        return augmented


class RFTrainer:
    """Random Forest classifier trainer."""
    
    def __init__(self):
        _load_sklearn()  # Lazy load sklearn
        self.model = None
        self.scaler = _StandardScaler()
        self.feature_names = [
            'circularity', 'aspect_ratio', 'area',
            'mean_hue', 'mean_saturation', 'mean_lightness', 'iod'
        ]
    
    def _features_to_array(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dicts to numpy array."""
        return np.array([[f[name] for name in self.feature_names] for f in features])
    
    def train(
        self,
        features: List[Dict],
        labels: List[str],
        n_estimators: int = 100,
        class_weight: str = 'balanced',
        test_size: float = 0.2,
        cross_validate: bool = True
    ) -> Dict:
        """
        Train Random Forest classifier.
        
        Args:
            features: List of feature dicts
            labels: List of labels
            n_estimators: Number of trees
            class_weight: 'balanced' or None
            test_size: Fraction for test set
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dict with training results
        """
        X = self._features_to_array(features)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = _train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model = _RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        results = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': float(np.mean(y_pred == y_test)),
            'classification_report': _classification_report(y_test, y_pred),
            'confusion_matrix': _confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': dict(zip(self.feature_names, 
                                           self.model.feature_importances_.tolist()))
        }
        
        # Cross-validation
        if cross_validate:
            cv_scores = _cross_val_score(self.model, X_scaled, y, cv=5)
            results['cv_scores'] = cv_scores.tolist()
            results['cv_mean'] = float(cv_scores.mean())
            results['cv_std'] = float(cv_scores.std())
        
        return results
    
    def save(self, path: Path):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
    
    def predict(self, features: List[Dict]) -> Tuple[List[str], List[float]]:
        """Predict labels for features."""
        X = self._features_to_array(features)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidences = [float(np.max(p)) for p in probabilities]
        
        return predictions.tolist(), confidences


class CNNTrainer:
    """CNN classifier trainer using transfer learning."""
    
    def __init__(self, patch_size: int = 64):
        self.patch_size = patch_size
        self.model = None
        self.device = None
        self.class_names = ['artifact', 'normal', 'rod']
    
    def _preprocess_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """Preprocess patches for CNN."""
        processed = []
        for patch in patches:
            # Resize
            resized = cv2.resize(patch, (self.patch_size, self.patch_size))
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            # CHW format
            chw = np.transpose(normalized, (2, 0, 1))
            processed.append(chw)
        return np.array(processed)
    
    def train(
        self,
        patches: List[np.ndarray],
        labels: List[str],
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        augment_minority: bool = True,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train CNN classifier with transfer learning.
        
        Args:
            patches: List of image patches
            labels: List of labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            augment_minority: Whether to augment minority class (ROD)
            test_size: Fraction for test set
            
        Returns:
            Dict with training results
        """
        _load_sklearn()  # Lazy load sklearn
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            from torchvision import models
        except ImportError:
            raise ImportError("PyTorch required for CNN training. Install with: pip install torch torchvision")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Augment minority class
        if augment_minority:
            augmenter = DataAugmenter()
            aug_patches, aug_labels = [], []
            
            for patch, label in zip(patches, labels):
                if label == 'rod':
                    augmented = augmenter.augment_patch(patch, n_augments=5)
                    aug_patches.extend(augmented)
                    aug_labels.extend([label] * len(augmented))
                else:
                    aug_patches.append(patch)
                    aug_labels.append(label)
            
            patches, labels = aug_patches, aug_labels
            print(f"After augmentation: {len(labels)} samples")
            print(f"  ROD: {labels.count('rod')}")
        
        # Preprocess
        X = self._preprocess_patches(patches)
        y = np.array([self.class_names.index(l) for l in labels])
        
        # Split
        X_train, X_test, y_train, y_test = _train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create model
        self.model = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, len(self.class_names))
        )
        self.model = self.model.to(self.device)
        
        # Class weights
        class_weights = _compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        history = {'train_loss': [], 'test_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluate
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            acc = correct / total
            avg_loss = total_loss / len(train_loader)
            
            history['train_loss'].append(avg_loss)
            history['test_acc'].append(acc)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Test Acc={acc:.4f}")
        
        # Final evaluation
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        pred_names = [self.class_names[i] for i in all_preds]
        label_names = [self.class_names[i] for i in all_labels]
        
        results = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'final_accuracy': history['test_acc'][-1],
            'classification_report': classification_report(label_names, pred_names),
            'confusion_matrix': confusion_matrix(label_names, pred_names).tolist(),
            'history': history
        }
        
        return results
    
    def save(self, path: Path):
        """Save model to file."""
        import torch
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'patch_size': self.patch_size
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from file."""
        import torch
        from torchvision import models
        import torch.nn as nn
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        self.patch_size = checkpoint['patch_size']
        
        # Recreate model architecture
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, len(self.class_names))
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, patches: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """Predict labels for patches."""
        import torch
        
        X = self._preprocess_patches(patches)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predictions = [self.class_names[i] for i in predicted.cpu().numpy()]
        confidences = [float(probs[i, predicted[i]]) for i in range(len(predicted))]
        
        return predictions, confidences


def train_from_labels(
    image_dir: str,
    label_dir: Optional[str] = None,
    output_path: str = "model.pkl",
    model_type: str = "rf",
    **kwargs
) -> Dict:
    """
    Convenience function to train model from label files.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing label JSONs (default: same as image_dir)
        output_path: Path to save trained model
        model_type: "rf" for Random Forest, "cnn" for CNN
        **kwargs: Additional arguments for trainer
        
    Returns:
        Training results dict
    """
    # Load data
    loader = DataLoader(Path(image_dir), Path(label_dir) if label_dir else None)
    patches, features, labels = loader.load_labeled_data()
    
    if len(labels) == 0:
        raise ValueError("No labeled data found!")
    
    # Train
    if model_type == "rf":
        trainer = RFTrainer()
        results = trainer.train(features, labels, **kwargs)
        trainer.save(Path(output_path))
    elif model_type == "cnn":
        trainer = CNNTrainer()
        results = trainer.train(patches, labels, **kwargs)
        trainer.save(Path(output_path))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Model type: {model_type}")
    print(f"Train size: {results['train_size']}")
    print(f"Test size: {results['test_size']}")
    
    if 'cv_mean' in results:
        print(f"Cross-validation: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    
    print(f"\nClassification Report:\n{results['classification_report']}")
    print(f"Confusion Matrix:\n{np.array(results['confusion_matrix'])}")
    
    if 'feature_importance' in results:
        print("\nFeature Importance:")
        for name, imp in sorted(results['feature_importance'].items(), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.3f}")
    
    return results
