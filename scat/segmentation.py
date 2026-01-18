"""
U-Net Segmentation for deposit detection.
Learns to segment deposits at pixel level from labeled data.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Lazy imports for PyTorch
_torch = None
_nn = None
_F = None

def _load_torch():
    """Lazy load PyTorch modules."""
    global _torch, _nn, _F
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F
    return _torch, _nn, _F


# =============================================================================
# U-Net Architecture
# =============================================================================

class DoubleConv:
    """Double convolution block for U-Net."""
    
    @staticmethod
    def create(in_channels: int, out_channels: int):
        torch, nn, F = _load_torch()
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class UNet:
    """U-Net model for semantic segmentation."""
    
    @staticmethod
    def create(in_channels: int = 3, out_channels: int = 1, features: List[int] = None):
        """
        Create U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output classes (1 for binary segmentation)
            features: Feature sizes for each level [64, 128, 256, 512]
        """
        torch, nn, F = _load_torch()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        class UNetModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Encoder (downsampling)
                self.encoders = nn.ModuleList()
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                in_ch = in_channels
                for feature in features:
                    self.encoders.append(DoubleConv.create(in_ch, feature))
                    in_ch = feature
                
                # Bottleneck
                self.bottleneck = DoubleConv.create(features[-1], features[-1] * 2)
                
                # Decoder (upsampling)
                self.decoders = nn.ModuleList()
                self.upconvs = nn.ModuleList()
                
                rev_features = features[::-1]
                in_ch = features[-1] * 2
                for feature in rev_features:
                    self.upconvs.append(
                        nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2)
                    )
                    self.decoders.append(DoubleConv.create(feature * 2, feature))
                    in_ch = feature
                
                # Final convolution
                self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
            
            def forward(self, x):
                # Encoder path
                skip_connections = []
                
                for encoder in self.encoders:
                    x = encoder(x)
                    skip_connections.append(x)
                    x = self.pool(x)
                
                # Bottleneck
                x = self.bottleneck(x)
                
                # Decoder path
                skip_connections = skip_connections[::-1]
                
                for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
                    x = upconv(x)
                    skip = skip_connections[idx]
                    
                    # Handle size mismatch
                    if x.shape != skip.shape:
                        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                    
                    x = torch.cat([skip, x], dim=1)
                    x = decoder(x)
                
                return self.final_conv(x)
        
        return UNetModel()


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

@dataclass
class SegmentationSample:
    """A single training sample for segmentation."""
    image: np.ndarray
    mask: np.ndarray
    filename: str


class SegmentationDataLoader:
    """Load and prepare data for U-Net training."""
    
    def __init__(self, image_dir: Path, label_dir: Optional[Path] = None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else self.image_dir
    
    def find_label_files(self) -> List[Path]:
        """Find all label JSON files."""
        return list(self.label_dir.glob("*.labels.json"))
    
    def load_sample(self, label_file: Path) -> Optional[SegmentationSample]:
        """Load a single sample from label file."""
        with open(label_file) as f:
            data = json.load(f)
        
        # Find corresponding image (support both 'image_file' and 'filename' keys)
        image_ref = data.get('image_file', data.get('filename', ''))
        image_name = Path(image_ref).name
        image_path = self._find_image(image_name)
        
        if image_path is None:
            print(f"Warning: Image not found for {label_file.name}")
            return None
        
        # Load image
        from PIL import Image
        image = np.array(Image.open(image_path))
        h, w = image.shape[:2]
        
        # Create mask from deposits
        mask = self._create_mask_from_deposits(data['deposits'], image, (h, w))
        
        return SegmentationSample(
            image=image,
            mask=mask,
            filename=image_name
        )
    
    def _find_image(self, image_name: str) -> Optional[Path]:
        """Find image file with various extensions."""
        image_path = self.image_dir / image_name
        if image_path.exists():
            return image_path
        
        stem = Path(image_name).stem
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.TIF', '.TIFF']:
            candidate = self.image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        
        return None
    
    def _create_mask_from_deposits(
        self, 
        deposits: List[Dict], 
        image: np.ndarray,
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create segmentation mask from deposit data.
        
        For merged contours (may include background), extracts actual deposit pixels.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for dep in deposits:
            label = dep.get('label', 'unknown')
            if label not in ['normal', 'rod']:  # Skip artifacts for detection training
                continue
            
            contour = np.array(dep['contour'])
            if len(contour) < 3:
                continue
            
            # Check if this looks like a merged contour (large, irregular)
            is_merged = dep.get('merged', False)
            area = cv2.contourArea(contour)
            
            if is_merged or area > 500:  # Potentially merged, extract real pixels
                deposit_mask = self._extract_real_deposit_pixels(image, contour)
            else:
                # Small deposit, use contour directly
                deposit_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(deposit_mask, [contour], -1, 255, -1)
            
            mask = cv2.bitwise_or(mask, deposit_mask)
        
        return mask
    
    def _extract_real_deposit_pixels(
        self, 
        image: np.ndarray, 
        contour: np.ndarray
    ) -> np.ndarray:
        """
        Extract actual deposit pixels within a contour region.
        Useful for merged contours that may include background.
        """
        h, w = image.shape[:2]
        
        # Create mask for the contour region
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Extract ROI
        roi = image[y:y+bh, x:x+bw]
        roi_mask = region_mask[y:y+bh, x:x+bw]
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Apply mask to only consider pixels within contour
        gray_masked = cv2.bitwise_and(gray, gray, mask=roi_mask)
        
        # Find actual deposit pixels (darker than background)
        # Use Otsu's threshold within the region
        pixels_in_region = gray[roi_mask > 0]
        if len(pixels_in_region) == 0:
            return region_mask
        
        # Calculate threshold based on region statistics
        mean_val = np.mean(pixels_in_region)
        std_val = np.std(pixels_in_region)
        
        # Deposits are darker, so threshold at mean - 0.5*std
        thresh_val = mean_val - 0.3 * std_val
        
        # Create deposit mask
        _, deposit_thresh = cv2.threshold(
            gray_masked, 
            int(thresh_val), 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Only keep pixels within original contour
        deposit_thresh = cv2.bitwise_and(deposit_thresh, roi_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        deposit_thresh = cv2.morphologyEx(deposit_thresh, cv2.MORPH_OPEN, kernel)
        deposit_thresh = cv2.morphologyEx(deposit_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Put back into full-size mask
        result_mask = np.zeros((h, w), dtype=np.uint8)
        result_mask[y:y+bh, x:x+bw] = deposit_thresh
        
        return result_mask
    
    def load_all_samples(self) -> List[SegmentationSample]:
        """Load all samples from label files."""
        label_files = self.find_label_files()
        samples = []
        
        for label_file in label_files:
            sample = self.load_sample(label_file)
            if sample is not None:
                samples.append(sample)
        
        return samples


# =============================================================================
# Training
# =============================================================================

class DiceLoss:
    """Dice Loss for segmentation - better for imbalanced classes."""
    
    @staticmethod
    def compute(pred, target, smooth=1.0):
        """
        Compute Dice Loss.
        
        Args:
            pred: Predicted logits (before sigmoid)
            target: Ground truth mask
            smooth: Smoothing factor to avoid division by zero
        """
        torch, nn, F = _load_torch()
        
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice


class CombinedLoss:
    """Combined BCE + Dice Loss for better segmentation."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        torch, nn, F = _load_torch()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def __call__(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = DiceLoss.compute(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class SegmentationTrainer:
    """Train U-Net for deposit segmentation."""
    
    def __init__(
        self,
        image_size: int = 256,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        device: str = None
    ):
        torch, nn, F = _load_torch()
        
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
    
    def _augment_image_mask(self, img: np.ndarray, msk: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Apply various augmentations to image and mask.
        
        Returns list of (augmented_image, augmented_mask) tuples.
        """
        results = [(img, msk)]  # Original
        
        # 1. Horizontal flip
        results.append((np.fliplr(img).copy(), np.fliplr(msk).copy()))
        
        # 2. Vertical flip
        results.append((np.flipud(img).copy(), np.flipud(msk).copy()))
        
        # 3. 90 degree rotations
        results.append((np.rot90(img, 1).copy(), np.rot90(msk, 1).copy()))
        results.append((np.rot90(img, 2).copy(), np.rot90(msk, 2).copy()))
        results.append((np.rot90(img, 3).copy(), np.rot90(msk, 3).copy()))
        
        # 4. Brightness variations (image only, mask unchanged)
        for factor in [0.8, 1.2]:
            bright_img = np.clip(img * factor, 0, 1).astype(np.float32)
            results.append((bright_img, msk))
        
        # 5. Contrast variations (image only)
        for factor in [0.8, 1.2]:
            mean = img.mean()
            contrast_img = np.clip((img - mean) * factor + mean, 0, 1).astype(np.float32)
            results.append((contrast_img, msk))
        
        # 6. Gaussian noise (image only)
        noise_std = 0.02
        noisy_img = np.clip(img + np.random.normal(0, noise_std, img.shape), 0, 1).astype(np.float32)
        results.append((noisy_img, msk))
        
        # 7. Combined: flip + brightness
        flip_img = np.fliplr(img).copy()
        flip_msk = np.fliplr(msk).copy()
        bright_flip = np.clip(flip_img * 0.9, 0, 1).astype(np.float32)
        results.append((bright_flip, flip_msk))
        
        return results
    
    def prepare_data(
        self, 
        samples: List[SegmentationSample],
        augment: bool = True
    ) -> Tuple:
        """Prepare training data with enhanced augmentation."""
        torch, nn, F = _load_torch()
        
        images = []
        masks = []
        
        for sample in samples:
            # Resize
            img = cv2.resize(sample.image, (self.image_size, self.image_size))
            msk = cv2.resize(sample.mask, (self.image_size, self.image_size))
            
            # Normalize image
            img = img.astype(np.float32) / 255.0
            msk = (msk > 127).astype(np.float32)
            
            if augment:
                # Apply all augmentations
                augmented = self._augment_image_mask(img, msk)
                for aug_img, aug_msk in augmented:
                    images.append(aug_img)
                    masks.append(aug_msk)
            else:
                images.append(img)
                masks.append(msk)
        
        # Convert to tensors
        images = np.array(images)
        masks = np.array(masks)
        
        # Images: (N, H, W, C) -> (N, C, H, W)
        if len(images.shape) == 4:
            images = np.transpose(images, (0, 3, 1, 2))
        else:
            images = np.expand_dims(images, 1)
        
        # Masks: (N, H, W) -> (N, 1, H, W)
        masks = np.expand_dims(masks, 1)
        
        X = torch.FloatTensor(images)
        y = torch.FloatTensor(masks)
        
        return X, y
    
    def train(
        self,
        samples: List[SegmentationSample],
        epochs: int = 50,
        val_split: float = 0.2,
        patience: int = 10,
        progress_callback=None
    ) -> Dict:
        """
        Train U-Net model with early stopping and LR scheduling.
        
        Args:
            samples: List of training samples
            epochs: Maximum number of training epochs
            val_split: Validation split ratio
            patience: Early stopping patience (epochs without improvement)
            progress_callback: Function(epoch, loss, val_loss, val_iou) for progress updates
            
        Returns:
            Dict with training history
        """
        torch, nn, F = _load_torch()
        from torch.utils.data import TensorDataset, DataLoader
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        
        # Prepare data
        X, y = self.prepare_data(samples, augment=True)
        
        print(f"Training data: {len(X)} samples (with augmentation)")
        
        # Split into train/val
        n_val = int(len(X) * val_split)
        indices = torch.randperm(len(X))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Create model
        in_channels = X.shape[1]
        self.model = UNet.create(in_channels=in_channels, out_channels=1)
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,      # LR *= 0.5 when plateau
            patience=5,      # Wait 5 epochs before reducing
            min_lr=1e-6,
            verbose=True
        )
        
        # Training loop with early stopping
        history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'lr': []}
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_iou = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Calculate IoU
                    preds = torch.sigmoid(outputs) > 0.5
                    val_iou += self._calculate_iou(preds, batch_y > 0.5)
            
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            history['lr'].append(current_lr)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if progress_callback:
                progress_callback(epoch + 1, train_loss, val_loss, val_iou)
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        history['best_val_loss'] = best_val_loss
        history['stopped_epoch'] = epoch + 1
        return history
    
    def _calculate_iou(self, pred: 'torch.Tensor', target: 'torch.Tensor') -> float:
        """Calculate Intersection over Union."""
        intersection = (pred & target).float().sum()
        union = (pred | target).float().sum()
        
        if union == 0:
            return 1.0
        return (intersection / union).item()
    
    def save(self, path: Path):
        """Save trained model."""
        torch, nn, F = _load_torch()
        
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'image_size': self.image_size,
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load trained model."""
        torch, nn, F = _load_torch()
        
        checkpoint = torch.load(path, map_location=self.device)
        self.image_size = checkpoint.get('image_size', 256)
        
        self.model = UNet.create(in_channels=3, out_channels=1)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {path}")


# =============================================================================
# Inference
# =============================================================================

class UNetDetector:
    """Use trained U-Net for deposit detection."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.image_size = 256
        self.device = None
        
        if model_path:
            self.load(model_path)
    
    def load(self, path: Path):
        """Load trained model."""
        torch, nn, F = _load_torch()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=self.device)
        self.image_size = checkpoint.get('image_size', 256)
        
        self.model = UNet.create(in_channels=3, out_channels=1)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict segmentation mask for an image.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            threshold: Threshold for binary mask
            
        Returns:
            Binary mask (H, W) with deposit regions
        """
        torch, nn, F = _load_torch()
        
        if self.model is None:
            raise ValueError("No model loaded. Call load() first.")
        
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        img = cv2.resize(image, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        # (H, W, C) -> (1, C, H, W)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(img).to(self.device)
            output = self.model(x)
            prob = torch.sigmoid(output)
            mask = (prob > threshold).cpu().numpy()[0, 0]
        
        # Resize back to original size
        mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h))
        
        return mask * 255
    
    def detect(self, image: np.ndarray, min_area: int = 20) -> List[np.ndarray]:
        """
        Detect deposits and return contours.
        
        Args:
            image: Input image
            min_area: Minimum contour area
            
        Returns:
            List of contours
        """
        mask = self.predict(image)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        return contours


# =============================================================================
# Convenience Functions
# =============================================================================

def train_segmentation_model(
    image_dir: str,
    label_dir: Optional[str] = None,
    output_path: str = "model_unet.pt",
    epochs: int = 50,
    patience: int = 10,
    image_size: int = 256,
    batch_size: int = 4,
    progress_callback=None
) -> Dict:
    """
    Train U-Net segmentation model from labeled data.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing label JSON files (default: same as image_dir)
        output_path: Path to save trained model
        epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs without improvement)
        image_size: Image size for training
        batch_size: Batch size
        progress_callback: Progress callback function
        
    Returns:
        Training history dict
    """
    # Load data
    loader = SegmentationDataLoader(
        Path(image_dir),
        Path(label_dir) if label_dir else None
    )
    samples = loader.load_all_samples()
    
    if len(samples) < 5:
        raise ValueError(f"Not enough training samples. Found {len(samples)}, need at least 5.")
    
    print(f"Loaded {len(samples)} training samples")
    
    # Train
    trainer = SegmentationTrainer(
        image_size=image_size,
        batch_size=batch_size
    )
    
    history = trainer.train(
        samples,
        epochs=epochs,
        patience=patience,
        progress_callback=progress_callback
    )
    
    # Save
    trainer.save(Path(output_path))
    
    return history
