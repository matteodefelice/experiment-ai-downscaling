# %%
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================
# 1. LIGHTWEIGHT U-NET ARCHITECTURE
# ============================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
                                   padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightweightResidualBlock(nn.Module):
    def __init__(self, channels):
        super(LightweightResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class LightweightClimateUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(LightweightClimateUNet, self).__init__()
        
        self.base_channels = base_channels
        
        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels*2)
        self.enc3 = self._make_encoder_block(base_channels*2, base_channels*4)
        
        # Bottleneck
        self.bottleneck = LightweightResidualBlock(base_channels*4)
        
        # Decoder
        self.dec3 = self._make_decoder_block(base_channels*4, base_channels*2)
        self.dec2 = self._make_decoder_block(base_channels*2, base_channels)
        self.dec1 = self._make_decoder_block(base_channels, base_channels)
        
        # Skip connection fusion layers
        self.skip_fusion3 = nn.Conv2d(base_channels*8, base_channels*4, kernel_size=1)
        self.skip_fusion2 = nn.Conv2d(base_channels*4, base_channels*2, kernel_size=1)
        self.skip_fusion1 = nn.Conv2d(base_channels*2, base_channels, kernel_size=1)
        
        # Final output
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            nn.ReLU(inplace=True),
            LightweightResidualBlock(out_ch)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            nn.ReLU(inplace=True),
            LightweightResidualBlock(out_ch)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder with skip connections
        dec3_up = self.upsample(bottleneck)
        dec3_concat = torch.cat([dec3_up, enc3], dim=1)
        dec3_fused = self.skip_fusion3(dec3_concat)
        dec3 = self.dec3(dec3_fused)
        
        dec2_up = self.upsample(dec3)
        dec2_concat = torch.cat([dec2_up, enc2], dim=1)
        dec2_fused = self.skip_fusion2(dec2_concat)
        dec2 = self.dec2(dec2_fused)
        
        dec1_up = self.upsample(dec2)
        dec1_concat = torch.cat([dec1_up, enc1], dim=1)
        dec1_fused = self.skip_fusion1(dec1_concat)
        dec1 = self.dec1(dec1_fused)
        
        out = self.final_conv(dec1)
        
        return out


class LightweightClimateUNetResidual(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(LightweightClimateUNetResidual, self).__init__()
        self.unet = LightweightClimateUNet(in_channels, out_channels, base_channels)
        
    def forward(self, x):
        residual = self.unet(x)
        output = x + residual
        return output


# ============================================
# 2. CLIMATE DATASET
# ============================================
class ClimateDatasetSeparate(Dataset):
    def __init__(self, hr_data, lr_data, target_size=None, normalize=True):
        assert len(hr_data) == len(lr_data), "HR and LR must have same length"
        
        self.hr_data = hr_data
        self.lr_data = lr_data
        self.target_size = target_size if target_size else hr_data.shape[1:]
        self.normalize = normalize
        
        if self.normalize:
            self.hr_mean = np.nanmean(hr_data)
            self.hr_std = np.nanstd(hr_data)
            self.lr_mean = np.nanmean(lr_data)
            self.lr_std = np.nanstd(lr_data)
            
            if np.isnan(self.hr_std) or self.hr_std == 0:
                self.hr_std = 1.0
            if np.isnan(self.lr_std) or self.lr_std == 0:
                self.lr_std = 1.0
        
        print(f"HR NaN percentage: {np.isnan(hr_data).sum() / hr_data.size * 100:.2f}%")
        print(f"LR NaN percentage: {np.isnan(lr_data).sum() / lr_data.size * 100:.2f}%")
    
    def __len__(self):
        return len(self.hr_data)
    
    def __getitem__(self, idx):
        hr_sample = self.hr_data[idx].copy()
        lr_sample = self.lr_data[idx].copy()
        
        hr_valid_mask = ~np.isnan(hr_sample)
        
        hr_sample_filled = np.nan_to_num(hr_sample, nan=0.0)
        lr_sample_filled = np.nan_to_num(lr_sample, nan=0.0)
        
        if self.normalize:
            hr_sample_filled = (hr_sample_filled - self.hr_mean) / self.hr_std
            lr_sample_filled = (lr_sample_filled - self.lr_mean) / self.lr_std
        
        lr_tensor = torch.from_numpy(lr_sample_filled).unsqueeze(0).unsqueeze(0).float()
        lr_upsampled = torch.nn.functional.interpolate(
            lr_tensor,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        hr_tensor = torch.from_numpy(hr_sample_filled).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(hr_valid_mask).unsqueeze(0).float()
        
        return lr_upsampled.unsqueeze(0), hr_tensor, mask_tensor


# ============================================
# 3. MASKED MSE LOSS
# ============================================
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, pred, target, mask):
        diff = (pred - target) * mask
        n_valid = mask.sum()
        
        if n_valid == 0:
            return torch.tensor(0.0, device=pred.device)
        
        mse = (diff ** 2).sum() / n_valid
        return mse


# ============================================
# 4. TRAINING FUNCTION
# ============================================
def train_downscaling_model(model, train_loader, device, n_epochs=100, lr=0.001):
    model = model.to(device)
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        for lr_batch, hr_batch, mask_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            if torch.isnan(lr_batch).any() or torch.isnan(hr_batch).any():
                continue
            
            optimizer.zero_grad()
            sr_batch = model(lr_batch)
            
            if torch.isnan(sr_batch).any():
                continue
            
            loss = criterion(sr_batch, hr_batch, mask_batch)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return model


# ============================================
# 5. EVALUATION FUNCTION
# ============================================
def evaluate_model(model, test_loader, device, dataset_stats):
    """Evaluate model on test set"""
    model.eval()
    criterion = MaskedMSELoss()
    
    test_loss = 0
    test_rmse = 0
    valid_batches = 0
    
    with torch.no_grad():
        for lr_batch, hr_batch, mask_batch in test_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward pass
            sr_batch = model(lr_batch)
            
            # Calculate loss
            loss = criterion(sr_batch, hr_batch, mask_batch)
            test_loss += loss.item()
            
            # Calculate RMSE in original units (denormalized)
            sr_denorm = sr_batch * dataset_stats['hr_std'] + dataset_stats['hr_mean']
            hr_denorm = hr_batch * dataset_stats['hr_std'] + dataset_stats['hr_mean']
            
            # Masked RMSE
            diff = (sr_denorm - hr_denorm) * mask_batch
            n_valid = mask_batch.sum()
            if n_valid > 0:
                rmse = torch.sqrt((diff ** 2).sum() / n_valid)
                test_rmse += rmse.item()
            
            valid_batches += 1
    
    avg_test_loss = test_loss / valid_batches if valid_batches > 0 else 0
    avg_test_rmse = test_rmse / valid_batches if valid_batches > 0 else 0
    
    return avg_test_loss, avg_test_rmse


# ============================================
# 6. INFERENCE FUNCTION
# ============================================
def downscale_climate_data(model, lr_data, device, dataset_stats, target_size=(128, 128)):
    """Downscale new climate data"""
    model.eval()
    sr_results = []
    
    with torch.no_grad():
        for i in range(len(lr_data)):
            lr_sample = lr_data[i].copy()
            lr_sample = np.nan_to_num(lr_sample, nan=0.0)
            lr_sample = (lr_sample - dataset_stats['lr_mean']) / dataset_stats['lr_std']
            
            lr_tensor = torch.from_numpy(lr_sample).unsqueeze(0).unsqueeze(0).float()
            lr_upsampled = torch.nn.functional.interpolate(
                lr_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).to(device)
            
            sr_pred = model(lr_upsampled)
            sr_pred = sr_pred.squeeze().cpu().numpy()
            sr_pred = sr_pred * dataset_stats['hr_std'] + dataset_stats['hr_mean']
            
            sr_results.append(sr_pred)
    
    return np.array(sr_results)

def load_climate_data_from_files(hr_path, lr_path):
    """
    Load high-res and low-res data from separate files.
    
    In practice, replace with:
    - xarray for NetCDF: xr.open_dataset(hr_path)['temperature'].values
    - numpy for .npy files: np.load(hr_path)
    - Other formats as needed
    """
    # Example: Load from numpy files
    hr_data = xr.open_dataset(hr_path)['t2m'].values  # Shape: (n_samples, H, W)
    lr_data = xr.open_dataset(lr_path)['t2m'].values  # Shape: (n_samples, H_lr, W_lr)
    
    # For this example, simulate loading
    print(f"Loading HR data from: {hr_path}")
    print(f"Loading LR data from: {lr_path}")
    
    # Simulated data (replace with actual loading)
    # np.random.seed(42)
    # hr_data = np.random.randn(100, 128, 128) * 10 + 288  # 0.25° resolution
    # lr_data = np.random.randn(100, 32, 32) * 10 + 288    # 1° resolution
    print(type(hr_data))
    return hr_data, lr_data

# ============================================
# 7. MAIN WORKFLOW WITH TRAIN/TEST SPLIT
# ============================================
def example_workflow():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load separate HR and LR data
    hr_path = "hr_t2m.nc"  # e.g., ERA5 at 0.25°
    lr_path = "lr_t2m.nc"   # e.g., GCM at 1°
    
    hr_data, lr_data = load_climate_data_from_files(hr_path, lr_path)
    print(f"HR data shape: {hr_data.shape}")  # e.g., (100, 128, 128)
    print(f"LR data shape: {lr_data.shape}")  # e.g., (100, 32, 32)
    
    # Create full dataset
    dataset = ClimateDatasetSeparate(
        hr_data=hr_data,
        lr_data=lr_data,
        target_size=(128, 128),
        normalize=True
    )
    
    # Store normalization statistics
    dataset_stats = {
        'hr_mean': dataset.hr_mean,
        'hr_std': dataset.hr_std,
        'lr_mean': dataset.lr_mean,
        'lr_std': dataset.lr_std
    }
    
    # Split dataset: 80% train, 20% test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset split:")
    print(f"Training samples: {train_size}")
    print(f"Test samples: {test_size}\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = LightweightClimateUNetResidual(in_channels=1, out_channels=1, base_channels=32)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Memory footprint: ~{total_params * 4 / 1024 / 1024:.1f} MB\n")
    
    # Train model
    print("Training lightweight U-Net...\n")
    trained_model = train_downscaling_model(
        model, 
        train_loader, 
        device, 
        n_epochs=50)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    test_loss, test_rmse = evaluate_model(trained_model, test_loader, device, dataset_stats)
    print(f"\nTest Loss (normalized): {test_loss:.6f}")
    print(f"Test RMSE (original units): {test_rmse:.4f} K")
    
    # Example inference on new data
    print("\n" + "="*50)
    print("Downscaling new climate data...")
    print("="*50)
    new_lr_data = lr_data[:5]
    downscaled = downscale_climate_data(trained_model, new_lr_data, device, dataset_stats)
    print(f"Downscaled data shape: {downscaled.shape}")
    
    # Save model and statistics
    torch.save({
        'model_state_dict': model.state_dict(),
        'dataset_stats': dataset_stats,
        'test_loss': test_loss,
        'test_rmse': test_rmse
    }, 'climate_lightweight_unet_model.pth')
    print("\nModel saved successfully!")
    
    return trained_model, dataset_stats, test_loss, test_rmse, downscaled

# %%
if __name__ == "__main__":
    model, stats, test_loss, test_rmse, downscaled = example_workflow()

    # Reload hr data
    hr_data = xr.open_dataset('hr_t2m.nc')
    # Rebuild downscaled
    ds_data = xr.DataArray(
        np.flip(downscaled, 1),
        coords={
        "latitude": hr_data.coords["latitude"],
        "longitude": hr_data.coords["longitude"]
        },
    dims={'time': 5, 'latitude': 128, 'longitude': 128}
    )

    ds_data.where(hr_data['t2m'].isel(time=0) > 0).isel(time = 0).plot(vmin = 250, vmax = 290)
