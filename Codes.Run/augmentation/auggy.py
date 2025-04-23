import os
import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import albumentations as A
import random
from pathlib import Path

def create_augmented_dataset(input_dir, output_dir, samples_per_image=10, max_image_size=1024):
    """
    Create an augmented dataset of galaxy images while preserving key features.
    Memory-efficient version that resizes large images.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing original galaxy images
    output_dir : str
        Directory where augmented images will be saved
    samples_per_image : int
        Number of augmented samples to generate per original image
    max_image_size : int
        Maximum size (width or height) for images to prevent memory issues
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = list(Path(input_dir).glob('**/*.jpg')) + list(Path(input_dir).glob('**/*.png')) + \
                  list(Path(input_dir).glob('**/*.jpeg')) + list(Path(input_dir).glob('**/*.tif'))
    
    # Configure augmentation pipeline - carefully chosen to preserve galaxy features
    transform = A.Compose([
        A.RandomRotate90(p=0.5),              # 90-degree rotations
        A.Rotate(limit=180, p=0.7),           # Full rotation range for galaxies
        A.HorizontalFlip(p=0.5),              # Horizontal flipping
        A.VerticalFlip(p=0.5),                # Vertical flipping
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.7
        ),                                    # Careful brightness/contrast adjustment
        A.Affine(
            scale=(0.9, 1.1),                 # Minimal zoom to preserve features
            translate_percent=(0.1, 0.1),     # Small shifts
            p=0.5
        ),
    ])
    
    # Process each image
    for img_path in image_files:
        # Extract galaxy type/class from filename if available
        img_file = img_path.name
        galaxy_class = img_file.split('_')[0] if '_' in img_file else 'unknown'
        
        # Read image
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # Resize large images to prevent memory issues
            h, w = img.shape[:2]
            if h > max_image_size or w > max_image_size:
                # Calculate new dimensions while maintaining aspect ratio
                if h > w:
                    new_h, new_w = max_image_size, int(w * max_image_size / h)
                else:
                    new_h, new_w = int(h * max_image_size / w), max_image_size
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"Resized {img_file} from {h}x{w} to {new_h}x{new_w}")
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Could not process {img_path}: {e}")
            continue
            
        # Astronomy-specific processing before augmentation
        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Apply additional astronomy-specific augmentations
        augmented_images = []
        
        # Store original
        augmented_images.append((img_float, f"{galaxy_class}_original"))
        
        # Generate augmented samples
        for i in range(samples_per_image - 1):  # -1 because we already added the original
            # Apply the transformation
            augmented = transform(image=img)['image']
            aug_img = augmented.astype(np.float32) / 255.0
            augmented_images.append((aug_img, f"{galaxy_class}_aug_{i+1}"))
        
        # Additional astronomy-specific transformations
        for i in range(len(augmented_images)):
            img_aug, name = augmented_images[i]
            
            # Memory-efficient astronomy-specific enhancements
            # Use only one random enhancement per image to reduce memory usage
            enhancement = random.randint(0, 3)
            
            if enhancement == 1:
                # Log transformation - enhances dim features
                try:
                    img_aug = exposure.adjust_log(img_aug, 1.0)
                    name += "_log"
                except Exception as e:
                    print(f"Warning: Log transform failed: {e}")
                    
            elif enhancement == 2:
                # Gamma correction - adjusts contrast
                try:
                    gamma = random.uniform(0.8, 1.2)
                    img_aug = exposure.adjust_gamma(img_aug, gamma)
                    name += f"_gamma{gamma:.1f}"
                except Exception as e:
                    print(f"Warning: Gamma correction failed: {e}")
            
            elif enhancement == 3:
                # Apply per-channel histogram equalization to avoid memory errors
                try:
                    # Process each channel separately to avoid memory issues
                    for c in range(img_aug.shape[2]):
                        img_aug[:,:,c] = exposure.equalize_hist(img_aug[:,:,c])
                    name += "_histeq"
                except Exception as e:
                    print(f"Warning: Histogram equalization failed: {e}")
            
            augmented_images[i] = (img_aug, name)
        
        # Save all augmented images
        for img_aug, name in augmented_images:
            # Convert back to uint8 for saving
            img_save = (img_aug * 255).astype(np.uint8)
            
            # Save the image
            output_path = Path(output_dir) / f"{name}.jpg"
            
            # Use cv2 to save the image (more memory efficient than plt.imsave)
            cv2.imwrite(str(output_path), cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
            
        print(f"Processed {img_file} - created {len(augmented_images)} variants")

def visualize_augmentations(original_img_path, output_dir="augmentation_preview", samples=5, max_image_size=1024):
    """
    Visualize augmentations on a single image for preview purposes
    
    Parameters:
    -----------
    original_img_path : str
        Path to original galaxy image
    output_dir : str
        Directory to save visualization
    samples : int
        Number of augmentation samples to generate
    max_image_size : int
        Maximum size (width or height) for images to prevent memory issues
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # Read image
    try:
        img = cv2.imread(str(original_img_path))
        if img is None:
            print(f"Error: Could not read {original_img_path}")
            return
            
        # Resize large images to prevent memory issues
        h, w = img.shape[:2]
        if h > max_image_size or w > max_image_size:
            if h > w:
                new_h, new_w = max_image_size, int(w * max_image_size / h)
            else:
                new_h, new_w = int(h * max_image_size / w), max_image_size
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized preview image from {h}x{w} to {new_h}x{new_w}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error: Could not read {original_img_path}: {e}")
        return
        
    img_float = img.astype(np.float32) / 255.0
    
    # Configure augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), p=0.5),
    ])
    
    # Create figure for visualization
    plt.figure(figsize=(15, 8))
    
    # Plot original
    plt.subplot(2, 3, 1)
    plt.imshow(img_float)
    plt.title("Original")
    plt.axis('off')
    
    # Generate and plot augmentations
    for i in range(samples):
        plt.subplot(2, 3, i+2)
        
        # Apply augmentation
        augmented = transform(image=img)['image']
        aug_img = augmented.astype(np.float32) / 255.0
        
        # Apply astronomy processing - one technique per image
        enhancement = i % 3
        proc_name = "Basic Augmentation"
        
        if enhancement == 0:
            # Log transformation
            aug_img = exposure.adjust_log(aug_img, 1.0)
            proc_name = "Log Transform"
        elif enhancement == 1:
            # Gamma correction
            aug_img = exposure.adjust_gamma(aug_img, 1.2)
            proc_name = "Gamma Correction" 
        elif enhancement == 2:
            # Process each channel separately to avoid memory issues
            for c in range(aug_img.shape[2]):
                aug_img[:,:,c] = exposure.equalize_hist(aug_img[:,:,c])
            proc_name = "Hist Equalization"
            
        plt.imshow(aug_img)
        plt.title(f"Aug {i+1}: {proc_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "augmentation_preview.jpg"))
    plt.show()

def main():
    # Example usage
    input_directory = "C:\\Users\\SWARAJ PAWAR\\Downloads\\me-revieve\\NGC188"  # Change to your input directory
    output_directory = "C:\\Users\\SWARAJ PAWAR\\Downloads\\me-revieve\\NGC188\\augmented_galaxies"  # Where augmented images will be saved
    
    print("Starting galaxy image augmentation...")
    create_augmented_dataset(input_directory, output_directory, samples_per_image=15, max_image_size=1024)
    print(f"Augmentation complete. Check {output_directory} for results.")
    
    # Optional: Preview augmentations on a single image
    # Replace with path to one of your galaxy images
    # visualize_augmentations("galaxy_images/example_galaxy.jpg")

if __name__ == "__main__":
    main()