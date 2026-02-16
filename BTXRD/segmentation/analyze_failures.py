"""
Analyze Failed Segmentation Images (Dice = 0.0000)
===================================================
Investigate why certain images completely fail without medical knowledge.
Uses data-driven analysis to find patterns.
"""

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_validation_results():
    """Load the full validation results with Dice scores"""
    # You'll need to run inference again and save results to JSON
    # For now, we'll extract from the terminal output you showed
    
    # Failed images from your validation (Dice 0.0000)
    failed_images = [
        'IMG000554', 'IMG000284', 'IMG001557', 'IMG001656', 'IMG000724',
        'IMG000262', 'IMG000295', 'IMG000237', 'IMG000352', 'IMG001190',
        'IMG000536', 'IMG000144', 'IMG001806', 'IMG000385', 'IMG000546',
        'IMG001252', 'IMG001068', 'IMG001394', 'IMG000988', 'IMG000973',
        'IMG000741', 'IMG001034', 'IMG000721', 'IMG000691', 'IMG000524',
        'IMG000228', 'IMG000220', 'IMG000635', 'IMG001240', 'IMG001271',
        'IMG000845', 'IMG000456', 'IMG000650', 'IMG001753', 'IMG001859',
        'IMG000530', 'IMG000654', 'IMG001483', 'IMG000766', 'IMG001124',
        'IMG000529', 'IMG001654', 'IMG000309', 'IMG001699', 'IMG000523',
        'IMG000979', 'IMG000212', 'IMG001120', 'IMG000849', 'IMG000358',
        'IMG001319', 'IMG001021', 'IMG000707', 'IMG000825', 'IMG000693',
        'IMG000414', 'IMG000314', 'IMG000467', 'IMG000325', 'IMG000310',
        'IMG000491', 'IMG000910', 'IMG001751', 'IMG000370', 'IMG000924',
        'IMG000345', 'IMG000577', 'IMG001321'
    ]
    
    return failed_images

def analyze_image_properties(image_path):
    """Analyze basic image properties"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        return {
            'width': img.width,
            'height': img.height,
            'mean_brightness': img_array.mean(),
            'std_brightness': img_array.std(),
            'min_pixel': img_array.min(),
            'max_pixel': img_array.max(),
            'contrast': img_array.max() - img_array.min(),
        }
    except Exception as e:
        return None

def analyze_mask_properties(mask_path):
    """Analyze mask properties"""
    try:
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        
        # Threshold to binary
        binary_mask = (mask_array > 127).astype(np.uint8)
        tumor_pixels = binary_mask.sum()
        total_pixels = binary_mask.size
        tumor_ratio = tumor_pixels / total_pixels if total_pixels > 0 else 0
        
        return {
            'has_tumor': tumor_pixels > 0,
            'tumor_pixels': int(tumor_pixels),
            'tumor_ratio': float(tumor_ratio),
            'tumor_percentage': float(tumor_ratio * 100),
        }
    except Exception as e:
        return None

def find_image_and_mask(image_id, csv_path):
    """Find image and mask paths from CSV"""
    df = pd.read_csv(csv_path)
    row = df[df['image_path'].str.contains(image_id)]
    if len(row) > 0:
        return row.iloc[0]['image_path'], row.iloc[0]['mask_path']
    return None, None

def main():
    print("="*70)
    print("Analyzing Failed Segmentation Images")
    print("="*70)
    
    csv_path = Path('segmentation_val.csv')
    failed_images = load_validation_results()
    
    print(f"\nTotal failed images (Dice 0.0000): {len(failed_images)}")
    print(f"Total validation images: 187")
    print(f"Failure rate: {len(failed_images)/187*100:.1f}%\n")
    
    # Analysis results
    results = []
    missing_masks = []
    empty_masks = []
    image_properties = []
    mask_properties = []
    
    print("Analyzing each failed image...")
    for img_id in failed_images:
        img_path, mask_path = find_image_and_mask(img_id, csv_path)
        
        if img_path is None:
            print(f"  ⚠️  {img_id}: Not found in CSV")
            continue
        
        # Check if files exist
        img_path = Path(img_path)
        mask_path = Path(mask_path)
        
        if not img_path.exists():
            print(f"  ✗ {img_id}: Image file missing")
            continue
        
        if not mask_path.exists():
            print(f"  ✗ {img_id}: Mask file missing")
            missing_masks.append(img_id)
            continue
        
        # Analyze image
        img_props = analyze_image_properties(img_path)
        mask_props = analyze_mask_properties(mask_path)
        
        if img_props:
            img_props['image_id'] = img_id
            image_properties.append(img_props)
        
        if mask_props:
            mask_props['image_id'] = img_id
            mask_properties.append(mask_props)
            
            if not mask_props['has_tumor']:
                empty_masks.append(img_id)
                print(f"  ⚠️  {img_id}: Empty mask (no tumor annotation)")
            elif mask_props['tumor_percentage'] < 1.0:
                print(f"  ⚠️  {img_id}: Very small tumor ({mask_props['tumor_percentage']:.2f}%)")
    
    print("\n" + "="*70)
    print("Analysis Results")
    print("="*70)
    
    # Finding 1: Empty masks
    print(f"\n1. EMPTY MASKS (No tumor annotation):")
    print(f"   Count: {len(empty_masks)}/{len(failed_images)} ({len(empty_masks)/len(failed_images)*100:.1f}%)")
    if empty_masks:
        print(f"   ⚠️  CRITICAL: These images have ZERO tumor pixels in ground truth!")
        print(f"   This means Dice 0.0000 is CORRECT (true negatives).")
        print(f"   Examples: {', '.join(empty_masks[:5])}")
    
    # Finding 2: Very small tumors
    small_tumors = [m for m in mask_properties if 0 < m['tumor_percentage'] < 5]
    print(f"\n2. VERY SMALL TUMORS (<5% of image):")
    print(f"   Count: {len(small_tumors)}/{len(failed_images)} ({len(small_tumors)/len(failed_images)*100:.1f}%)")
    if small_tumors:
        print(f"   These are hard for the model to detect.")
        for m in small_tumors[:5]:
            print(f"   - {m['image_id']}: {m['tumor_percentage']:.2f}% tumor")
    
    # Finding 3: Image brightness analysis
    if image_properties:
        df_img = pd.DataFrame(image_properties)
        
        print(f"\n3. IMAGE BRIGHTNESS ANALYSIS:")
        print(f"   Mean brightness: {df_img['mean_brightness'].mean():.1f} ± {df_img['mean_brightness'].std():.1f}")
        print(f"   Contrast range: {df_img['contrast'].mean():.1f} ± {df_img['contrast'].std():.1f}")
        
        # Check for outliers (very dark or very bright)
        dark_images = df_img[df_img['mean_brightness'] < 50]
        bright_images = df_img[df_img['mean_brightness'] > 200]
        
        if len(dark_images) > 0:
            print(f"   ⚠️  Very dark images: {len(dark_images)}")
            print(f"      Examples: {', '.join(dark_images['image_id'].head(3).tolist())}")
        
        if len(bright_images) > 0:
            print(f"   ⚠️  Very bright images: {len(bright_images)}")
            print(f"      Examples: {', '.join(bright_images['image_id'].head(3).tolist())}")
    
    # Finding 4: Tumor size distribution
    if mask_properties:
        df_mask = pd.DataFrame(mask_properties)
        df_mask_with_tumor = df_mask[df_mask['has_tumor']]
        
        print(f"\n4. TUMOR SIZE DISTRIBUTION:")
        print(f"   Images with tumors: {len(df_mask_with_tumor)}/{len(failed_images)}")
        if len(df_mask_with_tumor) > 0:
            print(f"   Average tumor size: {df_mask_with_tumor['tumor_percentage'].mean():.2f}%")
            print(f"   Min tumor size: {df_mask_with_tumor['tumor_percentage'].min():.2f}%")
            print(f"   Max tumor size: {df_mask_with_tumor['tumor_percentage'].max():.2f}%")
    
    # Save detailed results
    print(f"\n" + "="*70)
    print("Saving Results...")
    print("="*70)
    
    output_dir = Path('failure_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON reports
    with open(output_dir / 'failed_images.json', 'w') as f:
        json.dump({
            'total_failed': len(failed_images),
            'failed_image_ids': failed_images,
            'empty_masks': empty_masks,
            'missing_masks': missing_masks,
        }, f, indent=2)
    
    if image_properties:
        pd.DataFrame(image_properties).to_csv(output_dir / 'image_properties.csv', index=False)
        print(f"✓ Saved: {output_dir / 'image_properties.csv'}")
    
    if mask_properties:
        pd.DataFrame(mask_properties).to_csv(output_dir / 'mask_properties.csv', index=False)
        print(f"✓ Saved: {output_dir / 'mask_properties.csv'}")
    
    # Create visualizations
    create_visualizations(image_properties, mask_properties, output_dir)
    
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if len(empty_masks) > 10:
        print(f"\n⚠️  CRITICAL FINDING:")
        print(f"   {len(empty_masks)} images have EMPTY ground truth masks!")
        print(f"   These are likely:")
        print(f"   - Healthy/normal images (no tumor)")
        print(f"   - Annotation errors")
        print(f"   \n   RECOMMENDATION: These should be REMOVED from validation set")
        print(f"   OR model should output empty predictions (which it doesn't).")
    
    if len(small_tumors) > 20:
        print(f"\n⚠️  Many images have very small tumors (<5% of image)")
        print(f"   These are genuinely hard to segment.")
        print(f"   RECOMMENDATION: Either:")
        print(f"   - Accept lower Dice on small tumors as expected")
        print(f"   - Add data augmentation focused on small objects")
        print(f"   - Use focal loss to handle class imbalance")
    
    print(f"\n✅ Analysis complete! Check '{output_dir}/' for detailed results.")

def create_visualizations(image_props, mask_props, output_dir):
    """Create analysis visualizations"""
    if not image_props or not mask_props:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Brightness distribution
    df_img = pd.DataFrame(image_props)
    axes[0, 0].hist(df_img['mean_brightness'], bins=30, edgecolor='black')
    axes[0, 0].set_title('Image Brightness Distribution')
    axes[0, 0].set_xlabel('Mean Brightness')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(df_img['mean_brightness'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Plot 2: Contrast distribution
    axes[0, 1].hist(df_img['contrast'], bins=30, edgecolor='black', color='green')
    axes[0, 1].set_title('Image Contrast Distribution')
    axes[0, 1].set_xlabel('Contrast (max - min)')
    axes[0, 1].set_ylabel('Count')
    
    # Plot 3: Tumor size distribution
    df_mask = pd.DataFrame(mask_props)
    df_mask_with_tumor = df_mask[df_mask['has_tumor']]
    if len(df_mask_with_tumor) > 0:
        axes[1, 0].hist(df_mask_with_tumor['tumor_percentage'], bins=30, edgecolor='black', color='orange')
        axes[1, 0].set_title('Tumor Size Distribution (% of image)')
        axes[1, 0].set_xlabel('Tumor Percentage')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].axvline(1.0, color='red', linestyle='--', label='1% threshold')
        axes[1, 0].legend()
    
    # Plot 4: Tumor vs No Tumor
    has_tumor = df_mask['has_tumor'].sum()
    no_tumor = len(df_mask) - has_tumor
    axes[1, 1].bar(['Has Tumor', 'Empty Mask'], [has_tumor, no_tumor], color=['orange', 'gray'])
    axes[1, 1].set_title('Ground Truth Mask Distribution')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'failure_analysis.png'}")
    plt.close()

if __name__ == '__main__':
    main()
