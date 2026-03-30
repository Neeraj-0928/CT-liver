import pandas as pd
import os

# Load CSV
df = pd.read_csv(r"C:\Users\Neeraj S\OneDrive\Desktop\ctliver\hcc-data-complete-balanced.csv")

# Folder with images
img_dir = r"C:\Users\Neeraj S\OneDrive\Desktop\ctliver\dataset\div-images"

# Get all image filenames from train and test subdirectories (exclude _mask files)
img_files = []
for split in ['train', 'test']:
    split_dir = os.path.join(img_dir, split)
    if os.path.exists(split_dir):
        for f in sorted(os.listdir(split_dir)):
            if f.endswith('.png') and not f.endswith('_mask.png'):
                # Store as "split/filename" format
                img_files.append(os.path.join(split, f))

print(f"Found {len(img_files)} images")

# Make sure the number of images matches the number of rows in CSV
if len(img_files) != len(df):
    print(f"Warning: {len(img_files)} images vs {len(df)} CSV rows. Using min length.")
    min_len = min(len(img_files), len(df))
    df = df.iloc[:min_len]            # keep only matching rows
    img_files = img_files[:min_len]   # keep only matching images

# Add Image column
df["Image"] = img_files

# Save updated CSV
df.to_csv(r"C:\Users\Neeraj S\OneDrive\Desktop\ctliver\hcc-data-complete-balanced.csv", index=False)
print("CSV updated successfully!")
print(df.head())
