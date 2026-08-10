import os
import glob
from rembg import remove
from rembg.session_factory import new_session

download_dir = '/Users/anshuman/Downloads'
output_dir = os.path.join(download_dir, 'cleaned_poses')
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(download_dir, 'character1*.png')) + \
        glob.glob(os.path.join(download_dir, 'character2*.png')) + \
        glob.glob(os.path.join(download_dir, 'character1*.jpg')) + \
        glob.glob(os.path.join(download_dir, 'character2*.jpg'))

# Remove duplicates if any
files = list(set(files))
files.sort()

print(f"Found {len(files)} character images to clean.")

# Initialize the best U-2-Net model session
session = new_session("u2net")

for file_path in files:
    try:
        basename = os.path.basename(file_path)
        # Always output as png for transparency
        out_name = os.path.splitext(basename)[0] + '.png'
        out_path = os.path.join(output_dir, out_name)
        
        print(f"Processing {basename}...")
        with open(file_path, 'rb') as i:
            input_data = i.read()
            
            # Using alpha_matting for the highest precision ("clean between every single point")
            output_data = remove(
                input_data, 
                session=session, 
                alpha_matting=True, 
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10
            )
            
            with open(out_path, 'wb') as o:
                o.write(output_data)
                
        print(f"Saved cleaned version to {out_path}")
    except Exception as e:
        print(f"Failed to process {basename}: {e}")

print("\nAll poses have been perfectly cleaned!")
