import os
from PIL import Image

src_dir = './genimage/dataset/midjourney'
dest_dir = './genimage_resized/dataset/midjourney'
# Walk through the source directory
for dirpath, _, filenames in os.walk(src_dir):
    # Structure mirror: recreate the directory structure at the destination
    structure = os.path.join(dest_dir, os.path.relpath(dirpath, src_dir))
    if not os.path.isdir(structure):
        os.makedirs(structure)
        print(f'Creating directory {structure}')
    else:
        print(f'Directory {structure} already exists')
    
    # Process each file in the current directory
    for filename in filenames:
        # Check for image files 
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(dirpath, filename)
            output_path = os.path.join(structure, filename.split('.')[0])

            # Resize and save the image
            with Image.open(input_path) as img:
                # Resize the image
                img = img.resize((256, 256), Image.BICUBIC)
                # Convert to JPEG (if not already) and save
                img.convert('RGB').save(output_path+'.JPEG', format='JPEG')
print("Done!")