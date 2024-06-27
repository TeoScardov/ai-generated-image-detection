import os
import zipfile
from PIL import Image
from io import BytesIO

src_dir = './genimage/dataset/temp'
dest_dir = './genimage_resized/dataset/midjourney'

for dirpath, _, filenames in os.walk(src_dir):
    for zip_filename in filenames:
        print(os.path.join(dirpath, zip_filename))
        with zipfile.ZipFile(os.path.join(dirpath, zip_filename), 'r') as z:
            for fileinfo in z.infolist():
                if fileinfo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Read the file as bytes, then open it as an image
                        with z.open(fileinfo) as file:
                            img = Image.open(file)
                            img.verify()  # Verify image integrity
                            file.seek(0)  # Reset file pointer after verify
                            #img = Image.open(file)
                            #img = img.convert('RGB')  # Convert to RGB

                            # Save the image as JPEG
                            output_path = os.path.join(dest_dir, os.path.basename(fileinfo.filename) + '.jpg')
                            #img.save(output_path, 'JPEG')

                            print(f"Processed and saved: {output_path}")

                    except (IOError, zipfile.BadZipFile) as e:
                        print(f"Corrupted image or bad zip: {fileinfo.filename} in {os.path.join(dirpath, zip_filename)}")

                break
        break
    break
print("Done!")
