import numpy as np 
import sys 
from PIL import Image
import os
data = np.load(sys.argv[1])
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)
images = data['arr_0']
for i in range(10):
    img = Image.fromarray(images[i].astype('uint8'), 'RGB')
    img.save(f"{output_dir}/image_{i+1}.png")
