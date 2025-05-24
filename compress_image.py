import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

img = Image.open("/content/sample.jfif")
img = img.resize((200, 200))
img_np = np.array(img)
pixels=img_np.reshape(-1,3)

print(pixels)

k_vals=[2,8,16,32]
for i in k_vals:
  kmeans=KMeans(n_clusters=i,random_state=42)
  kmeans.fit(pixels)
  new_colors = kmeans.cluster_centers_[kmeans.labels_]
  compressed_img = new_colors.reshape(img_np.shape).astype(np.uint8)


  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  ax1.imshow(img_np)
  ax1.set_title("Original Image")
  ax1.axis("off")

  ax2.imshow(compressed_img)
  ax2.set_title(f"Compressed Image with {i} colors")
  ax2.axis("off")

  plt.show()
