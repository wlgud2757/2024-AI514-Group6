import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# loda csv 
file_path = '/home2/s20235025/bigdata/GEMSEC/output/embeddings/tmp_GEMSEC.csv' 
data = pd.read_csv(file_path)

# t-sne
tsne = TSNE(n_components=2, random_state=42)
data_2d = tsne.fit_transform(data)

data['TSNE1'] = data_2d[:, 0]
data['TSNE2'] = data_2d[:, 1]

plt.figure(figsize=(10, 8))
plt.scatter(data['TSNE1'], data['TSNE2'], s=2, alpha=0.6)
plt.title('2D visualization of embeddings using t-SNE')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')

image_path = 'embedding_visualization_GEMSEC.png'
plt.savefig(image_path)