# meezer
Supervised Siamese-ish networks using hard negative examples.

![](https://i.redd.it/fq6l78zdlow21.jpg)

A good deal of this code, especially files dealing with building and extracting information from the `Annoy` index, is based off of [Ivis](https://github.com/beringresearch/ivis). The `Img2Vec` code is based off of [this repo](https://github.com/jaredwinick/img2vec-keras).

## Usage
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

from meezer import Meezer


# GET DATA
digits = load_digits()

X, labels = digits.data, digits.target
labels = labels.astype(int)

# SCALE DATA
X_scaled = MinMaxScaler().fit_transform(X)

# TRAIN A MODEL
model = Meezer(embedding_dims=2,
               k=25,
               batch_size=512,
               epochs=30,
               sub_epochs=10)
embeddings = model.fit_transform(X=X_scaled, Y=labels)

# VISUALIZE EMBEDDINGS
plt.scatter(x=embeddings[:, 0],
            y=embeddings[:, 1],
            c=labels,
            cmap='tab10',
            s=0.5,
            alpha=0.7)

for i, mnist_num in enumerate(set(labels)):
    mnist_num_idxs = np.argwhere(labels == mnist_num)
    plt.scatter(embeddings[mnist_num_idxs, 0],
                embeddings[mnist_num_idxs, 1],
                color=plt.cm.gist_rainbow(i / len(set(labels))),
                label=mnist_num)

plt.xlabel('embedding dim 1')
plt.ylabel('embedding dim 2')
plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')

plt.show()
```

### Development
Begin by installing [Docker](https://docs.docker.com/install/) if you have not already. Once Docker is running, run development from within the Docker container:

```bash
# build the Docker image
docker build -t meezer .

# run the Docker container in interactive mode.
docker run \
    -it \
    --rm \
    -v "${PWD}:/meezer" \
    -p 8888:8888 \
    meezer /bin/bash

# then run JupyterLab and begin development
jupyter lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```
