# meezer
Supervised Siamese networks using hard negative examples.

![](https://i.redd.it/fq6l78zdlow21.jpg)

A good deal of this code, especially files dealing with building and extracting information from the `annoy` index, is based off of [Ivis](https://github.com/beringresearch/ivis).

## Usage
```python
# get data
from sklearn import datasets

digits = datasets.load_digits()

X, labels = digits.data, digits.target
labels = labels.astype(int)

# scale data
from sklearn.preprocessing import MinMaxScaler

X_scaled = MinMaxScaler().fit_transform(X)

# train model
from meezer import Meezer

model = Meezer(embedding_dims=2,
               k=25,
               batch_size=512,
               epochs=30,
               sub_epochs=10)
embeddings = model.fit_transform(X=X_scaled, Y=labels)

# visualize embeddings
import matplotlib.pyplot as plt

color = mnist.target.astype(int)

plt.scatter(x=embeddings[:, 0],
            y=embeddings[:, 1],
            c=labels,
            cmap='tab10',
            s=0.5,
            alpha=0.7)
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
```
