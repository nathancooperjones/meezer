# meezer
Supervised Siamese networks using hard negative examples.

![](https://i.redd.it/fq6l78zdlow21.jpg)

A good deal of this code, especially files dealing with building and extracting information from the `annoy` index, is based off of [Ivis](https://github.com/beringresearch/ivis).

## Usage

### Development

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
