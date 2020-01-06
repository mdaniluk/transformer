**Simple Transformer implementation from scratch in PyTorch**

Goal of this implementation is to show the simplicity of transformer models and self-attention. This transformer model consists of stack of simple transformer blocks. It doesn't have encoder-decoder structure as historical transforer implementation.

`SelfAttention` is simple implementation of multi head attentnion module.

`TransformerBlock` is simple block that consists of attentnion layer, layer normalization and feed forward network with resnet connections between them.

`CTransformer` is designed for classifying sequences. It consists of several transformer blocks and takes the average of output tokens from last layer and apply linear projection to this final layer.

You can easily train it to classify sequences from IMDB dataset:
```
python classify.py
```
