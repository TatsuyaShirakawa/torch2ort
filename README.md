# torch2ort

This repository provide a snippet to convert [PyTorch](https://github.com/pytorch/pytorch) models to .onnx format and run them on the [onnxruntime](https://github.com/pytorch/pytorch) backend.

A small benchmarking script and its results are provided (up to around 4x speedup on cpu!)

# requirements

- Pytorch

# usage

```python
import torch2ort

model = ...  # your PyTorch model
sample_inputs = ...  # input sample for the model (model(*args) must be a valid invocation of the model)


# export the model parameter
torch2ort(model, sample_inputs, 'model.onnx')
```

Tips:

- if the model has multiple inputs/outputs, specify the `input_names`/`output_names`.
- if the resulting model should run with dynamic input sizes, specify the `dynamic_axes`.

see [the PyTorch's official document](https://pytorch.org/docs/stable/onnx.html?highlight=onnx%20export#torch.onnx.export) for detail.


# benchmarking result

`benchmark.py` is provided as a sample scripts and it's also used as a benchmarking PyTorch vs ONNX Runtiem.

```shell
$ PYTHONPATH=. python benchmark.py
```

The following is the result on my environment.

- MacBook Pro 2017
- PyTorch 1.3.1
- Torch 1.1.0

|model             |export to onnx|inference speedup|onnxruntime inference|onnxruntime loading|pytorch inference|pytorch loading|
|------------------|-------------:|----------------:|--------------------:|------------------:|----------------:|--------------:|
|AlexNet           |        1.5740|            1.445|              0.43757|            0.40803|           0.6323|        0.38255|
|DenseNet          |        5.4311|            1.636|              1.44570|            0.14502|           2.3646|        0.12408|
|GoogLeNet         |        1.7377|            2.761|              0.64903|            0.08658|           1.7920|        1.43373|
|Inception3        |        3.8546|            1.722|              1.35792|            0.30002|           2.3379|        2.18923|
|MobileNetV2       |        1.6284|            3.628|              0.39412|            0.06650|           1.4299|        0.08358|
|SqueezeNet        |        0.3354|            2.819|              0.34291|            0.01533|           0.9668|        0.02001|
|alexnet           |        1.4804|            1.388|              0.45147|            0.42214|           0.6267|        0.43909|
|densenet121       |        5.9478|            1.748|              1.51079|            0.15186|           2.6410|        0.15419|
|densenet161       |       10.1604|            1.935|              3.41980|            0.39313|           6.6163|        0.45145|
|densenet169       |        9.8547|            1.736|              1.80425|            0.25697|           3.1331|        0.26670|
|densenet201       |       15.6927|            1.956|              2.29748|            0.38666|           4.4929|        0.34543|
|googlenet         |        1.7677|            2.654|              0.68180|            0.08900|           1.8092|        1.13330|
|inception_v3      |        3.7513|            2.404|              1.30499|            0.46943|           3.1374|        2.04938|
|mnasnet0_5        |        1.7749|            1.243|              0.75621|            0.04164|           0.9399|        0.05891|
|mnasnet0_75       |        1.7429|            2.773|              0.45004|            0.07818|           1.2481|        0.10413|
|mnasnet1_0        |        1.2453|            1.652|              0.86568|            0.05251|           1.4305|        0.07240|
|mnasnet1_3        |        1.7538|            2.335|              0.84682|            0.08466|           1.9775|        0.11473|
|mobilenet_v2      |        1.8357|            4.967|              0.42690|            0.05401|           2.1206|        0.06863|
|resnet101         |        5.3256|            1.562|              2.92355|            0.52005|           4.5665|        0.67009|
|resnet152         |        9.1453|            1.518|              4.24764|            0.70319|           6.4487|        0.85021|
|resnet18          |        0.7750|            1.706|              0.76714|            0.20623|           1.3087|        0.19307|
|resnet34          |        1.2727|            1.572|              1.36226|            0.25520|           2.1421|        0.31072|
|resnet50          |        1.8907|            1.677|              1.65347|            0.28681|           2.7728|        0.34946|
|resnext101_32x8d  |        6.8531|            1.356|              7.19412|            1.12189|           9.7567|        1.26343|
|resnext50_32x4d   |        2.2067|            1.678|              1.95374|            0.28224|           3.2785|        0.37155|
|shufflenet_v2_x0_5|        1.9117|            4.696|              0.06995|            0.02667|           0.3285|        0.02303|
|shufflenet_v2_x1_0|        1.8652|            3.134|              0.16931|            0.04011|           0.5307|        0.03527|
|shufflenet_v2_x1_5|        1.7786|            2.455|              0.24195|            0.05760|           0.5941|        0.04503|
|shufflenet_v2_x2_0|        1.8294|            2.957|              0.35095|            0.08628|           1.0379|        0.06686|
|squeezenet1_0     |        0.3361|            3.032|              0.32825|            0.01229|           0.9954|        0.02565|
|squeezenet1_1     |        0.3363|            3.086|              0.18758|            0.01164|           0.5789|        0.01879|
|vgg11             |        6.7686|            1.483|              3.46716|            1.04234|           5.1434|        1.73912|
|vgg11_bn          |       14.3098|            1.743|              2.84201|            1.13846|           4.9522|        1.88943|
|vgg13             |       16.8019|            1.431|              4.44358|            0.97715|           6.3586|        1.85754|
|vgg13_bn          |        6.6337|            1.747|              4.17161|            1.18022|           7.2892|        2.07307|
|vgg16             |        2.2614|            1.630|              5.10733|            1.02900|           8.3257|        1.82539|
|vgg16_bn          |        2.5671|            1.812|              5.36224|            1.33612|           9.7173|        1.85792|
|vgg19             |        2.6076|            1.577|              6.55693|            1.06265|          10.3401|        1.97738|
|vgg19_bn          |        3.1829|            1.898|              7.45348|            2.18802|          14.1469|        2.45015|
|wide_resnet101_2  |        6.2256|            1.884|              8.28896|            2.60679|          15.6146|        1.86539|
|wide_resnet50_2   |        2.4820|            1.483|              4.33635|            0.81254|           6.4319|        0.89325|
