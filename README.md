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
$ PYTHONPATH=. python benchmark.py && python write_table.py > table.md
```

The following is the result on my environment.

- Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
- PyTorch 1.3.0
- ONNX Runtime 1.1.0

|                  |export to onnx|pytorch loading|onnxruntime loading|pytorch inference|onnxruntime inference|inference speedup|
|------------------|-------------:|--------------:|------------------:|----------------:|--------------------:|----------------:|
|AlexNet           |        0.9811|        0.38704|           0.368711|           0.3341|              0.28626|           1.1670|
|DenseNet          |        6.1644|        0.13341|           0.109668|           1.1213|              0.84672|           1.3243|
|GoogLeNet         |        1.7894|        1.15746|           0.039746|           1.1585|              0.33369|           3.4717|
|Inception3        |        4.1139|        1.93306|           0.208725|           1.1586|              0.67515|           1.7161|
|MobileNetV2       |        1.3818|        0.05635|           0.023670|           0.5208|              0.40347|           1.2908|
|SqueezeNet        |        0.2814|        0.01831|           0.007411|           0.5552|              0.21223|           2.6160|
|alexnet           |        0.9399|        0.34310|           0.356806|           0.3250|              0.28313|           1.1478|
|densenet121       |        6.1252|        0.11990|           0.102589|           1.1197|              0.87489|           1.2798|
|densenet161       |       11.2434|        0.35648|           0.324227|           2.3424|              1.88442|           1.2431|
|densenet169       |       11.5896|        0.20542|           0.207295|           1.5468|              1.06526|           1.4520|
|densenet201       |       16.2522|        0.26958|           0.288477|           1.8082|              1.39373|           1.2974|
|googlenet         |        1.8890|        1.07623|           0.042628|           1.1854|              0.36347|           3.2612|
|inception_v3      |        4.1494|        1.91553|           0.225508|           0.9857|              0.68757|           1.4335|
|mnasnet0_5        |        1.2311|        0.04131|           0.020212|           0.3349|              0.35842|           0.9345|
|mnasnet0_75       |        1.3830|        0.07538|           0.023905|           0.5092|              0.39709|           1.2823|
|mnasnet1_0        |        1.3296|        0.06462|           0.029505|           0.4595|              0.56778|           0.8093|
|mnasnet1_3        |        1.2824|        0.10391|           0.035885|           0.5171|              0.67696|           0.7638|
|mobilenet_v2      |        1.4721|        0.05825|           0.026072|           0.5295|              0.40873|           1.2954|
|resnet101         |        5.0237|        0.49024|           0.372793|           1.6979|              1.75562|           0.9671|
|resnet152         |        9.9498|        0.69135|           0.536964|           2.2374|              2.43889|           0.9174|
|resnet18          |        0.4048|        0.12684|           0.042950|           0.5068|              0.37365|           1.3563|
|resnet34          |        0.9720|        0.23753|           0.108595|           0.7009|              0.77761|           0.9014|
|resnet50          |        1.6556|        0.27452|           0.155284|           1.1346|              0.89928|           1.2617|
|resnext101_32x8d  |        5.6422|        0.95038|           0.552027|           3.0462|              3.42194|           0.8902|
|resnext50_32x4d   |        1.6075|        0.27081|           0.151423|           1.2252|              1.02581|           1.1944|
|shufflenet_v2_x0_5|        1.8927|        0.02315|           0.018735|           0.2448|              0.06722|           3.6411|
|shufflenet_v2_x1_0|        1.9067|        0.02840|           0.022499|           0.3453|              0.12990|           2.6581|
|shufflenet_v2_x1_5|        1.9033|        0.04769|           0.027738|           0.3676|              0.19524|           1.8829|
|shufflenet_v2_x2_0|        1.9355|        0.05266|           0.039547|           0.4784|              0.31223|           1.5321|
|squeezenet1_0     |        0.2926|        0.01919|           0.008320|           0.5753|              0.27552|           2.0880|
|squeezenet1_1     |        0.2910|        0.01884|           0.008188|           0.3095|              0.10929|           2.8316|
|vgg11             |        1.9824|        1.50417|           0.615598|           1.5383|              1.39883|           1.0997|
|vgg11_bn          |        2.1148|        1.50543|           0.631828|           1.7939|              1.40683|           1.2751|
|vgg13             |        2.0346|        1.46172|           0.637871|           1.8819|              2.00731|           0.9375|
|vgg13_bn          |        2.1282|        1.50485|           0.652634|           2.3694|              2.02672|           1.1691|
|vgg16             |        2.1572|        1.51867|           0.657221|           2.2836|              2.58640|           0.8829|
|vgg16_bn          |        2.2916|        1.56916|           0.673388|           2.4494|              2.54326|           0.9631|
|vgg19             |        2.3379|        1.62409|           0.675378|           3.1279|              3.11363|           1.0046|
|vgg19_bn          |        2.4367|        1.56127|           0.696483|           2.7655|              3.08861|           0.8954|
|wide_resnet101_2  |        6.2284|        1.35643|           0.939198|           3.6419|              3.89191|           0.9358|
|wide_resnet50_2   |        2.1915|        0.70986|           0.392496|           2.0093|              1.97991|           1.0148|
