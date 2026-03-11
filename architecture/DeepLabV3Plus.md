# DeepLabV3+ Implementation

This document sketches the high‑level architectures of the **DeepLabV3+** and **DeepLabV3+ Late Fusion** models used in the repository. The diagrams are detailed enough for inclusion in a methods section of a paper.

```mermaid
flowchart TB
    %% Input
    Input["Input Image<br/>(3×H×W)"]

    %% Encoder backbone
    subgraph Encoder[ResNet-101 backbone]
        Conv1["conv1 7×7, /2"]
        BN1["bn1"]
        ReLU1["relu"]
        MaxPool["maxpool /2"]
        Layer1["layer1 (1/4, 256ch)"]
        Layer2["layer2 (1/8, 512ch)"]
        Layer3["layer3 (1/16, 1024ch)"]
        Layer4["layer4 atrous (1/16, 2048ch)"]
    end

    Input --> Conv1 --> BN1 --> ReLU1 --> MaxPool
    MaxPool --> Layer1 --> Layer2 --> Layer3 --> Layer4

    %% ASPP
    Layer4 --> ASPP["ASPP (out 256ch)"]

    %% Low-level connection
    Layer1 --> LowLevel["low-level feat (256ch)"]

    %% Decoder
    ASPP --> Decoder["Decoder"]
    LowLevel --> Decoder
    Decoder --> Head["classifier → logits (C)\n↑bilinear→H×W"]

    %% Late fusion variant
    subgraph LateFusion[DeepLabV3+ Late Fusion]
        RGBInput["RGB Input"]
        LIDARInput["LiDAR Input"]
        RGBBranch["DeepLabV3+\n(backbone→ASPP→decoder w/o final cls)"]
        LIDARBranch["DeepLabV3+\n(same structure) "]
        Fusion["Fusion (residual_avg)\n+ prev?"]
        SharedHead["shared classifier → C"]
        Upsample["↑bilinear→H×W"]

        RGBInput --> RGBBranch
        LIDARInput --> LIDARBranch
        RGBBranch --> Fusion
        LIDARBranch --> Fusion
        Fusion --> SharedHead --> Upsample
    end
```

### Component details

- **Encoder:** Standard ResNet‑101 layers; `layer4` uses atrous convolutions (dilation=2) to maintain resolution. Spatial strides are \( /2, /4, /8, /16 \) respectively.
- **ASPP:** Atrous Spatial Pyramid Pooling with rates {1,6,12,18} plus global pooling; outputs 256‑channel feature map.
- **Decoder:** Projects low‑level `layer1` features to 48 channels, upsamples ASPP output to match, concatenates, runs two 3×3 convs with batch norm/RELU/dropout, then final classifier and bilinear upsampling to input size.
- **Late fusion:** Two parallel DeepLabV3+ pipelines (RGB and LiDAR). Final classifiers removed, features fused via simple average (plus optional previous-stage addition), then a shared 1×1 conv classifier produces segmentation logits. Individual branch predictions are also computed for diagnostics.

> This figure abstracts away weight initialisation, batch‑norm details, and training hyperparameters. It is intended to communicate structural design to readers and collaborators.
