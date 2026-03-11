# MaskFormerFusion Architecture

This document sketches the high‑level architecture of the **MaskFormerFusion** model used in the repository. The diagram emphasises the multi‑scale backbone, pixel decoder, query‑based transformer, and early fusion strategy.

## MaskFormerFusion

```mermaid
flowchart TB
    subgraph Input
        RGB["RGB Image"]
        LIDAR["LiDAR Image"]
    end

    %% Early fusion
    RGB & LIDAR -->|elementwise add| Backbone["Backbone (timm)\nfeatures_only=True"]

    %% Backbone outputs
    Backbone --> F1["scale1: C1, H1,W1"]
    Backbone --> F2["scale2: C2, H2,W2"]
    Backbone --> F3["scale3: C3, H3,W3"]
    Backbone --> F4["scale4: C4, H4,W4"]

    %% Pixel decoder (FPN-style)
    subgraph PixelDecoder[PixelDecoder]
        F1 --> Lat1["1x1 conv→256"]
        F2 --> Lat2["1x1 conv→256"]
        F3 --> Lat3["1x1 conv→256"]
        F4 --> Lat4["1x1 conv→256"]
        Lat4 --> Up3["↑bilinear + add Lat3"]
        Up3 --> Up2["↑bilinear + add Lat2"]
        Up2 --> Up1["↑bilinear + add Lat1"]
        Up1 --> OutPD["output_conv → 256@H1×W1"]
    end

    %% Transformer decoder with queries
    OutPD --> TransformerDecoder["TransformerDecoder\n(num_queries=Q)\n→ class logits & mask logits per query"]

    %% Merge masks into dense segmap
    TransformerDecoder --> Merge["segmap = Σ_q softmax(cls)[c] × sigmoid(mask_q"]
    Merge --> Output["↑bilinear to input resolution"]
```

**Notes:**
- Backbone outputs are lists of feature maps at progressively coarser resolutions.
- Early fusion (RGB + LiDAR) is performed immediately after the backbone.
- Pixel decoder reduces multi-scale maps to a single high-resolution feature map.
- Transformer decoder uses a fixed set of learnable query embeddings to predict class scores and masks; deep supervision returns outputs from all decoder layers.
- Inference merges query predictions using the MaskFormer formula into a dense segmentation map.

