# Mask2FormerFusion Architecture

This document sketches the high‑level architecture of the **Mask2FormerFusion** model used in the repository. The figure highlights the deformable‑attention pixel decoder and multi‑scale masked‑attention transformer.

```mermaid
flowchart TB
    subgraph InputMF
        RGB2["RGB Image"]
        LIDAR2["LiDAR Image"]
    end

    RGB2 & LIDAR2 -->|fusion_res units per scale| Backbone2["Backbone (timm)\nfeatures_only=True"]
    Backbone2 --> B1["C1,H1,W1"]
    Backbone2 --> B2["C2,H2,W2"]
    Backbone2 --> B3["C3,H3,W3"]
    Backbone2 --> B4["C4,H4,W4"]

    %% MSDeformAttn pixel decoder
    subgraph MSDec[MSDeformAttnPixelDecoder]
        B1 --> P1["proj→256"]
        B2 --> P2["proj→256"]
        B3 --> P3["proj→256"]
        B4 --> P4["proj→256"]
        P4 --> U3["↑ + add P3"]
        U3 --> U2["↑ + add P2"]
        U2 --> U1["↑ + add P1"]
        U1 --> PixelFeat["pixel_features 256@H1×W1"]
        U1 --> MultiScale["multi-scale list"]
    end

    PixelFeat --> Proj["Conv1×1 → d_model"]
    MultiScale --> ScaleProj["projections → d_model each"]

    %% Mask2Former decoder
    Proj & ScaleProj --> Mask2Dec["Mask2FormerDecoder\n(masked-attn over multi-scale)\noutputs class & mask per query"]

    Mask2Dec --> Merge2["same merging formula → segmap"]
    Merge2 --> Up2["↑bilinear to input size"]
```

**Notes:**
- Fusion_res units apply a residual conv to each backbone output before elementwise addition.
- Pixel decoder is the deformable-attention variant from the official Mask2Former code; it produces both `pixel_features` and a multi-scale feature list used by the decoder.
- A simple 1×1 projection aligns channels to the transformer embedding dimension `d_model`.
- The Mask2Former decoder uses masked attention across scales and returns deep-supervision outputs.

> This file stands alone for clarity when discussing the Mask2Former design; see `MaskFormer.md` for the original MaskFormer architecture.