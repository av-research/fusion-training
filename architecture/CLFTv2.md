# CLFTv2 Architecture

This document sketches the high‑level architecture of the **CLFTv2** model used in the repository. It builds on the original CLFT design by making the cross‑fusion path explicit and clarifying the resampling operations in each reassembly block.

```mermaid
flowchart TB
    %% Inputs
    subgraph Inputs
        RGB["RGB Image<br/>(3×H×W)"]
        LIDAR["LiDAR Point Cloud<br/>(X×H×W)"]
    end

    %% Backbone & hooks
    subgraph Backbone
        Transformer[timm Transformer encoder]
        Transformer -->|blocks 0..N| Hooks[registered hooks]
    end
    RGB -->|`modal='rgb'` or cross‑fusion| Transformer
    LIDAR -->|`modal='lidar'` or cross‑fusion| Transformer

    %% Stage loop
    subgraph Stage[i = last..0]
        direction TB
        Hooks --> Act["Tokens<br/>(B,L,D)"]

        subgraph ReRGB["Reassemble RGB"]
            Act --> ReadRGB["Read CLS token<br/>(ignore/add/proj)"]
            ReadRGB --> ConcatRGB["Tokens→map (B,D,H/p,W/p)"]
            ConcatRGB --> ResampleRGB["1×1 conv → ^D\n+ resize"]
        end
        subgraph ReXYZ["Reassemble XYZ"]
            Act --> ReadXYZ["Read CLS token"]
            ReadXYZ --> ConcatXYZ["Map reshape"]
            ConcatXYZ --> ResampleXYZ["Resample to ^D"]
        end

        subgraph CrossFusion["Cross‑modal Fusion"]
            ReRGB & ReXYZ --> Fusion["resConvRGB + resConvXYZ\n+ prev → resConv2 → ↑2"]
        end
        Fusion --> Prev["upsampled features"]
        Prev -->|next stage| Hooks
    end

    %% Outputs
    Prev --> Heads["Heads: depth/seg"]
    Heads --> Depth["HeadDepth: conv→interp→conv→sigmoid →1"]
    Heads --> Seg["HeadSeg: conv→interp→conv →C"]
    style CrossFusion fill:#f9f,stroke:#333,stroke-width:1px
```

### Component details

- The **cross‑fusion block** is boxed to emphasize RGB/XYZ merger; its internals show residual convs on each branch, addition with previous features, a second residual conv, and bicubic upsampling.
- **Reassemble** subgraphs break down into three explicit operations (read, concat, resample) with token and spatial dimensions annotated, aiding reproducibility.
- The `Stage` loop now displays the activation tensor shape and the backward iteration over hook indices, clarifying data flow during the forward pass.
- Output heads detail the convolutional pipelines, including interpolation steps and final channel counts.

> Although richer, the diagram still abstracts away low‑level parameter choices and training shortcuts; it is intended for inclusion in a methods section of a journal paper where readers need an intermediate level of granularity.
