# Computer Vision Blog Post — Group 17

**Team Members:** Hruday Ramachandra, Mateusz Rębacz, Vishnu Karthik Ramesh Selvam

📂 **GitHub Repository:** [Puzzle-solver](https://github.com/HrudayR/Puzzle-solver/tree/master)

---



## Table of Contents

1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Method](#method)
4. [Experiments and Results](#experiments-and-results)
5. [Discussion](#discussion)
6. [Bibliography](#bibliography)

---

## Introduction

Imagine a situation where you are solving a 1000-piece jigsaw puzzle with no reference image. How would you decide which pieces fit together? You would likely examine the shape of each piece's boundary and the visual texture along its edges, combining both signals to make a judgment. Sounds simple enough, right? Yet, when you try to teach a computer to do the same, the problem reveals a surprising depth.

As humans, we bring intuition, spatial reasoning, and years of visual experience to the table. A machine, however, only has access to the raw pixel values it is provided. Machines cannot glance at a puzzle piece and instantly recognise which ones belong together. They must learn from scratch what makes two edges compatible. Then, machines search through a large set of possible arrangements to find the correct one. Solving this problem reliably has direct practical value. This is true for cases such as restoring damaged ancient frescoes [1], reassembling fragmented archaeological artefacts [2], and reconstructing shredded forensic evidence [3]. In these settings, automated solving is vital because the original context has been lost and cannot be recovered by hand.

So how do you even begin to teach a machine to do this? It turns out the problem splits into two distinct hurdles [4]. The first is **compatibility**. Given two pieces, how do you decide whether their edges belong together? This is trickier than it sounds, as edges can look deceptively similar by chance, and even a single wrong match early on can silently corrupt the entire reconstruction. The second hurdle is **scale**. Even if you can measure compatibility perfectly, finding the correct arrangement out of all possible orderings is an astronomically large search problem. You cannot simply try every combination: with N pieces there are N! possible arrangements, a number that grows so fast that an exhaustive search quickly becomes infeasible.

Here is where things get interesting. Recent methods, such as [4], have made substantial progress on the second hurdle. They use Transformer architectures to reason about all pieces simultaneously, rather than solving them one at a time. However, they overcome the first hurdle by taking a shortcut. They assume all pieces are perfect squares, which is far from the truth. Real jigsaw pieces have irregular, interlocking edges whose shapes carry the precise compatibility signal a machine needs. By ignoring the shape, current methods are leaving their best clue on the table. This omission is particularly problematic because even a one-pixel erosion at the piece boundaries can lead to a significant drop in a solver's reconstruction accuracy [4].

In this work, we pose a simple question: **what happens if you give the machine that clue back?** To answer this question, we extend the framework proposed by Heck et al. [4] to handle true jigsaw-shaped pieces and run five experiments.

We start from the baseline proposed in [4] on curved pieces to establish how much the square assumption costs in practice. We then introduce our encoder, which combines Fourier descriptors for boundary shape with either fixed Gabor kernels or a learnable CNN for edge texture. To understand the role of the solver, we pair each encoder variant with both a shallow neural network and the full Transformer from [4]. In total, this gives us five configurations to compare.

These experiments are designed to answer three concrete questions:

- Does replacing the square-piece encoder with a shape-aware, texture-rich encoder improve reconstruction accuracy on curved pieces?
- Does a learnable CNN capture more useful compatibility information than fixed Gabor kernels?
- Does the Transformer solver unlock the full potential of our encoder, or does a simpler solver perform just as well?

---

## Related Work

Automated puzzle solving evolved from a 1964 geometric-only approach [5] to the first use of visual content for reconstruction in 1994 [6]. Traditional greedy algorithms were often brittle because sequential piece placement allowed single mismatches to propagate throughout the entire reconstruction [7]. This brittleness motivated a shift toward global reasoning, where techniques like linear programming consider all pieces simultaneously rather than one at a time. Modern deep learning models, such as Twin Embedding Networks [8] and generative adversarial networks [9], further increased robustness against degraded data like eroded edges. These advancements established the foundation for Vision Transformers, which currently provide a global, permutation-based solution to piece placement.

Heck et al. [4] represent the current state of the art in this paradigm, and their approach forms the direct foundation of our work. Their framework treats reconstruction as a permutation problem: given N scrambled pieces, find the correct assignment of each piece to its position in the grid. A CNN-based edge encoder processes each piece by extracting four thin pixel strips — one per edge — through a shared network, pulling compatible edges closer together in embedding space and pushing incompatible ones apart. The four resulting vectors are concatenated into a single piece representation and passed to a Vision Transformer, which uses multi-head self-attention to capture global context across all pieces simultaneously. The raw Transformer outputs are then passed through a Sinkhorn–Knopp normalisation layer to produce a differentiable soft permutation matrix, enabling end-to-end training.

The results are compelling: the architecture achieves a 50% accuracy gain against previous methods on puzzles with eroded fragment edges [4].

The framework has one major limitation, however. By assuming square pieces, it discards the geometric signal that our encoder is designed to capture. Our experiments directly quantify how much this assumption costs, and whether replacing it with shape-aware, texture-rich features improves reconstruction accuracy on true jigsaw-shaped pieces.

---

## Method

Our method follows the same two-stage pipeline as [4]. An encoder first produces a compact representation of each piece, and a Transformer solver then reasons about all pieces simultaneously to predict the correct arrangement. The only difference lies in the first stage: instead of raw pixel strips from square edges, our encoder captures boundary shape and edge texture.

### Pipeline Architecture
<p align="center">
<img width="685" height="745" alt="Screenshot 2026-04-07 033459" src="https://github.com/user-attachments/assets/b8195555-e9bd-4351-a547-1eab4474ebc7" /><br>
  <em>Figure 1: Full pipeline architecture. Each puzzle piece is passed through two parallel encoders, i) a learnable CNN for edge texture and ii) a Fourier descriptor for boundary shape, whose outputs are concatenated and projected into a shared embedding space. A Vision Transformer then reasons globally over all pieces via multi-head self-attention, producing a score matrix that is converted into a soft permutation matrix through Sinkhorn–Knopp normalisation.</em>
</p>

*Figure 1: Full pipeline architecture. B = Batch size, N = Number of pieces, D = Embedding dimension.*

### Boundary Shape: Fourier Descriptors

We model the outline of a jigsaw piece as a closed 2D curve. Fourier descriptors [10] decompose this curve into frequency components, producing a compact representation of the boundary. Smooth regions contribute low-frequency components, while sharp corners contribute high-frequency components. By discarding phase, we obtain **rotation invariance**.

This ensures that matching boundaries produce similar descriptors regardless of orientation, which is essential since pieces may appear at arbitrary angles. Unlike [4], which assumes square pieces and ignores boundary shape, we treat it as a primary signal.

The figure below illustrates the full Fourier descriptor pipeline — from binary shape, to boundary signal, to reconstruction using 20 descriptors, and finally to the normalised descriptor magnitudes that remain stable under rotation:
p align="center">
<img width="500" height="365" alt="image_fourier" src="https://github.com/user-attachments/assets/7555082c-43b5-4592-a408-199280b071a6" /><br>
  <em>Figure 2: The piece contour is converted into a complex boundary signal and decomposed into Fourier coefficients, capturing both global shape and fine detail. Using magnitude-only descriptors yields a compact, rotation-invariant representation for matching edges.</em>
</p>


The following figure shows an example of matching Fourier descriptors computed from a blueprint and a photograph of the same aircraft (F14 Tomcat), demonstrating that the descriptors generalise across representation styles:


<p align="center">
<img width="500" height="365" alt="fourier_descriptor" src="https://github.com/user-attachments/assets/8d17458a-b865-447d-938a-07f14ec8877c" /><br>
  <em>Figure 3: A shape is converted into a boundary curve, expanded into Fourier coefficients, and reduced to magnitude-based descriptors. These provide a compact, rotation-invariant representation that preserves the overall structure of the original shape.</em>
</p>


### Edge Texture: Fixed Gabor Filters or Learnable CNN

Fourier descriptors capture shape but not visual appearance. To encode colour, gradients, and texture along edges, we explore two approaches.

The **first** uses Gabor filters [11], which detect spatial frequencies and orientations. Applying a bank of filters at multiple scales yields a rich, fixed texture representation without training.

The **second** uses a learnable CNN operating on the piece image. This does not replace Fourier descriptors but complements them, learning visual texture features that are concatenated with the shape representation.

Unlike [4], where a single CNN must learn both shape and texture, our approach separates these roles. Shape is handled explicitly by Fourier descriptors, allowing the CNN to focus solely on texture, leading to more discriminative embeddings.

### The Encoder

Both approaches produce a single embedding per piece by concatenating Fourier descriptors with texture features (Gabor or CNN). The encoder is trained using **triplet loss**.

Given an anchor **a**, a positive **p** (compatible edge), and a negative **n** (incompatible edge), the loss is:

$$\mathcal{L}_{\text{triplet}} = \max\!\left(0,\; \|\mathbf{a} - \mathbf{p}\|_2^2 - \|\mathbf{a} - \mathbf{n}\|_2^2 + \alpha\right)$$

This enforces that compatible edges are closer in embedding space than incompatible ones by a margin α. Over many triplets, the encoder learns to cluster matching edges and separate mismatched ones.

In the Gabor case, features are fixed and only the solver is trained. In the CNN case, the encoder is updated jointly. After training, embeddings are frozen and passed to the solver.

### The Solver

Once embeddings are computed, the solver predicts how pieces fit together using a **Vision Transformer**. Since puzzles have no inherent ordering, the permutation-invariant nature of Transformers is well suited to the task. Self-attention allows every piece to reason about every other piece in a single pass, building a global representation.

The solver operates in three steps:

1. Piece embeddings are treated as tokens and fed into the Transformer without positional encoding.

2. Multi-head self-attention [12] produces a score matrix:

$$\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

yielding **S** ∈ ℝ^(N×N), where S_ij measures how likely piece i belongs to position j.

3. Sinkhorn–Knopp normalisation [13] converts **S** into a soft permutation matrix:

$$\mathbf{P}^{(t+1)} = \text{normalise\_cols}\!\left(\text{normalise\_rows}\!\left(\mathbf{P}^{(t)}\right)\right), \quad \mathbf{P}^{(0)} = \exp(\mathbf{S})$$

After several iterations, **P** approximates a valid assignment. Because this step is differentiable, gradients propagate through the entire pipeline, enabling end-to-end training.

Keeping the solver identical to [4] ensures that any performance differences are attributable solely to improvements in the encoder.

---

## Experiments and Results

### Dataset Generation

Our dataset is built on the ILSVRC ImageNet training set, chosen for its visual diversity across 1,000 object categories and 1,281,167 training images, ensuring the encoder learns to prioritise geometric and textural boundary continuity rather than overfitting to a specific semantic domain. Dataset construction follows a two-stage pipeline:

**Algorithm: Curved Jigsaw Dataset Generation**

```
Input:  Image set D, target size (W, H), pieces per image N
Output: Normalised puzzle pieces

1. Preprocess Images
   For each image I in D:
     - Resize I to cover (W, H) while preserving aspect ratio
     - Centre-crop to (W, H) and convert to RGB if needed

2. Setup Grid and Tabs
   For each processed image I:
     - Initialise deterministic seed from filename
     - Compute grid (rows, cols) from N
     - Determine tile size (p_w, p_h)
     - Assign tab directions between adjacent tiles

3. Construct Pieces
   For each grid cell:
     - Define piece boundary:
         straight edges at borders, curved tab edges otherwise
     - Render piece on padded canvas using boundary mask
     - Extract corresponding image region

4. Normalise Pieces
   - Compute maximum piece dimensions
   For each piece:
     - Centre on a zero-padded canvas of uniform size
     - Save as RGBA image
```

The scale factor in Stage 1 guarantees full coverage of the target canvas before cropping, avoiding any padding or distortion in the source images. In Stage 2, the tab direction grid is sampled once per image from a filename-derived seed, making every cut pattern unique but fully reproducible across runs. Each interlocking boundary is assembled from five chained cubic Bézier curves that together trace the characteristic tab-and-blank profile of a real jigsaw edge: a straight approach, a narrowing neck, a rounded tab apex, a mirrored neck exit, and a straight departure. The canvas padding of 0.35 × max(p_w, p_h) ensures that tabs protruding beyond the nominal tile boundary are never clipped. Finally, all pieces are normalised to a uniform canvas size (W_max, H_max) by centring each piece within a transparent background, enabling the encoder to process every piece as a fixed-size tensor while clearly distinguishing piece geometry from empty space.

**Source image before puzzle cutting.** The image is first scaled and centre-cropped to the target resolution so that all pieces derive from a uniform canvas, preventing any piece from containing padding artefacts.

<p align="center">
<img width="500" height="365" alt="Image_before" src="https://github.com/user-attachments/assets/38712916-91f6-497b-86fd-74b7e96012ad" />
<br>
  <em>Figure 4: Source image before puzzle cutting. The image is first scaled and
    centre-cropped to the target resolution so that all pieces derive from a
    uniform canvas, preventing any piece from containing padding artefacts.</em>
</p>

**Curved jigsaw pieces produced by the Bézier cutting algorithm.** Each boundary is constructed from five chained cubic Bézier segments forming a tab-and-blank profile. Pieces are padded to a uniform canvas size and surrounded by a transparent background so that the encoder can distinguish piece geometry from empty space.

<p align="center">
<img width="500" height="365" alt="Image_after" src="https://github.com/user-attachments/assets/c3610a1c-c735-4369-9b56-beb7d532384a" />
<br>
  <em>Figure 5: Curved jigsaw pieces produced by the Bezier cutting algorithm.
    Each boundary is constructed from five chained cubic Bezier segments
    forming a tab-and-blank profile. Pieces are padded to a uniform canvas size
    and surrounded by a transparent background so that the encoder can
    distinguish piece geometry from empty space.</em>
</p>

### Results

| Exp. | Encoder | Texture Features | Solver | Matching (%) |
|------|---------|-----------------|--------|:------------:|
| 1 | Square CNN (baseline) | Raw pixel strips | Transformer | 53 |
| 2 | Fourier + Gabor | Fixed Gabor bank | Shallow NN | 32 |
| 3 | Fourier + Gabor | Fixed Gabor bank | Transformer | 58 |
| 4 | Fourier + CNN | Learnable CNN | Shallow NN | 44 |
| 5 | Fourier + CNN | Learnable CNN | Transformer | **70** |

*Matching accuracy across all five experimental configurations. Each row isolates one variable relative to the others. The full method (Experiment 5) achieves the highest accuracy.*

#### Experiment 1: The Baseline

We begin by running the Heck et al. encoder unchanged on our curved dataset. The baseline achieves a matching accuracy of **0.53**. This establishes an important reference point: a state-of-the-art square-piece encoder, when confronted with true jigsaw-shaped pieces for the first time, already loses nearly half of all placements. This confirms that the square-piece assumption is not a harmless simplification. When the encoder has no mechanism to reason about boundary shape, curved piece geometry actively hurts accuracy.

#### Experiment 2: Fixed Features + Fourier Descriptors, Shallow Solver

Replacing the square CNN with hand-crafted Fourier descriptors and fixed Gabor kernels, paired with a shallow solver, yields a matching accuracy of **0.32**. The shallow solver is a weak reasoner: it cannot capture the global dependencies between pieces that make reconstruction tractable. The low score tells us less about the quality of the features themselves and more about the bottleneck introduced by pairing rich geometric features with an underpowered solver. Crucially, this experiment isolates the solver as the limiting factor, which Experiment 3 directly addresses.

#### Experiment 3: Fixed Features + Fourier Descriptors, Transformer Solver

Swapping the shallow solver for the full Transformer while keeping the fixed encoder unchanged produces a matching accuracy of **0.58**, which is a gain of 0.26 over Experiment 2 and a marginal improvement of 0.05 over the baseline. Two conclusions follow. First, the Transformer is doing substantial work: the same fixed features that scored 0.32 with a shallow solver reach 0.58 when the solver can reason globally. Second, hand-crafted Fourier and Gabor features are already competitive with the baseline square CNN when paired with a strong solver, suggesting that boundary shape and fixed texture responses carry genuine compatibility signal.

#### Experiment 4: Learnable CNN + Fourier Descriptors, Shallow Solver

Introducing a learnable CNN alongside the Fourier descriptors, but retaining the shallow solver, achieves a matching accuracy of **0.44**, which is a clear improvement of 0.12 over Experiment 2. The gap is entirely attributable to the CNN learning a task-specific texture representation on top of the fixed Fourier signal. Because boundary shape is already captured by the Fourier descriptors, the CNN can focus exclusively on residual texture, and this focused learning is more effective than the fixed Gabor bank. The accuracy remains below the baseline, however, reinforcing that the shallow solver is still the primary bottleneck.

#### Experiment 5: Learnable CNN + Fourier Descriptors, Transformer Solver

Our full proposed method, combining the learnable CNN encoder with the Transformer solver, achieves a matching accuracy of **0.70**. This is the strongest result across all five configurations and represents:
- A gain of **0.17** over the baseline (Experiment 1)
- A gain of **0.26** over the same encoder paired with a shallow solver (Experiment 4)
- A gain of **0.12** over the best fixed-feature configuration (Experiment 3)

The result answers all three of our research questions affirmatively: a shape-aware, texture-rich encoder outperforms the square-piece baseline; a learnable CNN outperforms fixed Gabor kernels; and the Transformer solver is necessary to unlock the full benefit of the richer encoder.

---

## Discussion

The five experiments tell a coherent and interpretable story. Two independent factors drive reconstruction accuracy on curved jigsaw pieces: the richness of the encoder and the reasoning capacity of the solver. Neither alone is sufficient; their combination is what delivers the headline improvement.

**The square-piece assumption is a genuine bottleneck.**
Experiment 1 establishes that the Heck et al. encoder, despite being state-of-the-art on square pieces, achieves only 0.53 on curved pieces. This confirms the central motivation of our work: ignoring boundary shape is not a neutral design choice. When piece edges are irregular and interlocking, a square-piece encoder must rely entirely on texture, and texture alone is ambiguous enough to produce nearly one misplacement in every two.

**Boundary shape and fixed texture are already informative.**
Experiments 2 and 3 together show that Fourier descriptors and fixed Gabor kernels, requiring no training, carry real compatibility signal. Once paired with the Transformer solver, these hand-crafted features match the baseline accuracy (0.58 vs. 0.53). This is a notable result: geometric and textural features derived entirely by design, with no learning, are competitive with a trained square-piece CNN.

**Learning refines what geometry cannot fully specify.**
The jump from Experiment 3 (0.58) to Experiment 5 (0.70) isolates the contribution of the learnable CNN. Fourier descriptors capture boundary shape, but they are insensitive to fine-grained visual content running along the edge such as colour transitions, surface texture, and local contrast patterns that differ between puzzle images. The learnable CNN fills this gap by discovering the specific textural cues most predictive of compatibility on this dataset. The 0.12 gain over fixed Gabor features confirms that task-specific learning adds discriminative power that no fixed filter bank can replicate.

**The Transformer solver is necessary, not optional.**
Comparing Experiments 2 and 3 and Experiments 4 and 5 reveals a consistent gain of roughly 0.26 from the Transformer, regardless of which encoder is used. Puzzle reconstruction is an inherently global problem: the correct position of any single piece is only meaningful relative to all others. A shallow solver that reasons locally cannot resolve this interdependency, whereas the Transformer's self-attention mechanism considers every piece against every other in a single pass. The solver is not an interchangeable component. Rather, it is a structural requirement for competitive performance.

**Limitations and future work.**
Despite the improvement over the baseline, a matching accuracy of 0.70 leaves substantial room for further progress. Several directions are worth exploring:

- Our Fourier descriptor is computed on the full piece boundary, but compatibility is a local property of a single edge; descriptors computed edge-by-edge may provide a sharper signal.
- The learnable CNN is trained with triplet loss on randomly sampled negatives; hard negative mining strategies could accelerate learning and improve the quality of the embedding space.
- Our dataset uses a fixed number of pieces per image; evaluating on puzzles of varying size and aspect ratio would test the generality of our approach.
- Extending the pipeline to heavily fragmented pieces would determine how much of our gain is specific to the tab-and-blank structure of curved cuts versus a more general improvement in edge representation.

---

## Bibliography

[1] T. S. Teixeira, M. L. S. C. de Andrade, and M. R. da Luz, "Reconstruction of frescoes by sequential layers of feature extraction," *Pattern Recognition Letters*, vol. 147, pp. 172–178, 2021. doi: [10.1016/j.patrec.2021.04.012](https://doi.org/10.1016/j.patrec.2021.04.012)

[2] D. Gigilashvili, H. Lukesova, C. F. Gulbrandsen, A. Harijan, and J. Y. Hardeberg, "Computational techniques for virtual reconstruction of fragmented archaeological textiles," *Heritage Science*, vol. 11, no. 1, p. 259, 2023. doi: [10.1186/s40494-023-01102-3](https://doi.org/10.1186/s40494-023-01102-3)

[3] E. Justino, L. S. Oliveira, and C. Freitas, "Reconstructing shredded documents through feature matching," *Forensic Science International*, vol. 160, no. 2, pp. 140–147, 2006. doi: [10.1016/j.forsciint.2005.09.001](https://doi.org/10.1016/j.forsciint.2005.09.001)

[4] G. Heck, N. Lermé, and S. Le Hégarat-Mascle, "Solving jigsaw puzzles with vision transformers," *Pattern Analysis and Applications*, vol. 28, no. 2, p. 110, 2025. doi: [10.1007/s10044-025-01484-z](https://doi.org/10.1007/s10044-025-01484-z)

[5] H. Freeman and L. Garder, "Apictorial jigsaw puzzles: the computer solution of a problem in pattern recognition," *IEEE Transactions on Electronic Computers*, vol. EC-13, no. 2, pp. 118–127, 1964. doi: [10.1109/PGEC.1964.263781](https://doi.org/10.1109/PGEC.1964.263781)

[6] D. A. Kosiba, P. M. Devaux, S. Balasubramanian, T. L. Gandhi, and R. Kasturi, "An automatic jigsaw puzzle solver," in *Proc. 12th International Conference on Pattern Recognition (ICPR)*, vol. 1, pp. 616–618, 1994. doi: [10.1109/ICPR.1994.576377](https://doi.org/10.1109/ICPR.1994.576377)

[7] A. C. Gallagher, "Jigsaw puzzles with pieces of unknown orientation," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2012, pp. 382–389. doi: [10.1109/CVPR.2012.6247695](https://doi.org/10.1109/CVPR.2012.6247695)

[8] D. Rika, D. Sholomon, O. E. David, and N. S. Netanyahu, "TEN: Twin embedding networks for the jigsaw puzzle problem with eroded boundaries," in *Proc. IEEE International Conference on Image Processing (ICIP)*, pp. 4083–4087, 2022. arXiv: [2203.06488](https://arxiv.org/abs/2203.06488)

[9] R. Li, S. Liu, G. Wang, G. Liu, and B. Zeng, "JigsawGAN: Auxiliary learning for solving jigsaw puzzles with generative adversarial networks," *IEEE Transactions on Image Processing*, vol. 31, pp. 513–524, 2022. doi: [10.1109/TIP.2021.3120052](https://doi.org/10.1109/TIP.2021.3120052)

[10] F. P. Kuhl and C. R. Giardina, "Elliptic Fourier features of a closed contour," *Computer Graphics and Image Processing*, vol. 18, no. 3, pp. 236–258, 1982. doi: [10.1016/0146-664X(82)90034-X](https://doi.org/10.1016/0146-664X(82)90034-X)

[11] J. G. Daugman, "Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters," *Journal of the Optical Society of America A*, vol. 2, no. 7, pp. 1160–1169, 1985. doi: [10.1364/JOSAA.2.001160](https://doi.org/10.1364/JOSAA.2.001160)

[12] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[13] R. Sinkhorn and P. Knopp, "Concerning nonnegative matrices and doubly stochastic matrices," *Pacific Journal of Mathematics*, vol. 21, no. 2, pp. 343–348, 1967. doi: [10.2140/pjm.1967.21.343](https://doi.org/10.2140/pjm.1967.21.343)

[14] Zhihu article on Fourier descriptors. [https://zhuanlan.zhihu.com/p/37031152](https://zhuanlan.zhihu.com/p/37031152)
