---
source: mathdance.pdf
total_pages: 10
extracted_at: 2026-05-11T23:10:13.723424
images_dir: images
---

## MATHDance: Mamba-Transformer Architecture with Uniform Tokenization for High-Quality 3D Dance Generation

Kaixing Yang ∗

yangkaixing@ruc.edu.cn Renmin University of China Beijing, China

Yuxuan Hu huyuxuan1999@ruc.edu.cn Renmin University of China Beijing, China

Jun He †

hejun@ruc.edu.cn Renmin University of China Beijing, China

## Xulong Tang ∗

xulong.tang@maloutech.com Malou Tech Inc

Texas, USA

Xiangyue Zhang xiangyuezhang@whu.edu.cn Wuhan University

Wuhan, China

## Hongyan Liu †

liuhy@sem.tsinghua.edu.cn Tsinghua University Beijing, China

Stage1:Kinematic-Dynamic-basedQuantization Ziqiao Peng ∗

![Figure](images/figure_0014.png)

**[Image: figure_0014.png (972x215, 107.5KB)]**

pengziqiao@ruc.edu.cn Renmin University of China Beijing, China

Puwei Wang † wangpuwei@ruc.edu.cn Renmin University of China Beijing, China

Zhaoxin Fan † zhaoxinf@buaa.edu.cn Beihang University Beijing, China

Stage 2:Hybrid Music-to-Dance Generation

Figure 1: To enhance choreographic consistency, MATHDance designs High-Fidelity Dance Tokenization stage for physical plausibility and Hybrid Music-to-Dance Generation stage for aesthetic quality.

![Figure](images/figure_0019.png)

**[Image: figure_0019.png (997x221, 119.2KB)]**

## Abstract

Music-to-dance generation represents a challenging yet pivotal task at the intersection of choreography, virtual reality, and creative content generation. Despite its significance, existing methods face substantial limitation in achieving choreographic consistency. To address the challenge, we propose MatchDance, a novel framework for music-to-dance generation that constructs a latent representation to enhance choreographic consistency. MatchDance employs a two-stage design: (1) a Kinematic-Dynamic-based Quantization Stage (KDQS), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) with kinematicdynamic constraints and reconstructs them with high fidelity, and

∗ Equal Contribution.

† Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MM'26, Rio de Janeiro, Brazil

© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 978-1-4503-XXXX-X/2025/06

[https://doi.org/XXXXXXX.XXXXXXX](https://doi.org/XXXXXXX.XXXXXXX)

(2) a Hybrid Music-to-Dance Generation Stage(HMDGS), which uses a Mamba-Transformer hybrid architecture to map music into the latent representation, followed by the KDQS decoder to generate 3D dance motions. Additionally, a music-dance retrieval framework and comprehensive metrics are introduced for evaluation. Extensive experiments on the FineDance dataset demonstrate state-of-the-art performance. Code will be released upon acceptance.

## CCS Concepts

· Applied computing → Arts and humanities ; · Human-centered computing ; · Computing methodologies → Computer vision ; Animation ;

## Keywords

AI for Art, Multimedia Learning, AI Generative Content, 3D Human Motion Generation, Music-Driven Dance Generation

## ACMReference Format:

Kaixing Yang, Xulong Tang, Ziqiao Peng, Yuxuan Hu, Xiangyue Zhang, Puwei Wang, Jun He, Hongyan Liu, and Zhaoxin Fan. 2026. MATHDance: Mamba-Transformer Architecture with Uniform Tokenization for HighQuality 3D Dance Generation. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (MM'26). ACM, New York, NY, USA, 10 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

Music-to-dance generation is a crucial task that translates auditory input into dynamic motion, with significant applications in virtual reality, choreography, and digital entertainment [15, 18, 38]. By automating this process, it enables deeper exploration of the intrinsic relationship between audio and movement [28, 41, 43], while expanding possibilities for creative content generation[44, 45].

Current music-to-dance generation approaches have witnessed rapid progress and can be broadly categorized into two paradigms[27, 48]: (1) One-stage methods directly map musical features to human motion[7, 16, 17]. (2) Two-stage methods first construct choreographic units and then learn their probability distributions conditioned on music [26, 27, 39, 42 ? ]. However, previous methods suffer from issues with choreographic consistency, resulting in a lack of physical plausibility and poor aesthetic quality. For example, the generated motions often contain unnatural limb movements, and exhibit repetitive or overly stationary patterns. Simultaneously, current evaluation metrics fail to capture the artistic essence of music and dance, thereby misleading the generative process toward producing dances that lack choreographic consistency. For example, the mainstream FID, computed on handcrafted stylistic, kinetic, and geometric features [15, 18], primarily reflects surface-level motion statistics and lacks sensitivity to deeper choreographic semantics. BAS [15, 18], meanwhile, reduces music-dance correspondence to beat-level alignment, overlooking hierarchical rhythmic structures.

The core idea of MATHDance is to enhance choreographic consistency by decoupling it into two complementary aspects: physical plausibility and aesthetic quality. Physical plausibility refers to movements that follow natural body mechanics and remain smooth and stable over time. Aesthetic quality, on the other hand, focuses on how well the dance matches the music in rhythm, style, and structure. To address these, MATHDance adopts a two-stage architecture, as shown in Fig. 1. (1) High-Fidelity Dance Tokenization (HFDT) focuses on physical plausibility. We introduce Finite Scalar Quantization (FSQ) for dance tokenization. Unlike the learned vector quantization in VQ-VAE, FSQ applies fixed uniform scalar quantization, leading to stable token usage and improved latent expressiveness. To further promote spatio-temporal coherence, we impose spatial constraints via Forward Kinematics (FK) reconstruction, and temporal constraints by considering velocity and acceleration during reconstruction. Together, HFDT effectively constrains the latent motion space to physically plausible regions. (2) Hybrid Music-toDance Generation (HMDG) aims to enhance aesthetic quality. We propose a Mamba-Transformer hybrid architecture that combines Mamba's efficient modeling of local dependencies with the Transformer's capacity for global context. Moreover, we utilize a Sliding Window Attention mechanism instead of traditional Casual Attention mechanism, to better align long-term dance generation scene. Furthermore, we leverage the Music Foundation Model MuQ[47] learned from multi-level music informatics tasks, for powerful music representation. HMDG promotes choreography that aligns with music in rhythm, structure, and expression, supporting aesthetic quality.

On the other hand, we introduce a retrieval-based evaluation protocol, which leverages the intrinsic connection between retrieval and generation tasks and utilizes contrastive learning to encode cross-modal music-dance semantics. This protocol has shown strong effectiveness in similar domains such as text-to-motion[5], video-to-music[49], and text-to-video generation[19]. By training a retrieval model on real-world data, it effectively captures artistic essence of music and dance. Specifically, the model uses temporal processing and downsampling layers in the encoders, with CLIP loss [24] applied to align cross-modal features. Subsequently, the retrieval-based feature complements standard metrics like FID and DIV[15], providing a deeper assessment of generative models.

In the music-to-dance generation, the contributions of this work are summarized as follows: (1) We present MATHDance, a two-stage framework that enhances choreographic consistency by decoupling it into physical plausibility and aesthetic quality. Extensive experiments on AIST++ and FineDance demonstrate its superiority in both generation quality and computational efficiency. (2) We propose a dance tokenization method that introduces FSQ with spatialtemporal reconstruction constraints. Additionally, we design a dance token generation method that utilizes a Mamba-Transformer backbone and leverage the Music Foundation Model MuQ for powerful music representations. (3) We introduce a retrieval-based evaluation protocol, and conduct robust experiments to validate its reliability.

## 2 Related Work

## 2.1 One-Stage Music-to-Dance Generation

Music and dance are inherently connected, motivating research on music-driven 3D dance generation, where musical features are used to predict human motion. Early works adopt encoder-decoder frameworks to generate entire motion sequences [2, 10, 12, 15, 29]. In AIGC, Generative Adversarial Networks (GANs) have been applied to enhance realism in music-to-dance generation [1, 7, 36]. More recently, Diffusion Models achieved notable performance in this domain [9, 16-18, 30, 35], although their high sampling cost limits long-sequence generation efficiency. However, the above methods lack explicit spatial constraints, often leading to nonstandard poses that extend beyond the dancing subspace.

## 2.2 Two-Stage Music-to-Dance Generation

Two-stage approaches leverage the periodicity of dance by (1) quantizing motion into dance token, and (2) learning music-conditioned distributions over them. Since these token originate from real motion, such methods inherently favor physical plausibility. (1) Dance Tokenization Stage. Early works [1, 8, 40] rely on uniform segmentation, which is computationally inefficient. Later, VQ-VAE [3, 5] enables learnable unit construction with reduced cost. Bailando [27, 28] further decouples upper/lower body units to expand unit capacity. Recent works [16, 28, 34] use detailed SMPL parameter instead of traditional 3D keypoints. However, they treat all joints equally, ignoring kinematic hierarchy of human body. (2) Dance Generation Stage. Choreomaster [1] and DanceRevolution [7] adopt RNN variants, while later methods [27, 28] use cross-modal Transformers for improved temporal modeling and music-motion alignment.

In conclusion, existing methods lack choreographic consistency, primarily due to two limitations: (1) VQ-VAE-based tokenization suffers from low codebook utilization, weakening physical plausibility; (2) Transformer-only architectures rely on positional encodings, offering weak inductive biases for the continuous nature of music and dance, thereby undermining aesthetic quality. To enhance choreographic consistency, MATHDance designs High-Fidelity Dance Tokenization stage for physical plausibility and Hybrid Music-toDance Generation stage for aesthetic quality.

## 2.3 Evaluation for Music-to-Dance Generation

Designing objective quantitative metrics for music-to-dance generation remains challenging due to its high subjectivity (aesthetic discrepancy). Early approaches [14, 29] measure generation quality by computing MSE or MAE distances between generated and real dances. Subsequent works employ feature extractors to compute feature-level distances, using metrics such as FID for realism and DIV for motion diversity. Some methods [1] adopt selfreconstructing motion autoencoders as feature extractors, while others [11, 18] use genre classifiers trained on labeled dance data. Several approaches [15, 27, 30] extract handcrafted kinetic [23] and geometric [22] features to evaluate motion quality. Simultaneously, accurate modeling of music-dance correspondence is also crucial in this task. Existing BAS-based methods [7, 15] focus primarily on beat-level alignment, while overlooking the richer interplay between music and dance, including rhythmic structures and stylistic semantics.

In conclusion, current evaluation metrics fails to capture the artistic essence of music and dance, thereby misleading the generative process toward producing dances that lack choreographic consistency. To address this, we introduce a retrieval-based evaluation protocol, which leverages the intrinsic connection between retrieval and generation tasks and utilizes contrastive learning to encode cross-modal music-dance semantics.

## 3 Methodology

## 3.1 Problem Definition

Given a music sequence 𝑀 = { 𝑚 0 , 𝑚 1 , ..., 𝑚 𝑇 } and a dance genre label 𝑔 , the goal is to generate a corresponding dance sequence 𝐷 = { 𝑑 0 , 𝑑 1 , ..., 𝑑 𝑇 } . Each music feature 𝑚 𝑡 is a 1024-dimensional MuQ representation [47], while each dance feature 𝑑 𝑡 = [ 𝜏 ; 𝜃 ] consists of SMPL root translation 𝜏 and 6D joint rotation [20, 46]. We synchronize 𝑀 and 𝐷 at 30 FPS to ensure precise temporal alignment.

## 3.2 High-Fidelity Dance Tokenization

Mainstream methods[26-28] typically adopt VQ-VAE for motion tokenization. However, VQ-VAE often suffers from codebook collapse, where only a small subset of codes are utilized, leading to limited representational diversity and degrading physical plausibility of reconstructed dance [21]. To address this, we adopt Finite Scalar Quantization (FSQ), which replaces vector-wise code selection with differentiable scalar rounding. Unlike VQ-based methods that rely on discrete vector assignments, FSQ quantizes each feature dimension independently. This uniform tokenization scheme ensures not only balanced usage of code space, but also stable gradient flow during training.

3.2.1 Uniform Dance Tokenization. Choreographic units form the basic elements of dance structure, exhibiting commonality across styles and tempos. We aim to learn a reusable codebook that unsupervisedly encodes any dance sequence into composable and interchangeable dance tokens, enabling the synthesis of novel high-fidelity motions through the recombination of existing tokens. Due to the relative independence between upper-body and lowerbody dance movements, the upper body primarily conveys emotion and stylistic details, while the lower body focuses on rhythm execution and spatial transitions. We create separate codebooks Z = {Z 𝑢 , Z 𝑙 } for the upper and lower body. This decomposition also allows the combination of different code pairs to cover a wider array of choreographic units.

The architecture of High-Fidelity Dance Tokenization (HFDT) stage is illustrated in Fig. 2. HFDT initiates with a Dance Encoder E (a three-layer 1D-CNN for information aggregation and a two-layer MLP for dimension adjustment) encoding the dance sequence 𝐷 = { 𝐷 𝑢 , 𝐷 𝑙 } into context-aware features z = { z u , z l } . These features are quantized using Finite Scalar Quantization (FSQ) to obtain ˆ z = { ˆ z u , ˆ z l } , which are then decoded by Dance Decoder D (a two-layer MLP for dimension adjustment and a three-layer 1D TransConv for information restoration) to reconstruct the dance movement ˆ 𝐷 = { ˆ 𝐷 𝑢 , ˆ 𝐷 𝑙 } . FSQ enables balanced utilization and stable gradient propagation via differentiable bounded rounding:

<!-- formula-not-decoded -->

where 𝑓 (·) is the bounding function, set as the sigmoid (·) function in practice. Each channel in ˆ z will be quantized into one of the unique 𝐿 integers, therefore we have ˆ z ∈ { 1 , . . . , 𝐿 } 𝑑 . The codebook size 𝑘 is calculated as 𝑘 = ˛ 𝑑 𝑖 = 1 𝐿 𝑖 , and 𝐿, 𝑑 are hyperparameter. sg refers to Stop-Gradient.

3.2.2 High-Fidelity Motion Reconstruction. Unlike VQ-VAE requiring additional loss to update any extra lookup codebook, FSQ directly integrates numerical approximations "Round" within its workflow. The Dance encoder E and decoder D are trained jointly via the motion reconstruction loss L 𝑟𝑒𝑐 :

<!-- formula-not-decoded -->

Simple reconstruction on SMPL parameters treats all joints equally, neglecting the complex hierarchical tree structure of human body joints, different joints vary in their tolerance to errors. For instance, errors at the root node propagate throughout all nodes, whereas errors at the hand node primarily affect only itself. Thus, we execute Forward Kinematic (FK)[20] techniques to derive 3D joints and apply spatial reconstruction loss L kin :

<!-- formula-not-decoded -->

Moreover, we introduce the temporal loss L dyn to better model the temporal dynamics of human motion:

<!-- formula-not-decoded -->

where 𝐷 ′ and 𝐷 ′′ represent the ground-truth velocity and acceleration, ˆ 𝐷 ′ and ˆ 𝐷 ′′ are the corresponding predictions, and 𝛼 1, 𝛼 2 are weighting factors.

Figure 2: Overview of MATHDance. Stage 1 (HFDT) quantizes dance into upper/lower-body tokens by FSQ with spatial and temporal reconstruction constraints. Stage 2 (HMDG) autoregressively generates dance tokens from MuQ-derived music features. HMDG utilizes a Mamba-Transformer hybrid architecture, where Transformer is equipped with Sliding Window Attention (SWA). Residual connections are omitted for clarity.

![Figure](images/figure_0068.png)

**[Image: figure_0068.png (2005x1172, 820.5KB)]**

## 3.3 Hybrid Music-to-Dance Generation

Mainstream Transformer-only generation architectures [15, 27] rely heavily on positional encodings to model temporal structure, which provides weak inductive biases for the inherent continuity in music and dance. To address this, we propose the Mamba-Transformer hybrid architecture that combines Mamba's efficient modeling of local dependencies with the Transformer's capacity for global context, thereby enhancing the aesthetic quality of generated dance.

3.3.1 Model Architecture. The Hybrid Music-to-Dance Generation (HMDG) stage adopts a Mamba-Transformer hybrid architecture to generate the appropriate probability distribution over dance tokens 𝑎 0: 𝑇 ′ -1 = { 𝑎 𝑙 0: 𝑇 ′ -1 , 𝑎 𝑢 0: 𝑇 ′ -1 } given the input music 𝑚 1: 𝑇 ′ and genre label 𝑔 . As illustrated in Fig. 2, HMDG consists of three major components: the Music Encoder, Genre Encoder, and Dance Decoder. (1) Genre Encoder. The one-hot genre label 𝑔 is embedded into a learnable feature, refined via a feed-forward network, and passed through Genre Dropout to support both genre-conditioned and genre-agnostic generation. The resulting feature is then passed to the Dance Decoder via Cross-Genre SWA. (2) Music Encoder. Music features 𝑚 1: 𝑇 ′ are first extracted by the Music Foundation Model MuQ, followed by an MLP for dimension adjustment. These features are then processed by an 𝑁 𝑒 -layer temporal module comprising Mamba, Self-Music SWA, and Feed-Forward submodules. The resulting feature is also then passed to the Dance Decoder via Cross-Music SWA. (3) Dance Decoder. Previously generated dance tokens 𝑝 0: 𝑇 ′ -1 = { 𝑝 𝑙 0: 𝑇 ′ -1 , 𝑝 𝑢 0: 𝑇 ′ -1 } are embedded, and their lower and upper features are fused by element-wise addition. The fused representation is then fed into an 𝑁 𝑑 -layer hierarchical multimodal processing module that integrates Mamba, Self-Dance SWA, CrossMusic SWA, Cross-Genre SWA, and Feed-Forward submodules. Finally, a linear projection is applied to predict the probability distribution over dance tokens 𝑎 0: 𝑇 ′ -1 = { 𝑎 𝑙 0: 𝑇 ′ -1 , 𝑎 𝑢 0: 𝑇 ′ -1 } at the next timestep. During training, we apply a supervised cross-entropy loss [27] to align the predicted action 𝑎 𝑡 with the next-step target token 𝑝 𝑡 + 1. During inference, HMDG supports: (1) autoregressive generation for short sequences ( ≤ 12s); and (2) sliding-window prediction with 12s overlap for long sequences.

3.3.2 MuQ-based Music Representation. A powerful music representation forms the foundation of effective dance generation. Directly using raw audio is impractical due to its high temporal resolution (e.g., 16kHz), which introduces redundancy and hinders alignment with motion sequences. Existing methods either overemphasize low-level acoustic features -e.g., Librosa-based descriptors [15]-or rely solely on high-level semantic embeddings, e.g., MERT [33] and Jukebox [30], thus failing to capture a comprehensive understanding of music relevant to dance generation. In contrast, MuQ [47] is a Music Foundation Model pretrained via self-supervised learning across hierarchical music informatics tasks, including beat detection, instrument classification, music tagging, and etc. MuQ not only achieves state-of-the-art performance on various benchmarks, but also adopts a lightweight architecture that supports real-time feature extraction. In practice, the extracted MuQ features are further downsampled to match the frame rate of the dance token sequence.

3.3.3 Global-Context Modeling. In Transformers [31], the attention layer defines computational dependencies among sequential elements and is implemented as:

<!-- formula-not-decoded -->

where 𝑄 , 𝐾 , and 𝑉 denote the query, key, and value matrices, and 𝑀 is the attention mask. Although music-to-dance generation is typically applied to long sequences, training is commonly conducted on short clips due to limited computational resources. During inference, the sequence is first autoregressively extended up to the training length (step 1), and then completed using a sliding window approach for the remaining portion (step 2). However, standard causal attention [25] aligns only with step 1 and fails to model the dominant second phase during inference, resulting in a mismatch between training and inference. To mitigate this misalignment, we introduce the Sliding Window Attention (SWA) mechanism by equipping 𝑀 with a windowed mask that reflects the actual inference procedure.

3.3.4 Local-Denpendency Modeling. While the Transformer excels at long-range modeling, its position-invariant design and reliance on positional encoding [31] limit its ability to capture local temporal dependencies-crucial in music and dance due to their strong local continuity. In contrast, Mamba [4] exhibits strong performance in fine-grained local modeling, benefiting from its sequential inductive bias [32]. We adopt Mamba to capture local dependencies via its selective state-space mechanism. Specifically, Mamba adaptively learns transition parameters through fully-connected layers and employs structured matrices to improve efficiency. At each time step 𝑡 , the hidden state ℎ 𝑡 is updated as:

<!-- formula-not-decoded -->

where ¯ 𝐴 𝑡 , ¯ 𝐵 𝑡 , 𝐶 𝑡 are dynamically updated parameters. Through discretization with sampling interval Δ , the state transitions become:

<!-- formula-not-decoded -->

where ( Δ 𝐴 ) -1 is the inverse of Δ 𝐴 , and 𝐼 denotes the identity matrix. The scan module captures temporal dependencies by applying trainable parameters across input segments.

Figure 3: Architecture of the retrieval model.

![Figure](images/figure_0082.png)

**[Image: figure_0082.png (940x306, 146.3KB)]**

## 4 Experiment

## 4.1 Dataset

Weevaluate our method on two benchmark datasets: (1) FineDance. FineDance [18] is the largest public dataset for 3D music-to-dance generation, featuring professionally performed dances captured via optical motion capture. It provides 7.7 hours of motion data at 30 fps across 16 distinct dance genres. Following [17], we evaluate on 20 test-set music clips, generating 1024-frame (34.13s) dance sequences. (2) AIST++. AIST++ [15] is a widely used benchmark comprising 5.2 hours of 60 fps street dance motion, covering 10 dance genres. Following [15], we use 40 test-set music clips to generate 1200-frame (20.00s) sequences.

## 4.2 Evaluation

Due to the subjectivity and abstraction of dance, evaluating musicto-dance generation remains a fundamental challenge. Mainstream methods lack a deep understanding of dance semantics, typically compute Stylistic, Kinetic and Geometric Features (S&amp;K&amp;G) [15]based FID for assessing dance quality and use BAS[15] for synchronization. However, S&amp;K&amp;G captures only low-level motion cues, ignoring higher-level semantics. BAS, in particular, focuses narrowly on beat alignment, overlooking the multifaceted interplay between music and dance, including rhythm and expressive semantics. Given the strong coupling between movement and music, training encoders on real music-dance pairs via contrastive learning enables modeling of their shared artistic characteristics. Thus, retrieval-based protocol offer a promising alternative for evaluating generative models, and have shown strong effectiveness across domains such as text-to-motion[5], video-to-music[49], and text-to-video generation[19].

4.2.1 Model Architecture. Inspired by [37], we design a musicdance retrieval model composed of three main components: a Music Encoder, a Dance Encoder, and a Contrastive Learning module, as illustrated in Fig. 3. Our retrieval model also incorporates MuQ [47] and SMPL [20] to represent music and dance, respectively. The extracted features are subsequently fed into a multi-layer block, which consists of a Transformer-based Temporal Processing module and an Average-Pooling-based Temporal Downsampling module. Finally, the music features 𝑓 𝑚 and dance features 𝑓 𝑑 utilize CLIP [24] loss for contrastive learning.

4.2.2 Metric Construction. Upon completion of model training, we extract music features 𝑓 𝑚 and dance features 𝑓 𝑑 using our retrieval model, both represented in a unified embedding space optimized via contrastive learning. To comprehensively assess the quality and alignment of music-to-dance generation, we introduce a suite of standardized metrics derived from these features, inspired by established practices in text-to-motion generation [5, 6]. (1) Recall at 5 (R@5): assesses macro-level semantic alignment between music and dance sequence. For evaluating music-dance alignment, we measure Recall at 5 (R@5), defined as the proportion of cases where the ground-truth music 𝑓 𝑚 is successfully retrieved within the top-5 ranks when using generated dance features 𝑓 𝑑 as queries. (2) Multi-Modality Distance (MM-Dist): evaluates micro-level feature distances between music and dances. To complement this with a fine-grained measure, we introduce Multi-Modality Distance (MM-Dist), which computes the average Euclidean distance between 𝑓 𝑚 and 𝑓 𝑑 across the dataset. (3) Fréchet Inception Distance (FID): quantifies macro-level distributional discrepancies between ground-truth and generated dances. For evaluating the global distributional similarity between generated dances and real dances, we compute the Fr'echet Inception Distance (FID) on 𝑓 𝑑 , reflecting discrepancies in the overall feature space. (4) Modality Distance (M-Dist): measures micro-level feature distances between ground-truth and generated dances. To assess fine-grained motion fidelity, we propose the Modality Distance (M-Dist), calculated as the average Euclidean distance between generated features 𝑓 𝑔𝑒𝑛 𝑑 and their corresponding ground-truth features 𝑓 𝑔𝑡 𝑑 at the clip level. (5) Diversity (DIV): captures the creativity and variability of generated dances. Finally, to quantify the expressive richness and variability of the generated dances, we include Diversity (Div), measured as the average pairwise Euclidean distance among a batch of generated 𝑓 𝑑 features. Together, these metrics provide a rigorous and multi-perspective evaluation framework that captures both the fidelity and semantic coherence of music-to-dance generation.

Table 1: Comparison with SOTAs on the FineDance dataset.

|           | Quality   | Quality   | Synchronization   | Synchronization   | Creativity   | Creativity   | User Study   | User Study   | Complexity   | Complexity   |
|-----------|-----------|-----------|-------------------|-------------------|--------------|--------------|--------------|--------------|--------------|--------------|
|           | R@5 ↑     | MM-Dist ↓ | FID ↓             | M-Dist ↓          | DIV ↑        | DS ↑         | DQ ↑         | DD ↑         | Params ↓     | Latency ↓    |
| GT        | 23.84     | 17.77     | 0.00              | 0.00              | 17.65        | 4.6          | 4.5          | 4.5          | -            | -            |
| Random    | 3.31      | 20.63     | 402.65            | 20.25             | 1.08         | -            | -            | -            | -            | -            |
| FineNet   | 14.49     | 17.97     | 171.39            | 16.30             | 13.43        | 3.9          | 3.8          | 3.1          | 94M          | 3.97s        |
| Bailando  | 13.91     | 18.56     | 142.73            | 16.15             | 15.22        | 3.9          | 3.7          | 3.7          | 152M         | 5.46s        |
| Lodge     | 21.85     | 17.56     | 63.28             | 11.79             | 15.24        | 4.2          | 3.8          | 3.6          | 235M         | 4.57s        |
| MATHDance | 25.83     | 17.32     | 50.81             | 12.89             | 15.47        | 4.3          | 4.2          | 4.0          | 102M         | 2.97s        |

## 4.3 Comparison

4.3.1 Quantitative Analysis. To compare the performance of different methods in terms of quantitative analysis, we evaluate MATHDance against Bailando[27], FineNet[18] and Lodge[17]. To provide proper reference, we also report evaluation metrics on real dance (GT) and randomly Gaussian noise (Random). As shown in Tab. 2 and Tab. 1, MATHDance achieves superior performance across all metrics on various dataset. On FineDance, it improves R@5 by 3.98 and reduces FID by 12.47 compared to Lodge, while maintaining strong MM-Dist and M-Dist scores. On AIST++, it further improves R@5 by 1.08 and reduces FID by 16.99, showing robustness across datasets. DIV remains competitive in both settings. In summary, by addressing the challenge of choreographic consistency, MATHDance achieves significant improvements in both dance quality and dance synchronization.

Table 2: Comparison on the AIST++ dataset.

| Methods   |   R@5 ↑ |   MM-Dist ↓ |   FID ↓ |   M-Dist ↓ |   DIV ↑ |
|-----------|---------|-------------|---------|------------|---------|
| FACT      |   10.63 |       20.78 |  126.44 |      16.81 |   12.37 |
| Bailando  |   16.87 |       18.71 |   92.77 |      14.38 |   14.89 |
| EDGE      |   18.45 |       18.93 |   86.03 |      13.92 |   14.03 |
| Lodge     |   21.66 |       17.24 |   67.81 |      13.62 |   15.21 |
| Ours      |   22.74 |       17.01 |   50.82 |      12.45 |   15.35 |

Table 3: Ablation on Music-to-Dance Generation stage.

| Methods   |   R@5 ↑ |   MM-Dist ↓ |   FID ↓ |   M-Dist ↓ |   DIV ↑ |
|-----------|---------|-------------|---------|------------|---------|
| GT        |   23.84 |       17.77 |    0.00 |       0.00 |   17.65 |
| Random    |    3.31 |       20.63 |  402.65 |      20.25 |    1.08 |
| w/o SWA   |   15.89 |       18.04 |  104.19 |      13.93 |   14.63 |
| w/o MuQ   |   21.86 |       18.66 |   63.27 |      15.34 |   16.40 |
| w/o Mamba |   17.21 |       18.35 |   80.27 |      15.92 |   16.21 |
| Ours      |   25.83 |       17.32 |   50.81 |      12.89 |   15.47 |

4.3.2 User Study. Dance's inherent subjectivity makes user feedback essential for evaluating generated movements[13], particularly in the music-to-dance generation. We select 30 music segments (34.13 seconds each) and generate dance sequence using models mentioned above. These sequences are evaluated through a doubleblind questionnaire, by 30 participants with backgrounds in dance practice. The questionnaires are based on a 5-point scale (Great, Good, Fair, Bad, Terrible) and assess three aspects: Dance Synchronization (DS, alignment with rhythm and style), Dance Quality (DQ, physical plausibility and aesthetics), and Dance Diversity (DD, variety and creativity). As shown in Tab. 1, MATHDance significantly outperforms the other methods across all metrics (DS = 4.3, DQ

Table 4: Ablation on Dance Tokenization stage.

| Model                                     | Joints                             | Joints                      | SMPL                                      | SMPL          | CUR                                     | CUR                              | CUR                             |
|-------------------------------------------|------------------------------------|-----------------------------|-------------------------------------------|---------------|-----------------------------------------|----------------------------------|---------------------------------|
| Model                                     | MSE ↓                              | MAE ↓                       | MSE ↓                                     | MAE ↓         | T@1 ↑                                   | T@5 ↑                            | T@10 ↑                          |
| FSQ (Comp.+Spat.+Temp.) FSQ (Comp.+Spat.) | 0.0076 0.0089                      | 0.0491 0.0507               | 0.0238 0.0240                             | 0.0847 0.0859 | 100% 100%                               | 100% 100%                        | 100% 100%                       |
| FSQ (Comp.) VQ-VAE (Comp.)                | 0.0123 0.0168 0.0159 0.0220 0.0182 | 0.0606 0.0842 0.0757 0.0807 | 0.0253 0.0349 0.0258 0.0308 0.0257 0.0280 | 0.0866 0.0909 | 100% 98.80% 87.35% 71.90% 82.24% 47.29% | 100% 63.05% 79.65% 46.75% 23.44% | 100% 37.60% 73.90% 31.85% 6.67% |
|                                           |                                    | 0.0737                      |                                           | 0.1038        |                                         |                                  |                                 |
| FSQ (Res.)                                |                                    | 0.0714                      |                                           |               |                                         |                                  |                                 |
| VQ-VAE (Res.)                             |                                    |                             |                                           | 0.0984        |                                         |                                  |                                 |
| FSQ (Orig.)                               |                                    |                             |                                           | 0.0912        |                                         |                                  |                                 |
| VQ-VAE (Orig.)                            | 0.0204                             |                             |                                           | 0.0932        |                                         | 14.67%                           | 5.86%                           |

Table 5: Ablation for dance-to-music retrieval.

Figure 4: Comparison with SOTAs on a soft folk music clip.

| Model              |   Recall@5 ↑ |   Recall@10 ↑ |   Median Rank ↓ |   Mean Rank ↓ |
|--------------------|--------------|---------------|-----------------|---------------|
| AVG->CNN           |        15.23 |         31.79 |            19.0 |         25.42 |
| MuQ->Librosa       |        11.26 |         23.84 |            25.0 |         36.26 |
| Transformer->Mamba |         5.96 |         14.57 |            34.0 |         45.22 |
| Transformer+Mamba  |        22.51 |         38.41 |            15.0 |         23.54 |
| Ours               |        23.84 |         38.41 |            14.0 |         18.79 |

![Figure](images/figure_0105.png)

**[Image: figure_0105.png (964x355, 137.8KB)]**

Figure 5: Ablation on Music-to-Dance Generation stage.

![Figure](images/figure_0106.png)

**[Image: figure_0106.png (959x398, 128.9KB)]**

= 4.2, DD = 4.0), demonstrating its superiority in terms of human preferences.

4.3.3 Complexity Analysis. We also include a comparison of generation complexity for producing a 1024-frame (34.13s) dance sequence. All latency evaluations are conducted on an RTX 3090 GPU with an Intel Xeon Gold 5218 CPU. As shown in Tab. 1, MATHDance demonstrates clear superiority in generation efficiency. It has a comparable parameter size to FineNet (102M vs. 94M), and achieves the lowest inference latency of 2.97s, outperforming all baselines by a large margin. This speed enables seamless integration into interactive systems, where rapid feedback is crucial for user engagement in practice.

## 4.4 Ablation Study

4.4.1 Hybrid Music-to-Dance Generation Stage. In this section, we investigate the impacts of Sliding Window Attention (SWA), MuQbased Music Representation (MuQ), and Mamba-enhanced Architecture (Mamba), as detailed in Tab. 3. (1) SWA . The SWA is utilized to adapt to the sliding-window inference phase, which is prevalent in long-sequence generation. Replacing sliding window attention with causal attention results in a notable decrease across all metrics, affirming SWA's critical role. (2) MuQ . To enhance music representation, we integrate MuQ, which, when replaced with Librosa, leads to obvious declines in R@5/MM-Dist/FID/M-Dist, while DIV shows a slight increase, thus validating MuQ's effectiveness in driving dance generation. (3) Mamba . The Mamba is introduced to bolster local dependencies, where significant improvements are noted in R@5/MM-Dist (8.52/1.03) and FID/M-Dist (29.46/3.03), showcasing its capability to enhance music-dance synchronization and dance quality.

4.4.2 High-Fidelity Dance Tokenization Stage. In this section, we investigate the impacts of Quantization Strategy, Spatial Loss, and Temporal Loss, as detailed in Tab. 4. Here we compare the performance of methods: Original Motion FSQ (Orig.), Original Motion VQ-VAE (Orig.), Residual Motion FSQ (Res.), Residual Motion VQVAE (Res.), Compositional Motion FSQ (Comp.), Compositional Motion VQ-VAE (Comp.), Compositional Motion FSQ with Spatial Loss (Comp.+Spat.) and Compositional Motion FSQ with Spatial and Temporal Loss (Comp.+Spat.+Temp.). To ensure fairness, we set the codebook number to 2/2/1 and the codebook size to 1024/1024/65536 for Comp., Res., and Orig., respectively. We evaluate by MSE/MAE on SMPL and 3D Joints, and analyze the Codebook Utilization Rate (CUR), which measures the proportion of codebook entries whose usage exceeds thresholds of 1, 5, and 10 times. Additionally, the codebook was used 192,375 times in total. (1) Quantization Strategy. To mitigate codebook collapse in VQVAE, we adopt FSQ with differentiable scalar quantization replacing discrete argmin selection. Across all codebook structures-Original, Compositional, and Residual-FSQ consistently outperforms VQVAE in SMPL and 3D joint reconstruction. Moreover, the superior result on CUR:T@1/5/10 indicates its efficient codebook utilization. (2) Spatial Loss. To enhance the spatial naturalness, we introduce Spatial Loss by imposing constraints on 3D joints derived from Forward Kinematics (FK), which yields significant improvements in both SMPL and joint reconstruction. (3) Temporal Loss. To improve temporal fidelity, we apply Temporal Loss by constraining joint velocity and acceleration, which also brings notable gains in reconstruction accuracy.

4.4.3 Ablation Study for Retrieval Model. Leveraging the efficiency and lower complexity inherent in retrieval tasks, we employ retrieval models to evaluate generative models. We investigate the effects of Temporal Downsampling, Music Representation, and Temporal Modeling in our proposed dance-music retrieval model, using Recall@5/10 and Median/Mean Rank as evaluation metrics. As shown in Tab. 5, replacing average pooling (AVG) with CNN for temporal downsampling leads to a noticeable drop across all metrics. Replacing MuQ with Librosa results in an even larger performance decline, also demonstrating MuQ's strong capability in music feature representation. Moreover, neither replacing nor adding Mamba layers brings improvement, suggesting that Mamba is more suitable for music-to-dance generation tasks with strong local dependency, rather than retrieval tasks that require global semantic understanding.

Figure 6: Ablation on Dance Tokenization stage.

![Figure](images/figure_0114.png)

**[Image: figure_0114.png (965x713, 256.4KB)]**

## 4.5 Qualitative Analysis

4.5.1 Comparison for Music-to-Dance Generation. To assess the visual quality of the generated dance sequences, we perform a qualitative comparison between MATHDance and several existing baseline models, as depicted in Fig. 4. In terms of expressiveness, MATHDance outperforms the competing methods in several key areas. Specifically, compared to MATHDance, Lodge occasionally generates awkward poses, such as unnatural limb bending; FineNet often exhibits excessive repetition of motion patterns; Bailando lacks expressiveness and visual aesthetics. These findings underscore the superiority of MATHDance in generating dance with physical plausibility and aesthetic quality.

4.5.2 Ablation on Dance Generation stage. In this section, we explore the impact of all components in the Hybrid Music-to-Dance Generation Stage from a qualitative perspective. As shown in Fig. 5, Slide Window Attention (SWA), MuQ-based Music Representation (MuQ), and Mamba-enhanced temporal modeling (Mamba) each demonstrate their crucial role. Specifically, removing Mamba results in simpler, repetitive movements with weak spatial dynamics and rhythm alignment. Removing SWA leads to minimal, low-energy motion with almost no displacement or musical interaction. Removing MuQ causes a mismatch with strong percussion beats, reducing the coherence between music and motion. In conclusion, all components in this stage effectively contribute to the overall performance.

4.5.3 Ablation on Dance Tokenization stage. In this section, we conduct a qualitative analysis to compare our method with Original Motion FSQ (Orig.), Original Motion VQ-VAE (Orig.), Residual Motion FSQ (Res.), Residual Motion VQ-VAE (Res.), Compositional Motion FSQ (Comp.), and Compositional Motion VQ-VAE (Comp.). As shown in Fig. 6, our method achieves the most accurate dance reconstruction results. Under the same codebook structure, FSQbased quantization strategies clearly outperform those based on VQ-VAE. Under identical quantization settings, the compositional structure exhibits superior performance compared to residual and original structures. Furthermore, our method, enhanced by spatialtemporal constraints, surpasses FSQ (Comp.). These results collectively demonstrate the superiority of our approach in the HighFidelity Dance Tokenization stage.

Table 6: Exploring the reliability of evaluation protocol.

| Evaluation   | DS ↑   | DQ ↑   | DD ↑   |
|--------------|--------|--------|--------|
| BAS          | 72%    | -      | -      |
| Kinetic      | -      | 59%    | 55%    |
| Geometric    | -      | 52%    | 58%    |
| Ours         | 84%    | 77%    | 71%    |

## 4.6 Reliability of Evaluation Protocol

To validate metric reliability, we assess metric-user alignment using randomly selected 100 generated dance pairs from User Study in Sec. 4.3.3. We calculate the alignment rate, defined as the proportion of pairs where human preferences and evaluation metrics consistently agree on the preferred or less preferred option. We compare our retrieval model to Kinetic and Geometric baselines by measuring alignment with DQ (via M-Dist/FID) and DD (via DIV), and assess how well DS aligns with R@5/MM-Dist versus BAS. As shown in Tab. 6, our evaluation metrics achieve strong alignment with human judgments across all three aspects: approximately 12% higher than the second-best in DS, 18% higher in DQ, and 13% higher in DD. The excellent result also validates the effectiveness of using retrievalbased evaluation for generative models in the music-to-dance task.

## 4.7 Implementation Details

- 4.7.1 High-Fidelity Dance Tokenization. Data. We train the tokenizer on 12 s SMPL sequences (30 fps), i.e., 𝑇 = 360 and 𝑋, ˆ 𝑋 ∈ R 360 × 147 (6D rotations). Training clips are constructed using a sliding window (window 360, stride 30). Architecture. Upper- and lower-body branches share the same codebook configuration. A 3-layer CNN encoder 𝐸 temporally downsamples the input, and a 3-layer transposed-conv decoder 𝐺 upsamples it, producing latent codes 𝑝 𝑢 , 𝑝 𝑙 ∈ R 45 ( 𝑇 ′ = 45). We use Finite Scalar Quantization with codebook size 1000, 𝐿 = [ 8 , 5 , 5 , 5 ] , and feature dimension 512. Loss &amp; Optimization. We supervise reconstruction with an SMPL-parameter loss L smpl and a joint-position loss L joint, augmented with velocity and acceleration terms weighted by 𝛼 1 = 0 . 5 and 𝛼 2 = 0 . 25. We train for 200 epochs with Adam ( 𝛽 = ( 0 . 5 , 0 . 99 ) ), fixed learning rate, and batch size 32.

4.7.2 Hybrid Music-to-Dance Generation. Data. We train on the quantized latent codes ( 𝑝 𝑢 , 𝑝 𝑙 ) extracted from 12 s sequences, using the same sliding-window augmentation (window 360, stride 30). Model. MATHDance adopts a Mamba-Transformer hybrid with 𝑁 𝑒 = 6 layers in the music encoder and 𝑁 𝑑 = 6 layers in the dance decoder. To handle missing genre labels at test time, we apply genre dropout (0.3) in the genre encoder. The Mamba block uses model dim 512, state size 16, conv kernel 4, and expansion 2. The Transformer block uses hidden size 512, 8 heads, FFN dim 2048, and dropout 0.25. For sliding-window attention, we use an autoregressive step of 30 and window stride 15. All submodules use LayerNorm and residual connections.

Inputs/Outputs. Genre labels (16 FineDance classes) are embedded to 512 dims via nn.Embedding . MuQ music features (1024 dims) are projected to 512 dims using a 2-layer MLP. The decoder predicts a 2000-way softmax distribution: indices 0-999 correspond to the upper-body codebook and 1000-1999 to the lower-body codebook. Optimization. We train for 300 epochs with Adam ( 𝛽 = ( 0 . 9 , 0 . 99 ) ), fixed learning rate, and batch size 128.

4.7.3 Dance-Music Retrieval for Evaluation. Wetrain a dance-music retrieval model on SMPL motion and MuQ music representations. Training pairs are constructed with a sliding window (window 360, stride 180). Both music and dance encoders use 𝐿 = 9 Transformer layers (hidden 512, 8 heads, dropout 0.25), with temporal downsampling implemented by average pooling (rate 2). We optimize with CLIP loss (temperature 4.6052) using Adam (lr 1 × 10 -5 , 𝛽 = ( 0 . 5 , 0 . 999 ) , batch size 32), and apply StepLR (step 5, 𝛾 = 0 . 33).

## 5 Conclusion

In this paper, we propose MATHDance, a two-stage latent-space framework for music-to-dance generation that ensures choreographic consistency through the High-Fidelity Dance Tokenization stage for physical plausibility, and the Hybrid Music-to-Dance Generation stage for aesthetic quality. Additionally, we introduce a retrieval-based evaluation protocol for music-to-dance generation. Experiments on the FineDance and AIST++ datasets demonstrate MATHDance's superiority in various aspects, and the reliability of our evaluation protocol. In future work, we plan to extend MATHDance with motion or text conditioning to enable more interactive and flexible dance generation.

## References

- [1] Kang Chen, Zhipeng Tan, Jin Lei, Song-Hai Zhang, Yuan-Chen Guo, Weidong Zhang, and Shi-Min Hu. 2021. Choreomaster: choreography-oriented musicdriven dance synthesis. ACM Transactions on Graphics (TOG) 40, 4 (2021), 1-13.
- [2] André Correia and Luís A Alexandre. 2024. Music to Dance as Language Translation using Sequence Models. arXiv preprint arXiv:2403.15569 (2024).
- [3] Kehong Gong, Dongze Lian, Heng Chang, Chuan Guo, Zihang Jiang, Xinxin Zuo, Michael Bi Mi, and Xinchao Wang. 2023. Tm2d: Bimodality driven 3d dance generation via music-text integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 9942-9952.
- [4] Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023).
- [5] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. 2022. Generating diverse and natural 3d human motions from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 51525161.
- [6] Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. 2022. Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In European Conference on Computer Vision . Springer, 580-597.
- [7] Ruozi Huang, Huang Hu, Wei Wu, Kei Sawada, Mi Zhang, and Daxin Jiang. 2020. Dance revolution: Long-term dance generation with music via curriculum learning. arXiv preprint arXiv:2006.06119 (2020).
- [8] Yuhang Huang, Junjie Zhang, Shuyan Liu, Qian Bao, Dan Zeng, Zhineng Chen, and Wu Liu. 2022. Genre-conditioned long-term 3d dance generation driven by music. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 4858-4862.
- [9] Nhat Le, Tuong Do, Khoa Do, Hien Nguyen, Erman Tjiputra, Quang D Tran, and Anh Nguyen. 2023. Controllable group choreography using contrastive diffusion. ACM Transactions on Graphics (TOG) 42, 6 (2023), 1-14.
- [10] Nhat Le, Thang Pham, Tuong Do, Erman Tjiputra, Quang D Tran, and Anh Nguyen. 2023. Music-driven group choreography. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 8673-8682.
- [11] Hsin-Ying Lee, Xiaodong Yang, Ming-Yu Liu, Ting-Chun Wang, Yu-Ding Lu, Ming-Hsuan Yang, and Jan Kautz. 2019. Dancing to music. Advances in neural information processing systems 32 (2019).
- [12] Juheon Lee, Seohyun Kim, and Kyogu Lee. 2018. Listen to dance: Music-driven choreography generation using autoregressive encoder-decoder network. arXiv preprint arXiv:1811.00818 (2018).
- [13] Dorothée Legrand and Susanne Ravn. 2009. Perceiving subjectivity in bodily movement: The case of dancers. Phenomenology and the Cognitive Sciences 8 (2009), 389-408.
- [14] Buyu Li, Yongchi Zhao, Shi Zhelun, and Lu Sheng. 2022. Danceformer: Music conditioned 3d dance generation with parametric motion transformer. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 36. 1272-1279.
- [15] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. 2021. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 13401-13412.
- [16] Ronghui Li, Hongwen Zhang, Yachao Zhang, Yuxiang Zhang, Youliang Zhang, Jie Guo, Yan Zhang, Xiu Li, and Yebin Liu. 2024. Lodge++: High-quality and Long Dance Generation with Vivid Choreography Patterns. arXiv preprint arXiv:2410.20389 (2024).
- [17] Ronghui Li, YuXiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li. 2024. Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 1524-1534.
- [18] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. 2023. Finedance: A fine-grained choreography dataset for 3d full body dance generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 10234-10243.
- [19] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. 2024. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 22139-22149.
- [20] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. 2023. SMPL: A skinned multi-person linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2 . 851-866.
- [21] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. 2023. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505 (2023).
- [22] Meinard Müller, Tido Röder, and Michael Clausen. 2005. Efficient content-based retrieval of motion capture data. In ACM SIGGRAPH 2005 Papers . 677-685.
- [23] Kensuke Onuma, Christos Faloutsos, and Jessica K Hodgins. 2008. FMDistance: A Fast and Effective Distance Function for Motion Capture Data. Eurographics (Short Papers) 7 (2008).
- [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning . PMLR, 8748-8763.
- [25] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. (2018).
- [26] Li Siyao, Tianpei Gu, Zhitao Yang, Zhengyu Lin, Ziwei Liu, Henghui Ding, Lei Yang, and Chen Change Loy. 2024. Duolando: Follower gpt with off-policy reinforcement learning for dance accompaniment. arXiv preprint arXiv:2403.18811 (2024).
- [27] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. 2022. Bailando: 3d dance generation by actorcritic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 11050-11059.
- [28] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. 2023. Bailando++: 3d dance gpt with choreographic memory. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).
- [29] Taoran Tang, Jia Jia, and Hanyang Mao. 2018. Dance with melody: An lstmautoencoder approach to music-oriented dance synthesis. In Proceedings of the 26th ACM international conference on Multimedia . 1598-1606.
- [30] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. 2023. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 448-458.
- [31] A Vaswani. 2017. Attention is all you need. Advances in Neural Information Processing Systems (2017).
- [32] Zunnan Xu, Yukang Lin, Haonan Han, Sicheng Yang, Ronghui Li, Yachao Zhang, and Xiu Li. 2024. Mambatalk: Efficient holistic gesture synthesis with selective state space models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems .
- [33] Kaixing Yang, Xulong Tang, Ran Diao, Hongyan Liu, Jun He, and Zhaoxin Fan. 2024. CoDancers: Music-Driven Coherent Group Dance Generation with Choreographic Unit. In Proceedings of the 2024 International Conference on Multimedia Retrieval . 675-683.

- [34] Kaixing Yang, Xulong Tang, Ziqiao Peng, Yuxuan Hu, Jun He, and Hongyan Liu. 2025. Megadance: Mixture-of-experts architecture for genre-aware 3d dance generation. arXiv preprint arXiv:2505.17543 (2025).
- [35] Kaixing Yang, Xulong Tang, Ziqiao Peng, Xiangyue Zhang, Puwei Wang, Jun He, and Hongyan Liu. 2025. FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation. arXiv preprint arXiv:2511.21029 (2025).
- [36] Kaixing Yang, Xulong Tang, Haoyu Wu, Qinliang Xue, Biao Qin, Hongyan Liu, and Zhaoxin Fan. 2024. CoheDancers: Enhancing Interactive Group Dance Generation through Music-Driven Coherence Decomposition. arXiv preprint arXiv:2412.19123 (2024).
- [37] Kaixing Yang, Xukun Zhou, Xulong Tang, Ran Diao, Hongyan Liu, Jun He, and Zhaoxin Fan. 2024. BeatDance: A Beat-Based Model-Agnostic Contrastive Learning Framework for Music-Dance Retrieval. In Proceedings of the 2024 International Conference on Multimedia Retrieval . 11-19.
- [38] Kaixing Yang, Jiashu Zhu, Xulong Tang, Ziqiao Peng, Xiangyue Zhang, Puwei Wang, Jiahong Wu, Xiangxiang Chu, Hongyan Liu, and Jun He. 2025. MACEDance: Motion-Appearance Cascaded Experts for Music-Driven Dance Video Generation. arXiv preprint arXiv:2512.18181 (2025).
- [39] Ziyue Yang, Kaixing Yang, and Xulong Tang. 2026. TokenDance: Token-to-Token Music-to-Dance Generation with Bidirectional Mamba. arXiv:2603.27314 [cs.AI] https://arxiv.org/abs/2603.27314
- [40] Zijie Ye, Haozhe Wu, Jia Jia, Yaohua Bu, Wei Chen, Fanbo Meng, and Yanfeng Wang. 2020. Choreonet: Towards music to dance synthesis with choreographic action unit. In Proceedings of the 28th ACM International Conference on Multimedia . 744-752.
- [41] Xiangyue Zhang, Yifan Jia, Jiaxu Zhang, Yijie Yang, and Zhigang Tu. 2025. Robust 2D skeleton action recognition via decoupling and distilling 3D latent features. IEEE Transactions on Circuits and Systems for Video Technology (2025).
- [42] Xiangyue Zhang, Jianfang Li, Jianqiang Ren, and Jiaxu Zhang. 2026. Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints. In Proceedings of the AAAI Conference on

Artificial Intelligence , Vol. 40. 12834-12842.

- [43] Xiangyue Zhang, Jianfang Li, Jiaxu Zhang, Ziqiang Dang, Jianqiang Ren, Liefeng Bo, and Zhigang Tu. 2025. Semtalk: Holistic co-speech motion generation with frame-level semantic emphasis. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 13761-13771.
- [44] Xiangyue Zhang, Jianfang Li, Jiaxu Zhang, Jianqiang Ren, Liefeng Bo, and Zhigang Tu. 2025. Echomask: Speech-queried attention-based mask modeling for holistic co-speech motion generation. In Proceedings of the 33rd ACM International Conference on Multimedia . 10827-10836.
- [45] Pengfei Zhou, Xiangyue Zhang, Xukun Shen, and Yong Hu. 2026. Not All Frames Are Equal: Complexity-Aware Masked Motion Generation via Motion Spectral Descriptors. arXiv:2603.29655 [cs.CV] https://arxiv.org/abs/2603.29655
- [46] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. 2019. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition . 5745-5753.
- [47] Haina Zhu, Yizhi Zhou, Hangting Chen, Jianwei Yu, Ziyang Ma, Rongzhi Gu, Yi Luo, Wei Tan, and Xie Chen. 2025. Muq: Self-supervised music representation learning with mel residual vector quantization. arXiv preprint arXiv:2501.01108 (2025).
- [48] Haolin Zhuang, Shun Lei, Long Xiao, Weiqin Li, Liyang Chen, Sicheng Yang, Zhiyong Wu, Shiyin Kang, and Helen Meng. 2023. GTN-Bailando: Genre consistent long-term 3d dance generation based on pre-trained genre token network. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE, 1-5.
- [49] Le Zhuo, Zhaokai Wang, Baisen Wang, Yue Liao, Chenxi Bao, Stanley Peng, Songhao Han, Aixi Zhang, Fei Fang, and Si Liu. 2023. Video background music generation: Dataset, method and evaluation. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 15637-15647.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
---

## Extracted Images

| # | File | Dimensions | Size |
|---|------|------------|------|
| 1 | figure_0014.png | 972x215 | 107.5KB |
| 2 | figure_0019.png | 997x221 | 119.2KB |
| 3 | figure_0068.png | 2005x1172 | 820.5KB |
| 4 | figure_0082.png | 940x306 | 146.3KB |
| 5 | figure_0105.png | 964x355 | 137.8KB |
| 6 | figure_0106.png | 959x398 | 128.9KB |
| 7 | figure_0114.png | 965x713 | 256.4KB |
