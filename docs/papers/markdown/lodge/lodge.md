---
source: lodge.pdf
total_pages: 14
extracted_at: 2026-05-11T23:07:01.923407
images_dir: images
---

## Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives Lodge Lodge

Ronghui Li 1 , 2 , YuXiang Zhang 1 ,Yachao Zhang 1 , Hongwen Zhang 4 ,Jie Guo 2 ,Yan Zhang 3 ‡ ,Yebin Liu 1 , Xiu Li 1 † 1 Tsinghua University, 2 Peng Cheng Laboratory 3 Meshcapade, 4 Beijing Normal University

N frames L frames

![Figure](images/figure_0003.png)

**[Image: figure_0003.png (1954x644, 353.3KB)]**

frames

Global Diffusion Figure 1. Lodge can parallelly generate extremely long dance. The sections highlighted in green represent the characteristic dance primitives. These are expressive 8-frame movements that not only support parallel generation but also contains choreographic patterns. They guide the diffusion network to generate long, expressive dances in parallel while adhering to choreographic rules.

## Abstract

Librosa

TE TE Music feature Genre Embedding G LD Concat Concat Concat Concat Split g g 𝑚𝑙 1 𝑚𝑙 2 𝑚𝑙 3 𝑚𝑙 4 g g g 𝑚𝑔 CA We propose Lodge, a network capable of generating extremely long dance sequences conditioned on given music. We design Lodge as a two-stage coarse to fine diffusion architecture, and propose the characteristic dance primitives that possess significant expressiveness as intermediate representations between two diffusion models. The first stage is global diffusion, which focuses on comprehending the coarse-level music-dance correlation and production characteristic dance primitives. In contrast, the second-stage is the local diffusion, which parallelly generates detailed motion sequences under the guidance of the dance primitives and choreographic rules. In addition, we propose a Foot Refine Block to optimize the contact between the feet and the ground, enhancing the physical realism of the motion. Our approach can parallelly generate dance sequences of extremely long length, striking a balance between global Global music feature g 𝑚𝑙 𝑖 choreographic patterns and local motion quality and expressiveness. Extensive experiments validate the efficacy of our method. Code, models, and demonstrative video results are available at: https://li-ronghui.github.io/lodge CA

† corresponding author

‡ This work was done while YZ was at ETH Z¨ urich.

CA

## 1. Introduction

Concatenate

feature

Keymotion

Hard/Soft Guidance LD LD LD Keymotion for soft guidance for hard guidance 𝑑ℎ 𝑑𝑠 Given a piece of long-term music, we aim at generating high-fidelity and diverse 3D dance motions in an automatic and efficient manner. An effective solution is desired not only in many applications e.g. movie and game production, but also of high potential to inspire dance designers with novel movements, and improve their productivity.

Choreography Augment Generated dance 𝑑 CA With rapid advances in generative AI in recent years, existing methods [19, 23, 33, 40, 41] demonstrated the ability to generate dance for seconds. However, dance in real applications often lasts for minutes. Dance performances and social dance usually last 3 to 5 minutes. Dance theater can last for more than 15 minutes or even an hour. Therefore, the extremely long dance generation is becoming increas-

CA

ingly important as the demand for engaging dance content continues to grow.

However, generating long dance sequences poses a notable challenge due to the substantial computational resources needed for training. Therefore, many methods are based on autoregressive models [14, 22, 40], and continuously generate dance movements based on a relatively small sliding window. This autoregressive nature accumulates the model prediction errors as time progresses, and prevents the model from learning global choreographic patterns. As a result, motion freezing often occurs after several seconds [54]. There are also some methods [40, 58, 62] maintain a latent space to represent motion, and combine a autoregressive based sequence model to learn music-dance paired relationship. However, the compressed latent space with limited representational capacity also makes these methods prone to overfitting, resulting in poor generalization and diversity. Recently, EDGE [47] proposed a diffusion-based dance generation model. During the denoising process, EDGE parallelly generate multiple dance segments with overlap while maintaining consistency between these overlapping parts using diffusion inpainting [26], and finally splices these segments into a long dance by linear interpolation. However, their dances lack an overall choreographic structure and shows incoherence at the splices frames.

In summary, these existing methods regarding dance generation solely as a sequence-to-sequence problem. They struggle to enhance the dance quality of fine-grained local details while neglect the coarse-level global choreography patterns between music and dance. Referring to [1, 4, 5, 44], dance is normally choreographed in a coarse-to-fine manner. Provided the entire music, dance designers first analyze the music attributes such as rhythm, genre and emotional tone, and create 'dance phrases', i.e. some shortterm expressive movements, which possess powerful expressiveness and richer semantic information. During this stage, dance designers can concentrate on design characteristic dance phrases, such as 'inversions' and 'moonwalks'. Arrange these characteristic dance phrases follow the structured information of the music, the overall dance structure is laid down. Subsequently, the entire dance is created by connecting dance phrases with transition movements.

Following the above insights, we think that the 'dance phrases' contains abundant distinctive movements and can convey global choreographic patterns. Therefore, similar to dance phrases, we propose characteristic dance primitives suitable for network learning. These dance primitives are expressive 8-frame key motions with high kinematic energy, with the following main advantages: (1) They are sparse, which reduce the computational demand. (2) They have rich semantically information, and can transfer choreographic patterns. (3) They possess expressive motion characteristics, which can guide motion diffusion model to generate more dynamic movements and avoiding monotony.

Next, we design a coarse-to-fine dance generation framework with two motion diffusion models and employ the characteristic dance primitives as their intermediate representation. The first stage is coarse-grained global diffusion, which takes as input long music and produces characteristic dance primitives. According to the fundamental choreographic rules, details in Sec. 3, these dance primitives are further augmented to align with the beats and structural information of the music. Subsequently, we employ parallel local diffusion to independently generate short dance segments. Based on some auto-selected dance primitives, we utilize diffusion guidance to strictly constrain consistency between the beginnings and ends of these segments. Therefore, these dance segments can be concatenated into a continuous long dance. Simultaneously, under the guidance of the other dance primitives, the quality, expressiveness, and diversity of each dance segment are enhanced.

In addition, to improve the motion realism and eliminate foot-skating artifacts, we introduce a foot refine block inspired by [59]. We find it is difficult to simply use footrelated losses [56] to optimize the SMPL [25] format motion rotation data, especially in complex dance movements. This is because the optimization objective exists in the linear joint position space while the SMPL format rotation data is mainly in nonlinear rotation space, and there is a domain gap hindering loss convergence. Therefore, we compute foot contact information and utilize the foot refine block to generate modification values addressing foot skating.

In summary, our main contributions are as follows:

- We introduce a coarse-to-fine diffusion framework that can produce long dances in a parallel manner. Our method is capable of learning the overall choreographic patterns while ensuring the quality of local movements.
- We propose the characteristic dance primitives that possess significant expressiveness as intermediate representations between two diffusion models.
- We propose a foot refine block and employ a foot-ground contact loss to eliminate artifacts such as skating, floating, and ground inter-penetration.

## 2. Related Works

## 2.1. Human Motion Synthesis

Human motion generation is an important task in the fields of computer vision and computer graphics. Researchers make significant contributions in this direction. For instance, MDM [46] successfully applies diffusion to the Text2Motion task, yielding high-quality motion results; GestureDiffuCLIP Ao et al. [2] achieves coordinated motion generation with speech and integrates style control through text and video guidance; SAGA[50] and Grasping[17] focuses on natural grasping motion genera- tion; [16, 57, 60] can produce human motions that interact with 3D scenes while avoiding collisions. CALM [45] and ASE [35] introduce reinforcement learning and physical simulation environments to enhance the physical realism of generated movements. Despite substantial progress in aspects like motion quality, diversity, controllability, interactivity, and physical realism, etc, generating dance motions remains a challenging problem due to the inherent complexity and long-duration nature of dance movements.

## 2.2. Music Driven Dance Generation

Numerous studies aim to generate high-quality dance that synchronizes with the input music. These approaches encompass various categories, including motion-graph methods [6], sequence model based methods [19, 22, 40], VQVAE based methods [40, 62], GAN-base methods [19], and diffusion based methods [23, 47].

The traditional motion-graph based methods [3, 30, 33] address this task as a similarity-based retrieval problem, which limits the diversity and creativity of generations. In recent years, deep learning models have gained significant prominence, yielding aesthetically appealing outcomes. In sequence-based methods, LSTM [13] and Transformer [49] networks are commonly employed. These networks typically take as input music and the preceding dance sequence, predicting the subsequent dance in an autoregressive manner. Li et al. propose FACT[22], which inputs music and seed motions into a Transformer network, generating new dance frame by frame in an autoregressive manner, but challenges such as error accumulation and motion freezing [64] phenomena persist. Based on VQ-VAE, Bailando incorporates a reinforcement learning-based action evaluator to optimize rhythm, while TM2D encodes the text-paired motion and music-paired dance into a shared codebook to achieve semantically controllable dance generation. The advantages of VQ-VAE lie in its ability to maintain a pre-trained codebook, ensuring the motion quality of decoded dance sequences. But the codebook also limits dance diversity and hinders the network's generalization. The Generative Adversarial Network (GAN) consists of a generator and a discriminator, engaged in adversarial training to produce realistic data. MNET [19] proposes a transformer-based dance generator and a multi-genre dance discriminator network to generate realistic dance clips and achieve genre control. However, these GAN-based methods suffer from mode collapse and training instability.

In recent years, with the rapid development of neural networks [7, 24, 38, 53, 58], Diffusion-based methods make significant strides in tasks such as image, video, and motion generation [8, 9, 12, 27-29, 43, 52]. FineDance[23] and EDGE[47] introduce Diffusion to generate diverse and high-quality dance clips of seconds, but they only focus on local motion quality of detailed dance clips and cannot quickly generate long-term dance movements that conform to the overall choreography rules.

## 3. Method

## 3.1. Preliminaries

Music and Dance Representation. Given a music clip, we follow [22] and employ Librosa [31] to extract the music 2D feature map m ∈ R L × 35 , in which L is the frame number and 35 is the music feature channels with 1-dim envelope, 20-dim MFCC, 12-dim chroma, 1-dim one-hot peaks, and 1-dim one-hot beats. In addition, we follow EDGE [47] and represent dance as d ∈ R L × 139 . This motion representation obeys the SMPL[25] format (without fingers) and consists of the following components: (1) 4-dim foot-ground contact binary label, corresponding to left toe, left heel, right toe, right heel, where 1 means contact with ground and 0 means no contact; (2) 3-dim root translation; (3) 132-dim rotation information in 6-dim rotation repersentation [61], the first 6-dim is global rotation and the remaining 126 dimensions correspond to the relative rotations of 21 sub-joints propagated along the kinematic chain.

The Diffusion Model. We follow DDPM [11] and EDGE [47] to build our dance generation model. The diffusion model consists of two main processes: a diffusion process and a denoising process. The diffusion process perturbs the ground truth dance data d 0 into d t over t steps, we follow [11] to simplify this multi-step diffusion process into one step, which can be formulated as:

<!-- formula-not-decoded -->

where ¯ α t is within the range of (0 , 1) and follows a monotonically decreasing schedule. ¯ α t converges to 0 as t goes to infinity, making d t converging to a sample from the standard normal distribution. The denoising process employs a Transformer base-network f θ to gradually recover the motion, generating ˆ d 0 conditioned on given music m . Instead of predicting the noise [55], we directly predict the ˆ d 0 like [47]. Therefore, the training process can be formulated as:

<!-- formula-not-decoded -->

Choreography Rules. Based on suggestions from professional choreographers and existing literature[1, 4, 5, 44], we want to generate long-duration dances that obeyed these three basic choreographic rules: (1) The overall genre of the music and the dance should be consistent, conveying similar moods and tones. (2) The beat of the music and the dance should be the same as far as possible. (3) The arrangement of dance should align with the structure of the accompanying music. For instance, identical meters in a musical phrase often correspond to symmetrical movements.

![Figure](images/figure_0044.png)

**[Image: figure_0044.png (1976x1001, 359.0KB)]**

N frames

Figure 2. An overview of our framework. 'TE' is Transformer Encoder, 'G' is the genre of dance, 'LD' is the Local Diffusion Model.

## 3.2. Two-stage Dance Generation

Concat g Concat Genre Embedding Dense FiLM MLP Dense FiLM Cross-Attention Dense FiLM Cond t 𝑑𝑙 𝑇1 𝑓𝑜𝑜𝑡𝑐 Given a extremely long music feature m ∈ R L × 35 , L = kN , we first split m into segments of length N without overlaps, i.e. { m i g ∈ R N × 35 } k i =1 . Our goal is to learn a neural network Loge, d i g = Lodge ( m i g ) , d g ∈ R N × 139 , d = concatenate ( ⌈ m i g ⌉ , dim = 0) , which means Lodge can parallelly generate extremely long dance sequences d ∈ R kN × 139 with a single inference.

𝑚𝑙 Diffusion TimeSteps T MLP Transformer Encoder Music Feature Self-Attention Dense FiLM MLP Dense FiLM Cross-Attention Dense FiLM Self-Attention 𝑑𝑙 𝑇 T=T-1 Denoising Method Overview. In order to simultaneously consider both the global choreographic rules and the local dance details, we design a coarse to fine diffusion network with two stages as shown in Figure 2. The first stage is the global diffusion, which uses the global music feature m g to learn the choreography patterns and produce characteristic dance primitives. The dance primitives are expressive key motions m k ∈ R 8 × 139 with a higher motion kinematic energy, where 8 is the frame number. Then, we perform choreographic augment operations on these dance primitives by the following three steps: (1) We categorize them into hard-cue key motions d h that support parallel generation and soft-cue key motions d s that enhance the dance performance. (2) Based on the second choreographic rule, we mirror these soft-cue key motions. (3) Based on the third choreographic rule, we align soft-cue key motions to the timing of the musical beats.

The second stage is the Local Diffusion (LD), which focuses on the quality of short-duration n frames dance generation, corresponding to several seconds. We further

Forward Kinematics 𝑓𝑜𝑜𝑡𝑣 𝑓𝑜𝑜𝑡𝑝 Transformer Encoder Cond Concat Cond Foot Refine Block Cond Global Average Pooling Jazz Urban Korean Sequence Transformer Block Linear Real/Fake? Real/Fake? Real/Fake? Multi Genre Discriminator 𝑑𝑙 𝑇1 𝑑 𝑙 = 𝑑𝑙 0 split each m g into { m j l ∈ R n × 35 } ⌈ N/n ⌉ j =1 . As shown in Figure 2, we use the characteristic dance primitives as an intermediate-level representation of our two-stage diffusion network. During the inference process, we replace the movements at the beginning and end of d t , as well as those at the timing of musical beats, with these dance primitives. This way, we transfer globally learned choreographic patterns and expressive dance primitives obtained by global diffusion to local diffusion in a diffusion guidance manner. Specifically, the hard-cue key motion uses the diffusion inpainting technique [26, 47] to control the start and end movements of the local diffusion. Meanwhile, during the diffusion denoising process, soft-cue key motions only serve as guidance in the initial 1000 × s steps, where 1000 is the diffusion denoising steps. By adjusting the hyperparameter 's', we can control the extent to which local diffusion is influenced by these soft-cue key motions. Notably, thanks to the hard cue motions, we can parallelly generate dance sequences d much longer than N with a single inference.

## 3.3. Global Diffusion

Previous works overlook the global dependencies between music and dance, focusing only on the music-dance relationship within a small window. To address this issue, we only task global diffusion to generate sparse dance primitives. Subsequently, multiple local diffusions work in parallel to generate complete long dances.

Given a global music features m g extracted by the Librosa[31]. We feed m g into a Transformer downsample network, which comprises a Linear layer and a Transformer Encoder Layer. Next, the compressed global music feature is sent to global diffusion. We adopte the EDGE framework as the foundation for global diffusion, making a single modification by adjusting the training objective to output sparse dance primitives. These primitives are key motions with only 8 frames, categorized as d h and d s .

Figure 3. Training process of Local Diffusion.

![Figure](images/figure_0055.png)

**[Image: figure_0055.png (1981x733, 325.3KB)]**

There are key motions and transition motions in dance, where key motions are those with velocity curves near local minima, displaying greater expressiveness and richer semantic information, while transition motions are relatively monotonic. To ensure that global diffusion concentrates solely on generating expressive key motions. We separated the expressive key movements and monotonous transitional movements in the dataset and trained global diffusion with only the expressive key motions. Since the global diffusion learns key motions on a global scale, it already implicitly captures some choreographic patterns. To further enhance the overall dance coherence, we do choreography augment operation on d s , guiding the local diffusion to produce dance that more closely adheres to choreography rules.

## 3.4. Local Diffusion

Training Process. Thanks to our coarse to fine diffusion architecture, the local diffusion only needs to train the network on n frames, which greatly accelerates the training speed and allows local diffusion to focus on the details of the dance movements for a few seconds. The training process of local diffusion can be seen in Figure 3. We follow EDGE to build the Sequence Transformer Block, which consists of self-attention layer[49], cross-attention layer[39], multi-layer perception layer and the feature-wise linear modulation (FilM)[36].

In addition to the reconstruction loss, we introduce several other losses to enhance training stability and physical realism like previous works [46, 47]. We compute the positional coordinates d ( i ) joint of the human body joints using forward kinematics, and then get the joint velocity d ( i ) j-vel and joint acceleration d ( i ) j-acc . We then add the following loss functions: joint position Eq. (4), velocity Eq. (5), and acceleration Eq. (6):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To optimize the contact between feet and the ground, we follow LEMO[56, 59] in decoupling the horizontal and vertical velocities of the feet, and optimizing the horizontal velocity f hv and downward vertical velocity f dv to 0 when the feet contact with the ground.

<!-- formula-not-decoded -->

where ˆ b is the predicted foot contact label. Our overall training object is the weighted sum of the losses:

<!-- formula-not-decoded -->

where λ genre is formulated as Eq. (9).

Foot Refine Block. The motion is expressed in the SMPL format, facilitating driven various human models and rendering. However, representing motion in the SMPL format involves a sequence of relative rotations and motion tree propagation. Small rotations near the root nodes, such as the legs and knees, result in significant rotations at the feet. Especially in dance movements, which involve a variety of foot actions, these challenges make it difficult for us to straightforwardly resolve foot skating issues by simply using foot-related loss functions. We argue the main issue lies in the domain gap between the optimization objective and the data representation. The contact status between the feet and the ground is measured in a linear space based on joint positions, while the motion in the SMPL format exists in a nonlinear rotation space. To tackle this, we introduce the Foot Refine Block inspired by [59]. This module first computes the positions of foot keypoints foot p through forward kinematics, as well as foot velocity foot v . Then we calculate the foot-ground contact score foot c follow [59]. Building upon this, the Cross Attention mechanism is employed to further optimize foot movements.

Multi Genre Discriminator. Local diffusion can produce high-quality, diverse dance segments. As shown in the Figure 3, to ensure consistency with the overall musical style, we also concatenate the genre embedding g with the music features, resulting in m g l as the condition for local diffusion. We then use a multi-genre discriminator (MGD) to control the dance genre following MNET[19]. The training process of MGD can be formulated as:

<!-- formula-not-decoded -->

Parallel Inference. Given the input m j l , g and corresponding d h , d s , the local diffusion outputs d j l . By concatenating { m j l } N/n j =1 along the time dimension, we obtain d g . For simplicity in description, we omit ' j ' in subsequent writing. To achieve parallel generation of long dance sequences, we divide d h into the first four frames and the last four frames. The first four frames serve as the tail four frames of the previous d l , and the last four frames of d s serve as the leading four frames for the next d l . This approach requires the local diffusion to generate the intervening dance motions coherently. However, directly using Diffusion inpainting techniques to control the first and last frames of each segment results in incoherent motions. To address this issue, we use a joint acceleration loss L j-acc and incorporate a fine-tuning stage. In this stage, we mixture d t of Local Diffusion and the ground truth d l 0 by d l ′ t [: 4] = d l 0 [: 4] , d l ′ t [ -4 :] = d l 0 [ -4 :] , d l ′ t [4 : -4] = d l t [4 : -4] . The L recon loss in the fine-tuning stage is formulated as:

<!-- formula-not-decoded -->

## 4. Experiment

## 4.1. Experimental Setup

Datasets. We validate our method using the public musicdance paired dataset FineDance[23] and AIST++[22]. FineDance employs professional dancers to perform the dances and capture the data with an optical motion capture system. The currently available dance data of FineDance contains 7.7 hours, totaling 831,600 frames, with a frame rate of 30 fps, and includes 16 different dance genres. The average dance length of FineDance is 152.3 seconds compared to 13.3 seconds for the AIST++ dataset, so we use the FineDance dataset to train and test the long-term dance generation algorithm. We test the 20 pieces of music in the test set of the FineDance dataset and generate dance sequences with a length of 1024 frames (34.13 seconds).

AIST++ is also a widely used dance dataset, containing 5.2 hours of dance data, with a frame rate of 60 fps, and includes 10 dance genres.

Implementation details. In the experiments on the FineDance dataset, the global music feature length N is 1024, corresponding to 34.13 seconds; the local music feature length n is 256, corresponding to 8.53 seconds. The global diffusion output 13 characteristic dance primitives, where 5 are d h and 8 are d s . After the choreography augments operation, d s is mirrored to produce 16 instances, and it is aligned with the music's beat. The optimizer of global diffusion and local diffusion are Adan[51], we use the Exponential Moving Average(EMA) [20] strategy to make the loss convergence process more stable. The learning rate is 1 e -4 . In the inference phase, we have two diffusion sampling strategies DDPM [11] and DDIM [42] that can be used to generate dance. On the AIST++ dataset, we downsampled the dance to 30 fps for training. Then we generated dances with 30 fps. Finally, we interpolated the output dances to 60 fps and followed the experimental setup of Bailando [40] for testing. The music-dance data from AIST++ has been segmented into numerous short clips. Therefore, we change the global music feature length N to be 256 and the global music feature length n to be 128.

## 4.2. Comparisons on the FineDance dataset

As shown in Table 1, we compare our method with the advanced existing works. FACT [22] and MNET [19] are auto-gressive dance generation methods. Bailando [40] is an outstanding music-driven dance generation algorithm. It employs VQ-VAE to transform dance movements into tokens. Subsequently, a GPT model forecasts this token sequence, which is then decoded to render the final dance. To the best of our knowledge, EDGE [47] is a diffusionbased dance generation algorithm, achieving the strongest qualitative performance in short-duration dance generation. During the diffusion denoising process, they assign the lat- ter half of the previous dance segment to the first half of the subsequent segment, and utilize interpolation to maintain consistency, thereby achieving long-term dance generation. Motion Quality. To evaluate the motion quality of generated dance sequences, we follow the previous methods[22, 40] to calculate the Frechet Inception Distance ( FID )[10] distance between motion features of the generated dance and the ground truth dance sequences. The previous methods such as [40] calculate kinetic[34] and geometric[32] motion features using the global coordinates of all the SMPL[25] joints, which is suitable for measuring the quality of movements lasting only a few seconds. However, for longer motion sequences, where trajectories become more complex, this measurement approach focuses too heavily on root positions, neglecting local movements and resulting in data that lacks comparability. Therefore, we use the global coordinates of the root joint and the relative distances of other child joints to compute kinetic and geometric features. The kinematic feature (subscript 'k'), indicates the speed and acceleration of the movement and reflects the physical characteristics of the dance. Therefore the FID distance between kinematic features FID k measures the physical reality of the motion. The geometric feature (subscript 'g'), is calculated based on multiple predefined movement templates, thus the FID distance between geometric features FID g reflects the quality of the overall dance choreography. In addition, we follow [18] to report the Foot Skating Ratio (FSR) , which measures the proportion of frames in which either foot skids more than a certain distance while maintaining contact with the ground (foot height &lt; 5 cm).

Motion Diversity. To evaluate the motion diversity of generated dance sequences, we calculate the mean Euclidean distance within the motion feature space, as outlined in the works of Bailando[40]. DIV k represents the motion diversity in the kinematic feature space, while DIV g denotes the diversity in the geometric feature space. Table 1 reveals that our Lodge approach achieved the highest DIV g score, which can be credited to our adoption of global diffusion and characteristic dance primitives for mastering diverse choreography patterns.

Beat Alignment Score (BAS). To evaluate the beat consistency between the generated dance and the given music, we follow [22] and use the BAS to evaluate our methods, our approach demonstrated the highest Beat Alignment Score of 0.2397.

Production efficiency. In our inference process, we evaluated the average Run time taken for model generation. To ensure fairness in testing, we excluded data preprocessing time from our calculations. All experiments were conducted on the same computer equipped with an Nvidia A100 GPU and 256GB of memory.

Run Time in Table 1 presents the average Run Time required to generate 1024 frames of dance movements. Bai- lando achieved a outstanding performance, but its runtime increases linearly with the length of the sequence generated. EDGE, using the DDIM accelerated sampling strategy and linear interpolation, also achieved a fast level for generating long dance sequences. Our method uses DDPM sampling with a denoising step of 1000, taking 30.93 seconds. Using DDIM with 50 denoising steps takes only 4.57 seconds. Meanwhile, our parallel architecture ensures runtime remains stable even with longer sequences.

User study. We conducted a user study where 20 participants viewed 17 video pairs. Each pair consists of two dance sequences: one created by Lodge (DDPM) and the other by different methods or ground truth.

## 4.3. Comparisons on the AIST++ dataset

As Table 2 shows, we train Lodge on AIST++ and compare it with SOTAs. Due to the lack of long-duration dance in the AIST++ dataset, Lodge's performance does not reach the best metrics. However, compared to our baseline model EDGE, Lodge shows improvement in multiple metrics.

## 4.4. Ablation Studies

In this section, we use DDPM sampling strategy and perform ablation experiments on the FineDance dataset to evaluate the different parts: (1) the characteristic dance primitives, (2)the soft cue guidance, (3) the foot refine block.

Effect of the characteristic dance primitives. We conducted a series of ablation experiments to validate the effect of the characteristic dance primitives. In Table 3, 'C' indicates we use characteristic dance primitives to guide the local diffusion, 'M' represents we mirror the characteristic dance primitives. 'B' denotes beat alignment, we align the characterized dance primitives with the music's beats, guiding the Local diffusion to generate more expressive movements at these beat points. If beat alignment is not applied, then the characterized dance primitives are uniformly distributed across various timelines to guide the local diffusion.

The first row in Table 3 shows the results when not using characteristic dance primitives, relying solely on some d h for parallel long action generation but not using d s within a Local diffusion. This scenario leads to lower quality of motion (FID), diversity, and music rhythm alignment metrics. In contrast, rows two and three, which incorporate guidance from characteristic dance primitives, display significant improvements in Div and BAS. This improvement is because characteristic dance primitives are expressive key motions; their inclusion helps prevent the neural network from generating average, monotonous movements. The last row, achieving the optimal results, demonstrates the effectiveness of our strategy.

Effect of the Soft-cue Guidance. Our soft cue guidance weight can be adjusted using the hyperparameter 's', where a larger 's' value signifies a stronger effect. Table 4 demon- strates the outcomes resulting from setting various 's' values. With the increase in 's', there is a corresponding enhancement in FID k and Beat Alignment Score. The optimal performance is achieved when 's' is set to 1.

| Method       | Motion Quality   | Motion Quality   | Motion Quality       | Motion Diversity   | Motion Diversity   | BAS ↑   | Run Time ↓   | Wins ↑   |
|--------------|------------------|------------------|----------------------|--------------------|--------------------|---------|--------------|----------|
| Method       | FID k ↓          | FID g ↓          | Foot Skating Ratio ↓ | Div k ↑            | Div g ↑            | BAS ↑   | Run Time ↓   | Wins ↑   |
| Ground Truth | /                | /                | 6.22 %               | 9.73               | 7.44               | 0.2120  | /            | 42.6 %   |
| FACT[22]     | 113.38           | 97.05            | 28.44 %              | 3.36               | 6.37               | 0.1831  | 35.88s       | 96.7 %   |
| MNET[19]     | 104.71           | 90.31            | 39.36 %              | 3.12               | 6.14               | 0.1864  | 38.91s       | 92.3 %   |
| Bailando[40] | 82.81            | 28.17            | 18.76 %              | 7.74               | 6.25               | 0.2029  | 5.46s        | 68.2 %   |
| EDGE[47]     | 94.34            | 50.38            | 20.04 %              | 8.13               | 6.45               | 0.2116  | 8.59s        | 80.6 %   |
| Lodge (DDIM) | 50.00            | 35.52            | 2.76 %               | 5.67               | 4.96               | 0.2269  | 4.57 s       | /        |
| Lodge (DDPM) | 45.56            | 34.29            | 5.01 %               | 6.75               | 5.64               | 0.2397  | 30.93s       | /        |

Table 1. Compare with SOTAs on the FineDance dataset. Wins is the ratio of victories Lodge(DDPM) achieved in the user study.

Table 2. Compare with SOTAs on the AIST++ dataset.

| Method               | Motion Quality   | Motion Quality   | Motion Diversity   | Motion Diversity   |   BAS ↑ |
|----------------------|------------------|------------------|--------------------|--------------------|---------|
|                      | FID k ↓          | FID g ↓          | Div k ↑            | Div g ↑            |         |
| Ground Truth         | 17.10            | 10.60            | 8.19               | 7.45               |  0.2374 |
| Li et al . [21]      | 86.43            | 43.46            | 6.85               | 3.32               |  0.1607 |
| DanceNet [63]        | 69.18            | 25.49            | 2.86               | 2.85               |  0.1430 |
| DanceRevolution [15] | 73.42            | 25.92            | 3.52               | 4.87               |  0.1950 |
| FACT [22]            | 35.35            | 22.11            | 5.94               | 6.18               |  0.2209 |
| Bailando [40]        | 28.16            | 9.62             | 7.83               | 6.34               |  0.2332 |
| EDGE [47]            | 42.16            | 22.12            | 3.96               | 4.61               |  0.2334 |
| Lodge (DDPM)         | 37.09            | 18.79            | 5.58               | 4.85               |  0.2423 |

Table 3. Ablation study of the characteristic dance primitives.

| Ablations   | Ablations   | Metrics   | Metrics    | Metrics   |
|-------------|-------------|-----------|------------|-----------|
| C           | M           | FID k ↓   | Div k ↑    | BAS ↑     |
|             |             |           | 60.91 5.16 | 0.2090    |
| ✓           | ✓           | 60.20     | 5.54       | 0.2132    |
| ✓           |             |           | 52.18 5.75 | 0.2139    |
| ✓           | ✓           | 45.56     | 6.75       | 0.2397    |

Table 4. Ablation study of the soft cue guidance.

| Method       | FID k ↓   |   Div k ↑ |   BAS ↑ |
|--------------|-----------|-----------|---------|
| Ground Truth | /         |      9.73 |  0.2120 |
| s=0          | 60.91     |      5.16 |  0.2090 |
| s=0.05       | 59.66     |      5.43 |  0.2131 |
| s=0.25       | 60.51     |      5.41 |  0.2132 |
| s=0.5        | 60.46     |      5.35 |  0.2196 |
| s=0.75       | 59.89     |      5.32 |  0.2208 |
| s=0.95       | 53.63     |      5.37 |  0.2239 |
| s=1          | 45.56     |      6.75 |  0.2397 |

Effect of the Foot Refine Block. As shown in Table 5, after incorporating the Foot Refine Block, the motion quality FID k had a large improvement, especially the Foot Skating Ratio decreased from 5.94 % to 5.01 % , which proves that our proposed Foot Refine Block can effectively improve the foot-ground contact quality and reduce the probability of foot skating phenomenon.

Table 5. Ablation study of the foot refine block.

| Method                | FID k ↓   |   Div k ↑ |   BAS ↑ | Foot Skating Ratio ↓   |
|-----------------------|-----------|-----------|---------|------------------------|
| Ground Truth          | /         |      9.73 |  0.2120 | 6.22 %                 |
| w/o Foot Refine Block | 53.48     |      6.20 |  0.2216 | 5.94 %                 |
| w. Foot Refine Block  | 45.56     |      6.75 |  0.2397 | 5.01 %                 |

## 5. Conclusion and Limitation

In this work, we introduce Lodge, a two-stage coarse-to-fine diffusion network, and propose characteristic dance primitives as intermediate-level representations for the two diffusion models. Lodge has been extensively evaluated through user studies and standard metrics. Our generated samples demonstrate that Lodge can parallelly generate dances that conform to choreographic rules while preserving local details and physical realism. However, our method currently cannot generate dance movements with hand gestures or facial expressions, which are also crucial for performances. This limitation opens avenues for future research.

## Acknowledgment

This work was supported in part by the Shenzhen Key Laboratory of next generation interactive media innovative technology (No.ZDSYS20210623092001004), in part by the China Postdoctoral Science Foundation (No.2023M731957), in part by the National Natural Science Foundation of China under Grant 62306165, in part by the the Peng Cheng Laboratory (PCL2023A102), in part by the NSFC project No.62125107.

## References

- [1] [The three-phase choreographic process. https:// www.britannica.com/art/dance/The-threephase-choreographic-process . 2, 3](https://www.britannica.com/art/dance/The-three-phase-choreographic-process)
- [2] Tenglong Ao, Zeyi Zhang, and Libin Liu. Gesturediffuclip: Gesture diffusion model with clip latents. arXiv preprint arXiv:2303.14613 , 2023. 2
- [3] Alexander Berman and Valencia James. Kinetic imaginations: exploring the possibilities of combining ai and dance. In Twenty-Fourth International Joint Conference on Artificial Intelligence , page 2431-2437, 2015. 3
- [4] Lynne Anne Blom and L Tarin Chaplin. The intimate act of choreography . University of Pittsburgh Pre, 1982. 2, 3
- [5] Kang Chen, Zhipeng Tan, Jin Lei, Song-Hai Zhang, YuanChen Guo, Weidong Zhang, and Shi-Min Hu. Choreomaster: choreography-oriented music-driven dance synthesis. ACM Transactions on Graphics (TOG) , 40(4):1-13, 2021. 2, 3
- [6] Marianela Ciolfi Felice, Sarah Fdili Alaoui, and Wendy E Mackay. How do choreographers craft dance? designing for a choreographer-technology partnership. In Proceedings of the 3rd International Symposium on Movement and Computing , pages 1-8, 2016. 3
- [7] Xiao Dong, Xunlin Zhan, Yunchao Wei, Xiaoyong Wei, Yaowei Wang, Minlong Lu, Xiaochun Cao, and Xiaodan Liang. Entity-graph enhanced cross-modal pretraining for instance-level product retrieval. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2023. 3
- [8] Chunming He, Chengyu Fang, Yulun Zhang, Kai Li, Longxiang Tang, Chenyu You, Fengyang Xiao, Zhenhua Guo, and Xiu Li. Reti-diff: Illumination degradation image restoration with retinex-based latent diffusion model. arXiv preprint arXiv:2311.11638 , 2023. 3
- [9] Chunming He, Kai Li, Yachao Zhang, Yulun Zhang, Zhenhua Guo, Xiu Li, Martin Danelljan, and Fisher Yu. Strategic preys make acute predators: Enhancing camouflaged object detectors by generating camouflaged objects. 2024. 3
- [10] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems , 30, 2017. 7
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020. 3, 6
- [12] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 , 2022. 3
- [13] Sepp Hochreiter and J¨ urgen Schmidhuber. Long short-term memory. Neural computation , 9(8):1735-1780, 1997. 3
- [14] Ruozi Huang, Huang Hu, Wei Wu, Kei Sawada, Mi Zhang, and Daxin Jiang. Dance revolution: Long-term dance generation with music via curriculum learning. arXiv preprint arXiv:2006.06119 , 2020. 2
- [15] Ruozi Huang, Huang Hu, Wei Wu, Kei Sawada, Mi Zhang, and Daxin Jiang. Dance revolution: Long-term dance gen-

eration with music via curriculum learning. arXiv preprint arXiv:2006.06119 , 2020. 8

- [16] Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, and Song-Chun Zhu. Diffusionbased generation, optimization, and planning in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 16750-16761, 2023. 3
- [17] Korrawe Karunratanakul, Jinlong Yang, Yan Zhang, Michael J Black, Krikamol Muandet, and Siyu Tang. Grasping field: Learning implicit representations for human grasps. In 2020 International Conference on 3D Vision (3DV) , pages 333-344. IEEE, 2020. 2
- [18] Korrawe Karunratanakul, Konpat Preechakul, Supasorn Suwajanakorn, and Siyu Tang. Guided motion diffusion for controllable human motion synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 2151-2162, 2023. 7
- [19] Jinwoo Kim, Heeseok Oh, Seongjean Kim, Hoseok Tong, and Sanghoon Lee. A brand new dance partner: Musicconditioned pluralistic dancing controlled by multiple dance genres. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 34903500, 2022. 1, 3, 6, 8, 13
- [20] Frank Klinker. Exponential moving average versus moving exponential average. Mathematische Semesterberichte , 58: 97-107, 2011. 6
- [21] Jiaman Li, Yihang Yin, Hang Chu, Yi Zhou, Tingwu Wang, Sanja Fidler, and Hao Li. Learning to generate diverse dance motions with transformer. arXiv preprint arXiv:2008.08171 , 2020. 8
- [22] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 1340113412, 2021. 2, 3, 6, 7, 8, 13
- [23] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. Finedance: A fine-grained choreography dataset for 3d full body dance generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 1023410243, 2023. 1, 3, 6
- [24] Ronghui Li, Yuqin Dai, Yachao Zhang, Jun Li, Jian Yang, Jie Guo, and Xiu Li. Exploring multi-modal control in musicdriven dance generation. arXiv preprint arXiv:2401.01382 , 2024. 3
- [25] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multiperson linear model. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia) , 34(6):248:1-248:16, 2015. 2, 3, 7
- [26] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 11461-11471, 2022. 2, 4
- [27] Yue Ma, Yali Wang, Yue Wu, Ziyu Lyu, Siran Chen, Xiu Li, and Yu Qiao. Visual knowledge graph for human action rea-

soning in videos. In Proceedings of the 30th ACM International Conference on Multimedia , pages 4132-4141, 2022. 3

- [28] Yue Ma, Xiaodong Cun, Yingqing He, Chenyang Qi, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Magicstick: Controllable video editing via control handle transformations. arXiv preprint arXiv:2312.03047 , 2023.
- [29] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. arXiv preprint arXiv:2304.01186 , 2023. 3
- [30] Adriano Manfr` e, Ignazio Infantino, Filippo Vella, and Salvatore Gaglio. An automatic system for humanoid dance creation. Biologically Inspired Cognitive Architectures , 15:1-9, 2016. 3
- [31] Brian McFee, Colin Raffel, Dawen Liang, Daniel P Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference , pages 18-25, 2015. 3, 5
- [32] Meinard M¨ uller, Tido R¨ oder, and Michael Clausen. Efficient content-based retrieval of motion capture data. In ACM SIGGRAPH 2005 Papers , pages 677-685. 2005. 7
- [33] Ferda Ofli, Engin Erzin, Y¨ ucel Yemez, and A Murat Tekalp. Learn2dance: Learning statistical music-to-dance mappings for choreography synthesis. IEEE Transactions on Multimedia , 14(3):747-759, 2011. 1, 3
- [34] Kensuke Onuma, Christos Faloutsos, and Jessica K Hodgins. Fmdistance: A fast and effective distance function for motion capture data. In Eurographics (Short Papers) , pages 8386, 2008. 7
- [35] Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters. ACM Transactions On Graphics (TOG) , 41(4):1-17, 2022. 3
- [36] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence , 2018. 5
- [37] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 13
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨ orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684-10695, 2022. 3
- [39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems , 35:36479-36494, 2022. 5
- [40] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on
14. Computer Vision and Pattern Recognition , pages 1105011059, 2022. 1, 2, 3, 6, 7, 8, 13
- [41] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando++: 3d dance gpt with choreographic memory. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2023. 1
- [42] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 , 2020. 6
- [43] Teng Sun, Juntong Ni, Wenjie Wang, Liqiang Jing, Yinwei Wei, and Liqiang Nie. General debiasing for multimodal sentiment analysis. In Proceedings of the 31st ACM International Conference on Multimedia , pages 5861-5869. ACM, 2023. 3
- [44] Red Bull Editorial Team. How to choreograph a dance: 10 tips from the pros. https://www.redbull.com/zaen/how-to-choreograph-a-dance , 2020. 2, 3
- [45] Chen Tessler, Yoni Kasten, Yunrong Guo, Shie Mannor, Gal Chechik, and Xue Bin Peng. Calm: Conditional adversarial latent models for directable virtual characters. In ACM SIGGRAPH 2023 Conference Proceedings , pages 1-9, 2023. 3
- [46] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. arXiv preprint arXiv:2209.14916 , 2022. 2, 5
- [47] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 448-458, 2023. 2, 3, 4, 5, 6, 8, 13
- [48] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems , 30, 2017. 13
- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017. 3, 5
- [50] Yan Wu, Jiahao Wang, Yan Zhang, Siwei Zhang, Otmar Hilliges, Fisher Yu, and Siyu Tang. Saga: Stochastic wholebody grasping with contact. In European Conference on Computer Vision , pages 257-274. Springer, 2022. 2
- [51] Xingyu Xie, Pan Zhou, Huan Li, Zhouchen Lin, and Shuicheng Yan. Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. arXiv preprint arXiv:2208.06677 , 2022. 6
- [52] Zunnan Xu, Yachao Zhang, Sicheng Yang, Ronghui Li, and Xiu Li. Chain of generation: Multi-modal gesture synthesis via cascaded conditional control. arXiv preprint arXiv:2312.15900 , 2023. 3
- [53] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Qimai Li, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. arXiv preprint arXiv:2311.13231 , 2023. 3
- [54] Siqi Yang, Zejun Yang, and Zhisheng Wang. Longdancediff: Long-term dance generation with conditional diffusion model. arXiv preprint arXiv:2308.11945 , 2023. 2
- [55] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondif-

fuse: Text-driven human motion generation with diffusion model. arXiv preprint arXiv:2208.15001 , 2022. 3

- [56] Siwei Zhang, Yan Zhang, Federica Bogo, Marc Pollefeys, and Siyu Tang. Learning motion priors for 4d human body capture in 3d scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 1134311353, 2021. 2, 5
- [57] Yan Zhang and Siyu Tang. The wanderings of odysseus in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 2048120491, 2022. 3
- [58] Yuxiang Zhang, Zhe Li, Liang An, Mengcheng Li, Tao Yu, and Yebin Liu. Lightweight multi-person total motion capture using sparse multi-view cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 5560-5569, 2021. 2, 3
- [59] Yuxiang Zhang, Hongwen Zhang, Liangxiao Hu, Hongwei Yi, Shengping Zhang, and Yebin Liu. Real-time monocular full-body capture in world space via sequential proxy-tomotion learning. arXiv preprint arXiv:2307.01200 , 2023. 2, 5, 6
- [60] Kaifeng Zhao, Yan Zhang, Shaofei Wang, Thabo Beeler, and Siyu Tang. Synthesizing diverse human motions in 3d indoor scenes. arXiv preprint arXiv:2305.12411 , 2023. 3
- [61] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 57455753, 2019. 3
- [62] Haolin Zhuang, Shun Lei, Long Xiao, Weiqin Li, Liyang Chen, Sicheng Yang, Zhiyong Wu, Shiyin Kang, and Helen Meng. Gtn-bailando: Genre consistent long-term 3d dance generation based on pre-trained genre token network. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1-5. IEEE, 2023. 2, 3
- [63] Wenlin Zhuang, Congyi Wang, Siyu Xia, Jinxiang Chai, and Yangang Wang. Music2dance: Music-driven dance generation using wavenet. arXiv preprint arXiv:2002.03761 , 3(4): 6, 2020. 8
- [64] Wenlin Zhuang, Congyi Wang, Jinxiang Chai, Yangang Wang, Ming Shao, and Siyu Xia. Music2dance: Dancenet for music-driven dance generation. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) , 18(2):1-21, 2022. 3

## A. Details of the Training Process

As shown in the figure below. We trained the two stages separately to save graphics memory. The Global Diffusion is trained on long music input and sparse key motions extracted from ground truth. The output key motions of Global Diffusion are categories in d h and d s to guide the Local Diffusion only in the inference phase.

Figure 4. The Training process of Lodge.

![Figure](images/figure_0180.png)

**[Image: figure_0180.png (945x736, 248.8KB)]**

## B. Details of the Hard/Soft Diffusion Guidance

```
6 d_h = d_h.reshape([(l+1)*8,139]) 7 d_h = d_h[4:-4].reshape([l,8,139]) 8 d_s = Mirror(d_s).reshape(4l,8,139) 9 # Get music beat index by the librosa toolkit 10 beats = librosa.beatidx(m) 11 value,mask = torch.zeros([l,n,139]) 12 value[:,:4,:] = d_h[:,:4,:] 13 value[:,-4:,:] = d_h[:,-4:,:] 14 value[:, beats-4:beats+4,:] = d_s 15 mask[:,:4,:] = 1 16 mask[:,-4:,:] = 1 17 mask[:, beats-4:beats+4,:] = 1 18 def guidance_sample(m,value,mask,s): 19 d = torch.rand([l,n,139]) 20 # There are 1000 diffusion steps. 21 for i in reversed(range(0, 1000)): 22 if i > 1000*(1-s): 23 # sample d from step t to step t-1 24 d = p_sample(d, m, t) 25 # The soft-cue diffusion guidance 26 value_ = q_sample(value, t -1) 27 d = value_*mask+(1.0 - mask) * d 28 # The hard-cue diffusion guidance 29 d[:,:4] = value[:,:4]*mask[:,:4]+(1.0 mask[:,:4] )*d[:,:4] 30 d[:,-4:] = value[:,-4:] *mask [:,-4:]+(1.0-mask[:,-4:])*d[:,-4:] 31 else: 32 d = p_sample(d, m, t) 33 d[:,:4] = value[:,:4]*mask[:,:4]+(1.0mask[:,:4])*d[:,:4] 34 d[:,-4:] = value[:,-4:]*mask[:,-4:]+(1.0mask[:,-4:])*d[:,-4:] 35 36 d = d.reshape([ln, 139]) 37 return d Music Lodge L frames N frames Generated dance Parallel Lodges Number of dance frames Gggggg Lodge L frames N frames Generated dance Parallel Lodges Number of dance frames Gggggg Music Feature
```

We categorize the characteristic dance primitives generated by global diffusion into hard-cue key motions d h and softcue key motions d s . We employ distinct diffusion guidance strategies for each, enabling them to guide local diffusion.

Listing 1. Pseudocode of the Hard/Soft Diffusion Guidance

The role of d h is to guide the local diffusion in generating the initial and final segments of the dance, ensuring that the concurrently generated dance fragments can seamlessly concatenate into a coherent, long-form dance. Therefore, we adopt Hard Diffusion Guidance for this purpose.

On the other hand, d s serves to provide guidance to local diffusion. In this case, we aim for the guidance to be flexible, avoiding any disruption to the coherence of the dance generated by local diffusion. Consequently, we propose the Soft Diffusion Guidance algorithm for d s . As illustrated in the pseudocode below, our proposed soft diffusion operates only for the first 1000 × (1 -s ) steps, where s is a hyperparameter. The impact of different s values on the results is detailed in Table 3 of the main paper.

```
1 import torch, librosa 2 # m is the given music feature, m.shape = [L, 35], L is the time length 3 m = m[:ln] # l = L//n, n is the output frame number of one local diffusion 4 d_h, d_s = GlobalDiffusion(m) 5 # d_h.shape = [(l+1),8,139]; d_s.shape = [2l ,8,139]
```

Figure 5. The inference process of Lodge.

![Figure](images/figure_0190.png)

**[Image: figure_0190.png (946x646, 213.9KB)]**

## C. Details of d s and d h

Global Diffusion

Librosa Music feature CA CA Their primary distinction lies in different purposes. The soft-cue key motion use d s to guide Local Diffusion to follow the overall choreographic patterns and increase motion

TE

TE

𝑚𝑔

CA

Hard/Soft Guidance

1

g

𝑚𝑙

2

g

𝑚𝑙

3

𝑚𝑙

g

4

g

𝑚𝑙

Lodge

Lodge

CA

expressiveness. While the primarily purpose of hard-cue key motion d h is to support parallel generation. Both d s and d h are 8-frame key motions generated by Global Diffusion. d h operates at the beginning and end of Local Diffusion, employing hard diffusion guidance to ensure strict consistency with the initial and final frames of the generated motion, thereby supporting parallel generation. Meanwhile, d s operates in the middle of Local Diffusion, serving as a soft cue to improve the dance quality.

## D. Additional Ablation Studies (tested on the FineDance dataset)

## D.1. The Characteristic Dance Primitives

To reduce the computational load of Global Diffusion and to convey global choreography patterns effectively, we propose the Characteristic Dance Primitives. These primitives are dimensionalized as ( l ′ , 8 , 139) , where l ′ represents the number of dance primitives, '8' denotes the temporal dimension encompassing a continuous sequence of eight frames, and '139' corresponds to the dimensions of the motion feature. However, it is feasible to configure Dance Primitives as discrete frames. Therefore, we conducted a four-fold temporal downsampling of the ground truth dance, which is utilized to train the Global Diffusion for generating discrete dance primitives. To evaluate the relative efficacy of these methodologies, we conduct ablation experiments on the dance primitives as Table 6.

Table 6. Ablation study of the characteristic dance primitives. 'Discrete' means the dance is generated by the guidance of discrete dance primitives, 'Continuous' means the dance is generated by the guidance of continuous dance primitives

| Method       | FID k ↓   |   Div k ↑ |   BAS ↑ |
|--------------|-----------|-----------|---------|
| Ground Truth | /         |      9.73 |  0.2120 |
| Discrete     | 55.17     |      5.44 |  0.1969 |
| Continuous   | 45.56     |      6.75 |  0.2397 |

The generated motion guided by discrete dance primitives often results in incoherence, primarily due to the lack of velocity information. This issue is reflected in the increased values of the FID k [22, 40] as shown in Table 6. Furthermore, the guidance provided by these discrete dance primitives disrupts the beat consistency between music and dance, which consequently leads to a significant decline in the Beat Alignment Score (BAS)[22].

## D.2. Ablation Studies of the Hyper-parameter N and n

As described in Section 3.2 of the main paper, N represents the temporal receptive field of the Global Diffusion. The length of global music feature input into Global Diffusion is N . Meanwhile, n denotes the frame number of dance generated by the Local Diffusion.

In this part, we investigate the impact of different N and n . Thanks to our parallel architecture, Lodge can directly generate dance with ln frames, where l is a positive integer. The primary objective of these ablation experiments is to explore how different values affect dance performance.

Table 7. Ablation study of the hyper-parameter N and n .

|    N |   n |   FID k ↓ |   Div k ↑ |   BAS ↑ |
|------|-----|-----------|-----------|---------|
| 1024 | 512 |     61.66 |      8.14 |  0.1864 |
| 1024 | 256 |     45.56 |      6.75 |  0.2397 |
| 1024 | 128 |     45.86 |      5.54 |  0.2212 |
|  512 | 256 |     59.72 |      5.30 |  0.2182 |
|  512 | 128 |     46.74 |      5.76 |  0.2124 |

As shown in Table 7, when n is 512, the quality of motion, as measured by FID k , deteriorates significantly due to the network's limited capability in modeling long sequences. This also results in a substantial increase in the cost of training Local Diffusion. Comparing cases where n is 128 and 256, we observe only a marginal difference in FID k . However, crucially, we find that maintaining coherence at this value requires frequent incorporation of d h within the Hard Diffusion Guidance. Such regular intervention tends to disrupt the overall dance structure. Therefore, we ultimately set n as 256.

Comparing the second and fourth rows, it's evident that when N is set to 1024, all metrics show improved performance. Additionally, a larger N enables more comprehensive modeling of the global dependencies between music and dance. Therefore, we ultimately set N as 1024.

## E. Visualization Results

Westrongly wish you to watch the video in our project page for more details. We conducted comparisons with state-ofthe-art dance algorithms, including FACT[22], MNET[19], Bailando[40], and EDGE[47]. Both FACT and MNET are models based on the Transformer and autoregressive architecture. They encounter significant motion freezing issues during long-duration generation. After several seconds, their motion tends to freeze. Bailando is a model designed based on VQ-VAE[48] and GPT[37]. Its primary limitation lies in the encoding capacity of VQ-VAE, which restricts the network's ability to produce complex dance movements. EDGEis a model based on Diffusion and serves as the backbone of this study. Its main issue is the lack of learning global choreography patterns, resulting in noticeable incoherence at the joints and a relative monotony in the move- ments. Our method, benefiting from the Coarse-to-Fine architecture, along with the Characteristic Dance Primitives and the Foot Refine Block, is capable of generating coherent, high-quality, and expressive dance sequences.

Figure 6. Compare with the SOTAs.

![Figure](images/figure_0230.png)

**[Image: figure_0230.png (986x782, 474.9KB)]**
---

## Extracted Images

| # | File | Dimensions | Size |
|---|------|------------|------|
| 1 | figure_0003.png | 1954x644 | 353.3KB |
| 2 | figure_0044.png | 1976x1001 | 359.0KB |
| 3 | figure_0055.png | 1981x733 | 325.3KB |
| 4 | figure_0180.png | 945x736 | 248.8KB |
| 5 | figure_0190.png | 946x646 | 213.9KB |
| 6 | figure_0230.png | 986x782 | 474.9KB |
