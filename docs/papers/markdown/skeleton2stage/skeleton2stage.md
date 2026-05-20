---
source: skeleton2stage.pdf
total_pages: 12
extracted_at: 2026-05-11T23:16:07.710153
images_dir: images
---

## Skeleton2Stage: Reward-Guided Fine-Tuning for Physically Plausible Dance Generation

Jidong Jia, Youjian Zhang, Huan Fu, and Dacheng Tao, Fellow, IEEE,

Abstract -Despite advances in dance generation, most methods are trained in the skeletal domain and ignore mesh-level physical constraints. As a result, motions that look plausible as joint trajectories often exhibit body self-penetration and Foot-Ground Contact (FGC) anomalies when visualized with a human body mesh, reducing the aesthetic appeal of generated dances and limiting their real-world applications. We address this skeletonto-mesh gap by deriving physics-based rewards from the body mesh and applying Reinforcement Learning Fine-Tuning (RLFT) to steer the diffusion model toward physically plausible motion synthesis under mesh visualization. Our reward design combines (i) an imitation reward that measures a motion's general plausibility by its imitability in a physical simulator (penalizing penetration and foot skating), and (ii) a Foot-Ground Deviation (FGD) reward with test-time FGD guidance to better capture the dynamic foot-ground interaction in dance. However, we find that the physics-based rewards tend to push the model to generate freezing motions for fewer physical anomalies and better imitability. To mitigate it, we propose an anti-freezing reward to preserve motion dynamics while maintaining physical plausibility. Experiments on multiple dance datasets consistently demonstrate that our method can significantly improve the physical plausibility of generated motions, yielding more realistic and aesthetically pleasing dances. The project page is available at: Skeleton2Stage Project Page.

Index Terms -Dance generation, Reinforcement learning, Physical simulation

## I. INTRODUCTION

D ANCE is a universal art form for expressing emotions and conveying messages [1], [2], widely used in film production, game development, and virtual reality.

Despite the complexity of skeletal movements in human dances and their intricate relationship with the music condition, recent deep-learning-based methods [3]-[6] have made significant progress in generating high-quality, natural, and diverse dances that align well with the given music.

However, due to the complexity of body meshes' representation, most existing methods overlook skinned meshes when training with dance data. Instead, they typically represent actions through bone rotations in the skeletal form. Therefore, when the generated motions are visualized with human body meshes, many physically implausible phenomena might occur, such as body interpenetration and imbalanced movement. These issues significantly degrade the aesthetic appeal and realism of the final visual results [7].

J. Jia is with School of Computer Science, Shanghai Jiao Tong University, Shanghai 200240, China (jjd1123@sjtu.edu.cn).

Y. Zhang is with Bosch, Shanghai 201206, China (Youjian.ZHANG@cn. bosch.com).

H. Fu is with Youku, Alibaba, Beijing 100124, China (hufu6371@uni. sydney.edu.au).

D. Tao is with Nanyang Technological University, Singapore (dacheng.tao@ ntu.edu.sg).

To tackle the problem, we introduce physics-based rewards to incorporate physical constraints from body meshes into the generative models via RLFT. Specifically, we first train an imitation policy on expert datasets. The well-trained imitation policy can control a physically simulated character in IssacGYM [8] to mimic the given motion. Then, we use the trained policy to construct an imitation reward that evaluates the physical plausibility of the generated motion based on its imitability. The effectiveness of this reward stems directly from the simulator's constraints, which makes the imitation policy struggle, or even fail, to replicate physically implausible motions ( e.g. body penetration, foot skating), naturally yielding a lower reward. Thereby, the imitation reward can implicitly impose the mesh-based physical constraints into generative models. Besides the imitation reward that evaluates the general physical plausibility, we further introduce an FGD reward and test-time FGD guidance to improve FootGround Contact (FGC) realism, which is challenging due to the dances' dynamic foot-ground interaction.

Furthermore, during the process of RLFT, we find that the physics-based rewards tend to favor movements with small magnitude, which encourages the generative models to generate freezing motions. This happens because freezing motions are intrinsically less prone to mesh-level artifacts (e.g., self-penetration, foot skating) and are also easier for the imitation policy to track in the simulator, thereby receiving high imitation rewards. Therefore, we propose an anti-freezing reward for models to balance the preference for freezing motions. Specifically, we evaluate the magnitude of the generated motions by computing the velocity and acceleration of pose and translation parameters. By combining the antifreezing reward with physics-based rewards, our method can mitigate the bias towards freezing motions and encourage large movements while improving the physical plausibility.

To validate our method, we fine-tune the state-of-the-art (SOTA) dance diffusion models EDGE [5], POPDG [6], GENMO [9], and Bailando++ [10] and evaluate the results on the AIST++ [3] and PopDanceSet [6]. Experiments show a significant reduction in physical implausibility, such as body interpenetration and abnormal FGC. Several metrics are designed to measure these physical improvements quantitatively. Our core contributions are as follows,

- Exposing and Bridging the 'Skeleton-to-Mesh' Gap:

1

We identify and address a critical yet often-overlooked gap between the skeleton motion generation and mesh body visualization. To bridge this, we introduce Skeleton2Stage, a novel framework that employs RLFT with a carefully designed physics-based rewards system.

- Effectively leveraging physical priors from simulators: Our method provides a simple yet effective way to distill physical priors from a physics simulator into generative models. We first find that an imitation policy serves as a reliable proxy for assessing the physical plausibility of motions. We then adopt it as a physics-based reward for RLFT, thereby overcoming the simulator's nondifferentiability and successfully incorporating simulatorderived physical priors into generative models.
- Physics-Based Rewards System: We propose a multifaceted reward system for RLFT. We first use the pretrained imitation policy as a physical-aware reward, imposing general physical constraints-especially those from skinned mesh-into the RLFT. Then we introduce the FGD reward and guidance to further rectify FGC artifacts, which is challenging due to the dynamic nature of the dance. Finally, we propose an anti-freezing reward to balance the rewards' preference for freezing motions.
- SOTA Physical Plausibility and Visual Quality: Our experimental results demonstrate a significant improvement in the physical plausibility of the generated dances, including penetration and foot-ground contact. The visual quality of the generated results has also been greatly enhanced in terms of realism and aesthetics.

## II. RELATED WORK

## A. Human Motion Generation and Music to Dance

Generating realistic human motion has been extensively studied. Previous approaches [11]-[13] primarily rely on graph-based methods. They decompose motions into clips and recombine them according to predefined principles. However, these methods struggle to generate diverse human motions, especially dances that exhibit variations in speed, length, and tempos. This limitation arises from the reliance on fixed motion units and rigid composition rules. In recent years, with the emergence of deep learning and large-scale human motion datasets [3], [6], [14]-[17], numerous works have explored the use of various neural networks to generate diverse human motion [9], [18]-[23]. For instance, in the domain of musicto-dance generation, recent methods utilize various network structures, including CNNs [24], RNNs [25]-[28], GCNs [29][31], GANs [32], [33], Transformers [3], [4], [10], [17], [34][39] and Diffusion Models [5], [6], [16], [40]-[43], to better capture the intricate relationship between the joint movements and its accompanying music.

While most existing methods focus on improving the synchronization between music and human joint movements, they often overlook mesh-level constraints from the laws of physics ( e.g. human body skin collisions). Consequently, these methods tend to generate physically implausible motions when visualized with a human body mesh. In contrast, our method introduces physics-based rewards and instills physical knowledge from heuristic constraints and physical simulators into the diffusion model.

## B. Physics-Based Human motion modeling

Physics-based human motion imitation is first utilized to generate realistic and controllable locomotion for characters in the physical simulator [44]-[52]. Recent advancements have also adopted physics-based human motion imitation for more downstream tasks such as 3D human pose estimation [53]-[63] and 3D human motion generation [64]-[67].

Among these methods, [64], [67] are most related to our method. They use the imitation policy as a physically-guided motion projection module during the inference of generative models. Compared to this straightforward combination of the generative model and imitation policy, our method uses the imitation policy as a physics-based reward to fine-tune the generative model via RLFT. This allows the generative model to internalize physical constraints from the simulator and generate physically plausible motions directly. This capability yields two key benefits : i) Motion projection during inference may cause issues such as jittering or even imitation failure. However, directly generating results by the fine-tuned generative model can ensure the naturalness of the movements. ii) While motion projection with the simulator is time-consuming, our method can save the time of motion projection. Moreover, prior projection-based methods disable self-collision checks in the simulator to improve imitation success rate, which limits them to addressing only foot-ground penetration rather than enforcing full-body mesh constraints (e.g., body collision) required for physically plausible mesh visualizations.

Similarly, we may also utilize some optimization-based motion refining methods [68]-[70] during the inference of generative models as [64]. However, similar issues persist. First, these methods are optimization-based, which will reduce efficiency during inference. Second, they heavily rely on the quality and diversity of datasets. Third, these methods focus on individual frames of poses, potentially causing abrupt motion when applied to the entire sequence. Finally, they still cannot reliably enforce full-body mesh constraints, such as self-interpenetration avoidance, which are critical for human body mesh visualizations.

## C. RLFT of Diffusion Models

With the success of fine-tuning large language models (LLMs) with RL [71]-[74], recent research has proposed RLFT algorithms [75]-[79] for text-to-image diffusion models. Similarly, the field of human motion generation has also begun to adopt fine-tuning approaches to improve synthesis quality and alignment [80], [81].

In contrast to previous methods that primarily relied on data-driven or heuristic rewards, our approach uniquely incorporates a physics-based human motion imitation policy, which controls a character in the physical simulator. By leveraging RLFT, we ensure that the physical constraints ( e.g. skin collisions and gravity) of human motion are instilled into the diffusion models, offering a more robust integration of physics than previous approaches.

Fig. 1. The overview of our method. Our method formulates the denoising process as a multi-step Markov Decision Process, allowing diffusion models to be fine-tuned via RL. To incorporate physical constraints into diffusion models, we introduce physics-based rewards, including an imitation reward assessing the general physical plausibility with an imitation policy and an FGD reward to handle the dynamic nature of dance. Additionally, we design an anti-freezing reward to mitigate the physics-based rewards' preference for freezing motions.

![Figure](images/figure_0032.png)

**[Image: figure_0032.png (2065x752, 418.6KB)]**

## III. METHOD

As overviewed in Fig. 1, our method adopts the RL strategy to instill physical constraints into dance diffusion models. Firstly, an imitation policy that can effectively mimic the dancing sequence in the simulator is trained on expert datasets AMASS and AIST++ (in Section III-A). This well-trained imitation policy can then act as a reward evaluator, assessing the physical plausibility-especially those aspects arising from full-body mesh constraints-of the generated motion (in Section III-B-'Imitation Reward'). Secondly, we treat the dance diffusion denoising process as a multi-step Markov Decision (MDP) Process. This allows us to employ multiple rewards to fulfill an RLFT strategy, including the imitation reward (in Section III-B).

## A. Imitation policy for Imitation Reward

To instill real-world physical constraints into our diffusion model, we need a metric to evaluate the physical plausibility of the generated motion. Inspired by [64], we leverage an imitation policy to serve this purpose. Unlike [64], we further enable the self-collision checks and utilize the trained imitation policy as a physics-aware reward evaluator to finetune the diffusion model with RLFT. Human motion imitation policy is commonly used in robotics to control agents in the physical simulator to replicate complex movements. Since the motions replicated in the simulator inherently satisfy the various physical laws we set for the simulator (such as gravity, object collisions, friction, etc.), we argue that the more the original motion conforms to physical laws, the more accurately it should be replicated in the physical simulator. Therefore, to utilize this characteristic during the RLFT for the motion diffusion model, we train an imitation policy to control the character with SMPL skinned mesh [82].

We formulate human motion imitation as an MDP [45], defined using states (s), actions (a), transition dynamics ( T ), reward functions (r), and a discount factor ( γ ) as follows,

<!-- formula-not-decoded -->

where x t represents the current pose of the controlled character, including joint angles, joint velocities, rigid bodies' positions, rotations, etc. x res t +1 is the residual between the character's current pose and the next pose to be imitated. ψ corresponds to SMPL parameters of the character. x j t and x j t are the current pose sequence of imitated motions and reference motions ( i.e. , the motion given to the imitation policy to mimic). More notations will be described below (Actions, Transition, Rewards, and Training Strategy). In each step, the imitation policy π θ takes action based on the given state to control the character in the physical simulator to mimic the next pose of the given reference motion. Subsequently, the character's state transits based on the transition dynamics, and we can extract imitated motions from it. Finally, we calculate the reward in Equ. 1 reflecting how well the reference motion is imitated. Here, we mainly illustrate Actions, Transition, and Rewards, and we recommend [45] for a comprehensive discussion.

Actions The policy outputs an action x t +1 , representing the target joint angles of the character's pose. A PD controller then drives the character towards this target pose. Additionally, to improve the imitation ability, A residual force [45] is applied to the character's pelvis. This extra force helps to stabilize the character during movement.

Transition T is the transition dynamics of the simulator and Φ denotes the physical constraints modeled by the physical simulator, including gravity, body collision, skeletal structure, etc. Unlike previous methods [64]-[66] that utilize a humanoid as the body representation, we further embed SMPL model for a more accurate human structure, and integrate body collision handling into both the training and inference stages, Specifically, we treat each part of the SMPL model's mesh as the collision volume (as the character in red color in Fig. 1) and enable the collision check between most collision volumes. We further disable collision checks between the adjacent body parts, which can effectively prevent neighboring joints from being stuck.

Rewards The reward has two components. The first is designed to encourage the imitated motion to match the ground truth. J includes components representing human motion: local and global joint rotations, joint velocities, and 3D world joint positions. Additionally, w j and α j are the weighting factors of each reward. Finally, x j t is the t-th frame of the imitated motions represented in j, which is extracted from s t , and x j t is the t-th frame of the ground truth in the training set. The second part is a regularization term that limits the residual force, as excessively large residual forces can harm the physical realism of the character. α r and w r are weighting factors, and F r is the residual force.

## B. Physics-based Dance Diffusion Model

RLFT Formulation We introduce an RLFT strategy to distill the physical constraints of the simulator into the diffusion model. To conduct RLFT on diffusion models, recent works [76], [77] proposed to formulate the diffusion denoising process as a multi-step MDP:

<!-- formula-not-decoded -->

where t denotes the t -th decision step, while T is the total number of steps. c is the condition signal ( i.e. music sequence in our task). x T -t is the whole sequence of the denoised motion at denoising step T -t . p θ is the diffusion model being fine-tuned. The reward consists of three components: r imit , which evaluates the physical plausibility of the generated motion, r anti , which is designed to eliminate freezing issues, and r FGD , which aims to improve FGC realism. As our method involves two MDP formulations, some notation may be confusing. It is noteworthy that all the notations from Euq. 1 and Euq. 2 are independent and should be interpreted separately. For example, t in Euq. 1 denotes the time step of the MDP, while t in Euq. 2 denotes the time step of the denoising process.

With this formulation, we can use any policy-based RL algorithm, such as REINFORCE [83], to optimize diffusion models based on task-oriented rewards. To ensure training stability, we adopt a pure on-policy training strategy. Unlike off-policy training, which may leverage the data collected by older policies for better sample efficiency, we only use the data collected by the newest policy to update itself. Meanwhile, to enhance the numerical stability and convergence, we also normalize the reward to have zero mean and unit variance as in [76]. The reward's mean and standard deviation statistics are tracked for each music independently:

<!-- formula-not-decoded -->

After collecting enough trajectories, we can update the diffusion model using the policy gradient,

<!-- formula-not-decoded -->

where p θ denotes the diffusion model, and A imit ( x 0 , c ) , A anti ( x 0 , c ) , and A FGD ( x 0 , c ) are the normalized imitation, anti-freezing, and FGD rewards for guiding the optimization, with α , β , and γ as their weights.

Imitation Reward As aforementioned, we leverage the learned imitation policy as an evaluator to determine whether the generated motion of the diffusion model is physically plausible. In this section, we just use the inference process of the trained imitation policy. Specifically, starting from the first pose of a motion sequence, we repeatedly compute the transition function T (Equ. 1) to obtain the next frame of imitation, eventually generating the entire sequence of imitated motion. Ideally, if the generated motion obeys the real-world physical constraints, the imitated motion should be the exact same sequence as the generated motion. However, if the generated motion somehow violates the physical laws-for instance, an arm penetrating through the body-the imitation policy will attempt to control the character to mimic the generated motion under the physical constraints of the physical simulator and produce an imitated motion that closely approximates the input motion while eliminating the interpenetration. In this case, a tracking error inevitably arises between the generated motion and the imitated motion. This error is a direct metric for physical implausibility: the more a generated motion defies physical laws, the harder it is for the policy to imitate, leading to a larger tracking error. Consequently, we measure this error and assign larger rewards to motions with smaller tracking errors. The reward is formulated as follows:

<!-- formula-not-decoded -->

where J is a set in which each component is a specified representation of human motion, including local and global joint rotations, joint velocities, and 3D world joint positions. x j 0 is the generated motions represented in j , and ˆ x j 0 is the imitated motions. Additionally, w j and α j are the weighting factors of each reward, and these values are kept the same as the training process of the imitation policy to ensure consistency in evaluating the difference between the reference and imitated motion.

Anti-freezing Reward The imitation reward is designed to rate higher rewards for physically plausible motions, however, we find that freezing/slow-speed motions can also receive high The 'Overall' and 'Physical' columns represent the win rate of our method over the others, i.e. a higher value indicates better performance of our method. '*' means noting that FACT's good Pen. Rate, PFC, and FGD mainly come from its tendency to generate freezing motions, which is further proved by the magnitude of motion in Table IV. ' ♢ ' means directly using the official checkpoint to reproduce the result. ' ♡ ' means using the official code to reproduce the result, as the official checkpoint is unavailable or broken. ' ♣ ' means directly using the result in their paper, as the official code is unavailable or unusable. Our findings, along with recent advances [5], [6], [10], [17], [37], [42], [43], [81], [84], suggest that FID is not a reliable metric for evaluating the realism and perceptual quality of dance generation (Referring to Section V-B for a detailed discussion).

TABLE I QUANTITATIVE RESULTS ON AIST++ DATASET

| Method                    | Overall   | Physical   | FID k /FID g ↓   | Pen. Rate ↓   | PFC ↓    | FGD ↓     |   BAS ↑ | Div k /Div g →   |
|---------------------------|-----------|------------|------------------|---------------|----------|-----------|---------|------------------|
| Ours                      | /         | /          | 63.59/24.78      | 90.38         | 0.3361   | 2.5153    |  0.2991 | 1.95/4.16        |
| EDGE ♢ (CVPR'23)          | 70%       | 77.5%      | 61.67/20.77      | 176.44        | 0.8715   | 11.8547   |  0.2847 | 2.03/3.79        |
| PAMD ♡ (arXiv'25)         | 62.5%     | 67.5%      | 56.90/19.87      | 142.49        | 0.9557   | 14.6663   |  0.3026 | 2.50/3.82        |
| FACT ♢ (ICCV'21)          | 77.5%     | 75%        | 69.04/19.63      | 97.98 ∗       | 1.2125 ∗ | 19.8958 ∗ |  0.2380 | 3.07/6.87        |
| Bailando ♢ (CVPR'22)      | 85%       | 87.5%      | 26.99/10.25      | 176.36        | 1.5466   | 50.5418   |  0.2320 | 5.83/7.30        |
| Bailando++ ♢ (TPAMI'23)   | 80%       | 67.5%      | 21.71/9.66       | 130.10        | 1.9124   | 49.7011   |  0.2383 | 6.63/7.01        |
| BADM ♣ (CVPR'24)          | -         | -          | -                | -             | 1.424    | -         |  0.2366 | -                |
| Beat-it ♣ (ECCV'24)       | -         | -          | -                | -             | 0.966    | -         |   0.661 | -                |
| DanceBA ♢ (ICCV'25)       | 80%       | 70%        | 31.90/16.22      | 156.22        | 1.3845   | 48.3995   |  0.2506 | 6.69/8.13        |
| DanceChat ♣ (arXiv'25)    | -         | -          | -                | -             | 0.828    | -         |    0.27 | -                |
| OpenDanceNet ♣ (arXiv'25) | -         | -          | 30.09/9.75       | -             | 1.003    | -         |  0.2545 | -                |
| Megadance ♣ (NeurIPS'25)  | -         | -          | 25.89/12.62      | -             | -        | -         |   0.238 | 5.84/6.23        |
| Ground Truth              | 47.5%     | 42.5%      | 19.47/9.33       | 135.27        | 1.4699   | 4.9451    |  0.2292 | 8.72/7.77        |

rewards, as they intrinsically have less physical artifact and are relatively easier to imitate. Therefore, without an antifreezing reward, the diffusion model tends to increase the probability of generating freezing movements to maximize the reward during the RLFT process. To mitigate this bias, we propose an anti-freezing reward to encourage the generative model to generate more dynamic motions while maintaining the physical plausibility. Specifically, we compute the velocity and acceleration of the motion from the pose sequence, and apply the mean square value as the anti-freezing reward:

<!-- formula-not-decoded -->

Foot-Ground Deviation Reward This reward is designed to improve the foot-ground contact realism more accurately, including foot-ground penetration and floating. Specifically, the reward is designed as the distance between the lowest joint of the motion and the ground:

<!-- formula-not-decoded -->

Foot-Ground Deviation Guidance Inspired by [85], we proposed an FGD Guidance to further enhance the FGC realism of the generated dance. We first calculate the distance between the lowest joint of the motion and the ground as G ( µ t , c ) = ∥ h foot -h ground ∥ 2 . And then use its gradient to guide the predicted mean at the denoising step t. Notably, we apply this guidance based on the model's predicted contact labels, disabling it during non-contact phases, which effectively reduces foot-ground penetration and floating while preserving jump motions. Finally, we apply a post-processing step that utilizes the model's predicted contact labels. This ensures that feet remain stationary on the ground during contact phases, effectively eliminating foot-skating artifacts.

## IV. EXPERIMENTS

a) Dataset: We conduct experiments on AIST++ [3], which is the most widely used benchmark for dance motion generation. It contains 1,408 high-quality dance motions paired with music across diverse genres. We further evaluate our method on PopDanceSet [6], which includes 736 highly dynamic dance sequences. Following EDGE [5], all training samples are clipped to 5 seconds at 30 FPS.

b) Implement Details: For the imitation policy, we adopt Isaac Gym [8] as the physical simulator, in which we can detect the collision between the character's torsos. The weighting factors for calculating the reward ( w j and α j ) are set to (0.6, 0.1, 0.2, 0.1) and (60, 0.2, 100, 40), respectively. And w r and α r are 0.1 and 30. For the diffusion model, we test our RLFT method on the EDGE [5], POPDG [6], and GENMO [9] denoiser, and we employ denoising diffusion implicit models (DDIM) as proposed in [86] with 50 diffusion steps and classifier-free guidance [87]. In each fine-tuning iteration, we sample 2,048 motions from the pretrained diffusion models, conditioned on the music in the AIST++ [3] or PopDanceSet [6] training dataset. We accumulate gradients across 50 denoising steps of all samples and perform one gradient update. Our optimizer adopts the Adam [88] optimizer, with a learning rate set to 1e-6.

c) Evaluation Metrics: Similar to [84], we evaluate our physically plausible dance generation results from three key perspectives: aesthetic quality, physical plausibility, and motion-condition consistency. For aesthetic quality, we conduct a user study ('Overall' in Table I). For motion-condition consistency, we adopt the Beat Alignment Score (BAS) [4]. Regarding physical plausibility, we employ both subjective and objective metrics, including a user study ('Physical' in Table I), Penetration Rate (Pen. Rate), Physical Foot Contact EDGE

![Figure](images/figure_0068.png)

**[Image: figure_0068.png (2066x1644, 2197.0KB)]**

EDGE w/ Ours

Fig. 2. The visual comparisons of EDGE [5] and our generated motions. Both motion sequences are generated with the same music and seed. Some body parts are enlarged for a better view. The red box signifies the presence of body penetration, while the green box indicates the improvement after the RLFT. The subscript number denotes the frame number.

(PFC) [5], Foot-Ground Deviation (FGD), and Motion Magnitude (Mot. Mag.).

## A. Benchmark Performance

We compare our method with previous dance generation approaches, as summarized in Table I. For fairness, all evaluation metrics are computed on 20-second dance clips.

Human Perception Results 'Overall' and 'Physical' (Table I) denote human-perceived aesthetic quality and physical plausibility, measured as the win rate of our method over others on two questions respectively: (1) Which dance looked and felt better overall? and (2) Which dance appears more physically plausible? The study was conducted on Prolific with 40 participants under the same criteria as [5]. Skeleton2Stage surpasses other approaches, demonstrating a clear improvement in physical plausibility. It is worth noting that our method in Table I leverages the pretrained EDGE model. This means the generated dance sequences share similar motion patterns with EDGE, as shown in Fig. 2 and the supplementary videos. Nevertheless, human evaluators perceive our results as having higher overall aesthetic quality than EDGE, indicating that physical plausibility can substantially influence aesthetic perception.

Penetration Rate Pen. Rate reflects the physical plausibility of generated motions in terms of body-part penetration. As shown in Table I, our proposed method achieves a substantial improvement in this metric. Compared to the baseline EDGE, we reduce the penetration rate by 49%. Although FACT also yields a low penetration rate, this is primarily due to its tendency to generate freezing motions (as discussed in Table IV), which are aesthetically undesirable. Qualitative comparisons between EDGE and our method are provided in Fig. 2, where we can clearly observe that, after RLFT, our model replaces previously implausible motions with physically valid alternatives. For instance, in the first, third, and fourth rows of Fig. 2, the model adjusts arm movements outward to avoid interpenetration; in the second row, it learns to prevent the penetration between two hands.

Fig. 3. The visual comparison for ablation studies. Each compared dance pair is generated from the same audio track. Left compares the results of PhysDiff with those of our proposed method. As shown, motion projection can result in falling motions due to the inability to accurately imitate the physically implausible movements. The right presents an example result of the model trained without an anti-freezing reward, in which the model tends to generate small-amplitude movements.

![Figure](images/figure_0076.png)

**[Image: figure_0076.png (2068x551, 629.3KB)]**

Foot-Ground Contact Score We evaluate both the Physical Foot Contact (PFC) [5] and Foot-Ground Deviation (FGD) scores. The former measures the overall plausibility of foot-ground contact, while the latter is specifically designed to quantify foot-ground penetration and floating. As shown in Table I, our method significantly outperforms previous works, demonstrating that our approach effectively enhances the physical realism of foot-ground interactions. The PFC metric is grounded in the principle that the character in the physical simulator should obey Newton's laws of motion, which form the theoretical basis of its definition.

Other Non-physical Metrics We report the results of other non-physical metrics in Table I to demonstrate that our method does not deteriorate in aesthetic performance, such as beat alignment score (BAS) and diversity (Divk, Divg). Since our approach is not specifically designed to address beat alignment or diversity, it achieves results comparable to EDGE. It seems that diffusion-based methods tend to yield lower diversity scores when generating long dance sequences. We attribute this to the way diffusion models generate long sequences-by stitching together multiple short clips. Diversity is then computed as the average across these clips; thus, while each short clip may exhibit high diversity, the overall composition can appear less diverse.

## B. Ablation Study

RLFT vs. Motion Projection While we use the imitation policy as an imitation reward to fine-tune diffusion models, it can also be applied as a motion projection (PhysDiff [64]) during the denoising process. We compare our method and PhysDiff, both built upon EDGE.

In our preliminary experiments, we observed that applying the physics-based projection consecutively four times (PhysDiff*(4step)) at the end of the diffusion process (the original setting in [64]) amplifies the errors caused by physically

TABLE II RLFT VS. MOTION PROJECTION

| Method           |   Pen. Rate |   PFC ↓ |   FGD ↓ | Succ. Rate ↑   |
|------------------|-------------|---------|---------|----------------|
| EDGE             |      176.44 |  0.8715 | 11.8547 | 100%           |
| PhysDiff*(1step) |      115.65 |  1.6240 | 11.7289 | 95%            |
| PhysDiff*(4step) |       88.81 |  3.7823 | 34.6713 | 45%            |
| Ours             |       90.38 |  0.3361 |  2.5153 | 100%           |

* indicates that we reproduced PhysDiff, as the official code is not available. Key differences between RLFT and motion projection (proposed in PhysDiff) are discussed in the supplementary materials.

implausible motions, making imitation more prone to failure, as shown in Fig. 3(left) and Table II. Due to the extensive body interactions in dance movements, this issue occurs more frequently than in text-conditioned motion generation. Therefore, we adopt a setting that performs the physicsbased projection once to achieve better stability, which we denote as 'PhysDiff*(1step)'. Nevertheless, imitation failures still appear in some casese.g. , one leg passes through the supporting leg-leading to chaotic outcomes such as slipping or falling (see Fig. 3(left)). We infer that the performance of PhysDiff strongly depends on the success of motion imitation; however, certain penetrated motions cannot be faithfully imitated, resulting in severe failures.

The quantitative results in Table II further confirm that although the penetration rate of PhysDiff*(1step) decreases due to collision checking in the physical simulator, its PFC metric deteriorates significantly. This may be attributed to abnormal foot movements: (i) in imitation failure cases, the feet lose contact with the ground; and (ii) even without falling, when the generated motions exhibit foot sliding, the imitation policy still tends to keep tracking it, which drags the feet and induces jittery stick-slip behavior due to friction in the physical simulator. In contrast, our method remains stable by assigning very low rewards to such 'failing' and foot-sliding scenarios. Moreover, unlike direct imitation, which merely reproduces similar motion patterns, our approach gradually learns to avoid these challenging cases by replacing them with physically plausible motions that are, by nature, more

TABLE III THE VALIDATION OF THE GENERALITY OF OUR METHOD

| Method                                                | Venue      | Dataset     | Model     | Pen. Rate ↓          | PFC ↓                | FGD ↓                   | BAS ↑                | Div k /Div g →                |
|-------------------------------------------------------|------------|-------------|-----------|----------------------|----------------------|-------------------------|----------------------|-------------------------------|
| POPDG POPDG w/ Ours                                   | CVPR 2024  | PopDanceSet | Diffusion | 245.08 119.41        | 1.9265 1.2445        | 17.7154 1.1983          | 0.2410 0.2554        | 4.64/6.82 4.95/6.30           |
| Bailando++ w/o RL Bailando++ w/ BA Bailando++ w/ Ours | TPAMI 2023 | AIST++      | VQVAE+GPT | 136.62 130.10 112.49 | 1.5474 1.9124 1.2722 | 46.4480 49.7011 34.2014 | 0.2204 0.2383 0.2468 | 7.49/6.78 6.63/7.01 7.56/7.08 |
| GENMO* GENMO* w/ Ours                                 | ICCV 2025  | AIST++      | Diffusion | 228.04 138.78        | 0.5944 0.1703        | 34.9455 5.2832          | 0.2355 0.2309        | 3.08/4.47 2.81/4.04           |

* indicates that we reproduce the results based on their paper, as they didn't release the correct checkpoint, training, and evaluation guidance (see Section V-D). To ensure a fair comparison on Bailando++, we use its original RL method, modifying only the reward function. The baseline, termed 'bailando++ w/o RL', is the pretrained GPT model from their work before RL finetuning. Then we evaluate on two different reward configurations: (1) 'Bailando++ w/ BA' is finetuned by a beat alignment (BA) reward, the same as in their paper; (2) 'Bailando++ w/ Ours' is finetuned by our proposed rewards.

TABLE IV ANTI-FREEZING REWARD ABLATION.

| Method       |   Mot. Mag. ↑ |
|--------------|---------------|
| EDGE         |        0.4744 |
| FACT         |        0.2148 |
| Ours w/o AF  |        0.3362 |
| Ours         |        0.4670 |
| Ground Truth |        0.4881 |

imitatable by the physics-based imitation policy-a key advantage introduced by RLFT. Additional qualitative results are provided in the supplementary materials.

Other Physical Rewards Here, we do ablation studies to validate the effectiveness of other rewards.

Anti-freezing Reward: As mentioned earlier, physics-based rewards, while effective at addressing physical issues, often drive the model to produce freezing motions, since static poses inherently avoid physical violations. To mitigate this, we develop an Anti-Freezing (AF) reward. We evaluate its effectiveness using the motion magnitude (Mot. Mag.), defined as the average temporal difference of poses across the entire sequence. This metric reflects the degree of motion freezing in the generated sequences and should not be too low, as an excessively small value indicates freezing behavior. As reported in Table IV, Mot. Mag. reduces significantly without anti-freezing reward ( i.e. Ours w/o AF). Fig. 3(right) also shows the same conclusion that it tends to generate freezing motions without anti-freezing reward, which is undesirable in practice.

FGD Reward&amp;Guidance: We conduct ablation studies to validate the effectiveness of the FGD reward and guidance. As shown in Table V, our method produces more accurate foot-ground contact with FDG reward and guidance.

Generalization We further apply our method to several other works, including POPDG [6], GENMO [9], and Bailando++ [10]. For POPDG, we evaluate on their proposed dataset, which they indicate contains more complex dance motions. For the other two methods, we evaluate on the AIST++ dataset. As reported in Table III, integrating our method leads to consistent improvements across all cases. It demonstrates the generalization capability of our method.

TABLE V FGD REWARD AND GUIDANCE ABLATION.

| Method         |   PFC ↓ |   FGD ↓ |
|----------------|---------|---------|
| EDGE           |  0.8715 | 11.8547 |
| Ours w/o R.&G. |  0.8728 | 11.1405 |
| Ours w/o G.    |  0.7928 | 10.6357 |
| Ours           |  0.3361 |  2.5153 |
| Ground Truth   |  1.4699 |  4.9451 |

## V. CONCLUSIONS AND DISCUSSIONS

In this paper, we present Skeleton2Stage, a novel framework that addresses the physical implausibility issue caused by the skeleton-to-mesh gap in dance generation methods through RLFT. Specifically, we design a physics-based rewards system to instill physical laws derived from both physical simulators and heuristic constraints( e.g. body collisions, gravity, and friction) into the pretrained diffusion model via RLFT. The system comprises three key components: (1) an imitation reward evaluating general physical plausibility with an imitation policy, (2) an FGD reward and guidance to handle complex foot-ground contact in dances, and (3) an anti-freezing reward mitigating models' tendency to generate freezing motions. Extensive experiments demonstrate that our method significantly improves the physical plausibility of generated motions across multiple metrics, particularly in reducing body penetration and enhancing FGC realism. These results establish our method as an effective framework for integrating rich physical priors from physical simulators into generative models, thereby improving the aesthetic appeal of the generated dances when visualized with a human body mesh and advancing their potential for real-world applications. In the future, we aim to further refine both the physical plausibility and smoothness of generated motions.

## A. More metrics

To better evaluate the quality of generated motions, we add three more metrics to evaluate the results on the AIST++ dataset. The first metric is MotionCritic Score proposed in [81], which uses a pretrained network to rate the motion quality. The second metric is NRDF Score proposed in [69], which measures the distance of a pose to the manifold of plausible articulated poses. The third metric is FID. However, the feature extractor we use is a pretrained motion autoencoder as in [89]. As shown in Table VI, our method achieves the best MotionCritic Score and NRDF Score, while the FID is slightly worse than EDGE, which may be due to the distribution shift during RL-finetuning.

TABLE VI FID, MOTIONCRITIC SCORE, AND NRDF SCORE ON AIST++ DATASET

| Method     |   FID [89] ↓ |   MotionCritic [81] |   NRDF [69] |
|------------|--------------|---------------------|-------------|
| Ours       |       0.5420 |                2.26 |      1.9881 |
| EDGE       |       0.4988 |               -4.10 |      2.5036 |
| PAMD       |       0.6193 |               -5.00 |      2.1659 |
| FACT       |       0.5829 |               -4.87 |      2.0825 |
| Bailando   |       0.6072 |               -9.88 |     24.2318 |
| Bailando++ |       0.5866 |               -2.98 |      3.2060 |
| DanceBA    |       0.5563 |               -3.74 |      8.8400 |

## B. Discussion About the FID

Many previous works [5], [6], [10], [17], [37], [42], [43], [81], [84] have shown that FID is not a good metric to measure the quality of the generated motions in Dance Generation. We also observed that FID is not a reliable metric for motion quality. For instance, when we fine-tuned POPDG with a higher weighted anti-freezing reward, the FIDk / FIDg score improved to 73.72/29.40 (origin POPDG 76.23/33.37), while the PFC degraded to 2.5438 (origin POPDG 1.9265). This negative correlation underscores FID's unreliability. Moreover, we find that the POPDG training set has a high PFC of 17.2608, as its ground truth motions were generated through motion estimation, which introduces jitter. This causes FID to develop a preference for similarly jittery motions, making it an unreliable metric for dance quality. From Table VII, the fact that PAMD has worse PFC and better FID compared to EGDE also shows the unreliability of FID. Furthermore, we also plot the change of FIDk and PFC during our fine-tuning process of EDGE. As shown in Fig. 4, the PFC and FIDk also exhibit a negative correlation, which further suggests that FID may not be a reliable metric for evaluating dance quality.

TABLE VII FID METRICS

| Method         | FID k /FID g   |
|----------------|----------------|
| EDGE+Ours(20s) | 63.59/24.78    |
| EDGE(20s)      | 61.67/20.77    |
| EDGE+Ours(5s)  | 48.18/19.38    |
| EDGE(5s)       | 47.66/16.23    |
| PAMD           | 56.90/19.87    |
| FACT           | 69.04/19.63    |
| Bailando       | 26.99/10.25    |
| Bailando++     | 21.71/9.66     |
| DanceBA        | 31.90/16.22    |

Therefore, following [5], [84], we use user study, PFC, music dance alignment, and other physical metrics to measure the motion quality. And we also provide more metrics in Section V-A to prove the effectiveness of our method.

                                                                      

Fig. 4. The performance of FIDk and PFC during the fine-tuning process of EDGE.

![Figure](images/figure_0112.png)

**[Image: figure_0112.png (994x545, 104.1KB)]**

## C. Advantages Against PhysDiff

The key differences between our method and PhysDiff are shown in Fig. 5. Our method instills physical constraints from the simulator into the diffusion model, eliminating the need for an imitation policy during inference.

Fig. 5. Key differences: PhysDiff vs Ours.

![Figure](images/figure_0116.png)

**[Image: figure_0116.png (1010x407, 132.5KB)]**

Thus, the advantages of our method against PhysDiff in dance generation are two folds.

First, in preliminary experiments, we find that though the imitation policy is well-trained on motion datasets, the ability and generalization of imitation policy are limited, which will cause jittering and failure cases. Furthermore, due to the complex nature of dancing motion and the body collision detection introduced in the simulator, the imitation policy becomes more challenging to succeed. As shown in Table IX, the success rate is 95% for 1 step projection. Moreover, the imitation error will accumulate during the diffusion process. After iterative projection for 4 times (same setting as original PhysDiff), the success rate will decrease significantly to 45%, which seriously affected the visual quality. That is also why we did not follow the exact settings of PhysDiff. However, we are able to eliminate these artifacts in our proposed method.

Second, the motion projection is time-consuming. As shown in Table IX, our method can save a significant amount of time, making it more computationally efficient.

Moreover, collision detection is disabled in the original paper. In our reproduction, we enable collision checking in

TABLE VIII THE REPRODUCTION OF GENMO

| Method         |   Pen. Rate ↓ |   PFC ↓ |   FGD ↓ |   BAS ↑ |   Mot. Mag. → | Div k /Div g →   |
|----------------|---------------|---------|---------|---------|---------------|------------------|
| GENMO ♢        |        146.71 |  0.2904 | 22.7392 |  0.2013 |        0.2374 | 1.96/5.00        |
| GENMO*         |        228.04 |  0.5944 | 34.9455 |  0.2355 |        0.4144 | 3.08/4.47        |
| GENMO* w/ Ours |        138.78 |  0.1703 |  5.2832 |  0.2309 |        0.3961 | 2.81/4.04        |

♢ : Results from the official checkpoint, which tends to generate freezing dances. *: Results from our re-implementation based on the paper.

the physics simulator to enforce full-body mesh constraints, which is important for bridging the skeleton-to-mesh gap.

TABLE IX ADVANTAGE ANAGINST PHYSDIFF*

| Method           |   PFC ↓ | Succ. Rate ↑   | Infer. Time ↓   |
|------------------|---------|----------------|-----------------|
| PhysDiff*(1step) |  1.6240 | 95%            | 30s             |
| PhysDiff*(4step) |  3.7823 | 45%            | 120s            |
| Ours             |  0.3361 | 100%           | 3s              |

## D. The Reproduction of GENMO

Although GENMO has released a checkpoint, it appears to be incorrect, and they have not provided detailed training and evaluation guidance. Therefore, we reproduced the model from scratch based on the original paper and applied our proposed method to GENMO*, our reproduction. As shown in Table VIII, the official model (GENMO ♢ ) exhibits better physical plausibility (lower Pen. Rate and PFC) compared to our reproduction (GENMO*), but scores lower on the BAS metric. We attribute this trade-off to the tendency of GENMO ♢ to generate freezing motions, proved by its significantly lower Motion Magnitude (Mot. Mag.).

## E. Why Dance Generation?

We will further elaborate on the significance of selecting dance generation as the research focus. First, compared to general motion generation tasks, there are more interactions between different body parts in dance generation, e.g. hand and body contact, causing more frequent self-collision phenomena. This phenomenon severely affects the visual quality of the dance generation when visualized with a human body mesh. Therefore, we think it is more challenging and meaningful to obtain a physically plausible motion in the task of dance generation. Second, in addition to physical plausibility, the dance generation also demands high aesthetic quality. Therefore, we propose an anti-freezing reward specifically for the dance generation, as trivial static movements can diminish the visual quality of the generated dance.

## REFERENCES

- [1] K. LaMothe, 'The dancing species: how moving together in time helps make us human,' Aeon, June , vol. 1, 2019.
- [2] G. Zambrano, 'Why people dance?' 2023.
- [3] R. Li, S. Yang, D. A. Ross, and A. Kanazawa, 'Learn to dance with aist++: Music conditioned 3d dance generation,' 2021.
- [4] L. Siyao, W. Yu, T. Gu, C. Lin, Q. Wang, C. Qian, C. C. Loy, and Z. Liu, 'Bailando: 3d dance generation by actor-critic gpt with choreographic memory,' 2022.
- [5] J. Tseng, R. Castellon, and K. Liu, 'Edge: Editable dance generation from music,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 448-458.
- [6] Z. Luo, M. Ren, X. Hu, Y. Huang, and L. Yao, 'Popdg: Popular 3d dance generation with popdanceset,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , June 2024, pp. 26 984-26 993.
- [7] L. Hoyet, R. McDonnell, and C. O'Sullivan, 'Push it real: Perceiving causality in virtual interactions,' ACM Transactions on Graphics (TOG) , vol. 31, no. 4, pp. 1-9, 2012.
- [8] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State, 'Isaac gym: High performance gpu-based physics simulation for robot learning,' 2021.
- [9] J. Li, J. Cao, H. Zhang, D. Rempe, J. Kautz, U. Iqbal, and Y. Yuan, 'Genmo: A generalist model for human motion,' in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , 2025.
- [10] L. Siyao, W. Yu, T. Gu, C. Lin, Q. Wang, C. Qian, C. C. Loy, and Z. Liu, 'Bailando++: 3d dance gpt with choreographic memory,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 45, no. 12, pp. 14 192-14 207, 2023.
- [11] J. Lee, J. Chai, P. S. A. Reitsma, J. K. Hodgins, and N. S. Pollard, 'Interactive control of avatars animated with human motion data,' ACM Trans. Graph. , vol. 21, no. 3, p. 491-500, jul 2002. [Online]. Available: https://doi.org/10.1145/566654.566607
- [12] L. Kovar, M. Gleicher, and F. Pighin, 'Motion graphs,' ACM Trans. Graph. , vol. 21, no. 3, p. 473-482, jul 2002. [Online]. Available: https://doi.org/10.1145/566654.566605
- [13] O. Arikan and D. A. Forsyth, 'Interactive motion generation from examples,' ACM Transactions on Graphics (TOG) , vol. 21, no. 3, pp. 483-490, 2002.
- [14] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, 'AMASS: Archive of motion capture as surface shapes,' in International Conference on Computer Vision , Oct. 2019, pp. 5442-5451.
- [15] J. Lin, A. Zeng, S. Lu, Y. Cai, R. Zhang, H. Wang, and L. Zhang, 'Motion-x: A large-scale 3d expressive whole-body human motion dataset,' Advances in Neural Information Processing Systems , 2023.
- [16] R. Li, J. Zhao, Y. Zhang, M. Su, Z. Ren, H. Zhang, Y. Tang, and X. Li, 'Finedance: A fine-grained choreography dataset for 3d full body dance generation,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 10 234-10 243.
- [17] J. Zhang, Z. Kang, and Y. Wang, 'Opendance: Multimodal controllable 3d dance generation using large-scale internet data,' arXiv preprint arXiv:2506.07565 , 2025.
- [18] J. Sun, C. Wang, H. Hu, H. Lai, Z. Jin, and J.-F. Hu, 'You never stop dancing: Non-freezing dance generation via bank-constrained manifold projection,' in Advances in Neural Information Processing Systems , S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 9995-10 007. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2022/file/40bfe6177e8aed33c982264cf9e6e62c-Paper-Conference.pdf
- [19] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-or, and A. H. Bermano, 'Human motion diffusion model,' in The Eleventh International Conference on Learning Representations , 2023. [Online]. Available: https://openreview.net/forum?id=SJ1kSyO2jwu
- [20] B. Jiang, X. Chen, W. Liu, J. Yu, G. Yu, and T. Chen, 'Motiongpt: Human motion as a foreign language,' in Advances in Neural Information Processing Systems , A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 20 067-20 079.

- [[Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2023/file/3fbf0c1ea0716c03dea93bb6be78dd6f-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/3fbf0c1ea0716c03dea93bb6be78dd6f-Paper-Conference.pdf)
- [21] H. Liang, J. Bao, R. Zhang, S. Ren, Y. Xu, S. Yang, X. Chen, J. Yu, and L. Xu, 'Omg: Towards open-vocabulary motion generation via mixture of controllers,' 2024. [Online]. Available: https://arxiv.org/abs/ 2312.08985
- [22] W. Dai, L.-H. Chen, J. Wang, J. Liu, B. Dai, and Y. Tang, 'Motionlcm: Real-time controllable motion generation via latent consistency model,' 2024. [Online]. Available: https://arxiv.org/abs/2404.19759
- [23] L. Xiao, S. Lu, H. Pi, K. Fan, L. Pan, Y. Zhou, Z. Feng, X. Zhou, S. Peng, and J. Wang, 'Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space,' in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , October 2025, pp. 10 086-10 096.
- [24] D. Holden, J. Saito, and T. Komura, 'A deep learning framework for character motion synthesis and editing,' ACM Trans. Graph. , vol. 35, no. 4, jul 2016. [Online]. Available: https://doi.org/10.1145/2897824. 2925975
- [25] T. Tang, J. Jia, and H. Mao, 'Dance with melody: An lstm-autoencoder approach to music-oriented dance synthesis,' in Proceedings of the 26th ACM International Conference on Multimedia , ser. MM '18. New York, NY, USA: Association for Computing Machinery, 2018, p. 1598-1606. [Online]. Available: https://doi.org/10.1145/3240508.3240526
- [26] N. Yalta, S. Watanabe, K. Nakadai, and T. Ogata, 'Weakly supervised deep recurrent neural networks for basic dance step generation,' 2019.
- [27] O. Alemi, J. Franc ¸oise, and P. Pasquier, 'Groovenet: Real-time musicdriven dance movement generation using artificial neural networks,' networks , vol. 8, no. 17, p. 26, 2017.
- [28] R. Huang, H. Hu, W. Wu, K. Sawada, M. Zhang, and D. Jiang, 'Dance revolution: Long-term dance generation with music via curriculum learning,' 2023.
- [29] S. Yan, Z. Li, Y. Xiong, H. Yan, and D. Lin, 'Convolutional sequence generation for skeleton-based action synthesis,' in 2019 IEEE/CVF International Conference on Computer Vision (ICCV) , 2019, pp. 43934401.
- [30] X. Ren, H. Li, Z. Huang, and Q. Chen, 'Self-supervised dance video synthesis conditioned on music,' in Proceedings of the 28th ACM International Conference on Multimedia , ser. MM '20. New York, NY, USA: Association for Computing Machinery, 2020, p. 46-54. [Online]. Available: https://doi.org/10.1145/3394171.3413932
- [31] J. P. Ferreira, T. M. Coutinho, T. L. Gomes, J. F. Neto, R. Azevedo, R. Martins, and E. R. Nascimento, 'Learning to dance: A graph convolutional adversarial network to generate realistic dance motions from audio,' Computers &amp; Graphics , vol. 94, pp. 11-21, 2021.
- [32] H.-Y. Lee, X. Yang, M.-Y. Liu, T.-C. Wang, Y.-D. Lu, M.-H. Yang, and J. Kautz, 'Dancing to music,' Advances in neural information processing systems , vol. 32, 2019.
- [33] G. Sun, Y. Wong, Z. Cheng, M. S. Kankanhalli, W. Geng, and X. Li, 'Deepdance: music-to-dance motion choreography with adversarial learning,' IEEE Transactions on Multimedia , vol. 23, pp. 497-509, 2020.
- [34] B. Li, Y. Zhao, Z. Shi, and L. Sheng, 'Danceformer: Music conditioned 3d dance generation with parametric motion transformer,' 2023.
- [35] J. Li, Y . Yin, H. Chu, Y . Zhou, T. Wang, S. Fidler, and H. Li, 'Learning to generate diverse dance motions with transformer,' arXiv preprint arXiv:2008.08171 , 2020.
- [36] X. Li, R. Li, S. Fang, S. Xie, X. Guo, J. Zhou, J. Peng, and Z. Wang, 'Music-aligned holistic 3d dance generation via hierarchical motion modeling,' 2025. [Online]. Available: https://arxiv.org/abs/2507.14915
- [37] Q. Wang, X. Yang, Y. Dong, N. R. Govindaraj, G. Slabaugh, and S. Yuan, 'Dancechat: Large language model-guided music-to-dance generation,' arXiv preprint arXiv:2506.10574 , 2025.
- [38] K. Yang, X. Tang, Z. Peng, Y. Hu, J. He, and H. Liu, 'Megadance: Mixture-of-experts architecture for genre-aware 3d dance generation,' arXiv preprint arXiv:2505.17543 , 2025.
- [39] C. Fan, J. Guan, X. Zhao, D. Xu, Y. Lin, T. Ye, P. Feng, and H. Pan, 'Align your rhythm: Generating highly aligned dance poses with gating-enhanced rhythm-aware feature representation,' arXiv preprint arXiv:2503.17340 , 2025.
- [40] S. Alexanderson, R. Nagy, J. Beskow, and G. E. Henter, 'Listen, denoise, action! audio-driven motion synthesis with diffusion models,' ACM Trans. Graph. , vol. 42, no. 4, pp. 44:1-44:20, 2023.
- [41] R. Li, Y. Zhang, Y. Zhang, H. Zhang, J. Guo, Y. Zhang, Y. Liu, and X. Li, 'Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives,' in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR) , 2024.
- [42] Z. Huang, X. Xu, C. Xu, H. Zhang, C. Zheng, J. Qin, and S. He, 'Beatit: Beat-synchronized multi-condition 3d dance generation,' in European conference on computer vision . Springer, 2024, pp. 273-290.
- [43] C. Zhang, Y. Tang, N. Zhang, R.-S. Lin, M. Han, J. Xiao, and S. Wang, 'Bidirectional autoregessive diffusion model for dance generation,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 687-696.
- [44] L. Liu and J. Hodgins, 'Learning to schedule control fragments for physics-based characters using deep q-learning,' ACM Transactions on Graphics (TOG) , vol. 36, no. 3, pp. 1-14, 2017.
- [45] Y. Yuan and K. Kitani, 'Residual force control for agile human behavior imitation and extended motion synthesis,' Advances in Neural Information Processing Systems , vol. 33, pp. 21 763-21 774, 2020.
- [46] L. Liu and J. Hodgins, 'Learning basketball dribbling skills using trajectory optimization and deep reinforcement learning,' ACM Transactions on Graphics (TOG) , vol. 37, no. 4, pp. 1-14, 2018.
- [47] X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne, 'Deepmimic: Example-guided deep reinforcement learning of physics-based character skills,' ACM Transactions On Graphics (TOG) , vol. 37, no. 4, pp. 1-14, 2018.
- [48] Z. Wang, J. Merel, S. Reed, G. Wayne, N. de Freitas, and N. Heess, 'Robust imitation of diverse behaviors,' 2017.
- [49] J. Merel, Y. Tassa, D. TB, S. Srinivasan, J. Lemmon, Z. Wang, G. Wayne, and N. Heess, 'Learning human behaviors from motion capture by adversarial imitation,' 2017.
- [50] J. Won, D. Gopinath, and J. Hodgins, 'A scalable approach to control diverse behaviors for physically simulated characters,' ACM Trans. Graph. , vol. 39, no. 4, aug 2020. [Online]. Available: https://doi.org/10.1145/3386569.3392381
- [51] S. Park, H. Ryu, S. Lee, S. Lee, and J. Lee, 'Learning predictand-simulate policies from unorganized human motion data,' ACM Trans. Graph. , vol. 38, no. 6, nov 2019. [Online]. Available: https://doi.org/10.1145/3355089.3356501
- [52] K. Bergamin, S. Clavet, D. Holden, and J. R. Forbes, 'Drecon: data-driven responsive control of physics-based characters,' ACM Trans. Graph. , vol. 38, no. 6, nov 2019. [Online]. Available: https://doi.org/10.1145/3355089.3356536
- [53] P. Zell, B. Wandt, and B. Rosenhahn, 'Joint 3d human motion capture and physical analysis from monocular videos,' in 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) , 2017, pp. 17-26.
- [54] D. Rempe, L. J. Guibas, A. Hertzmann, B. Russell, R. Villegas, and J. Yang, 'Contact and human dynamics from monocular video,' 2020. [Online]. Available: https://arxiv.org/abs/2007.11678
- [55] S. Shimada, V. Golyanik, W. Xu, and C. Theobalt, 'Physcap: Physically plausible monocular 3d motion capture in real time,' 2020. [Online]. Available: https://arxiv.org/abs/2008.08880
- [56] S. Shimada, V. Golyanik, W. Xu, P. P´ erez, and C. Theobalt, 'Neural monocular 3d human motion capture with physical awareness,' 2021. [Online]. Available: https://arxiv.org/abs/2105.01057
- [57] Y. Yuan and K. Kitani, '3d ego-pose estimation via imitation learning,' in Proceedings of the European Conference on Computer Vision (ECCV) , 2018, pp. 735-750.
- [58] --, 'Ego-pose estimation and forecasting as real-time pd control,' 2019. [Online]. Available: https://arxiv.org/abs/1906.03173
- [59] M. Isogawa, Y. Yuan, M. O'Toole, and K. Kitani, 'Optical non-lineof-sight physics-based 3d human pose estimation,' 2020. [Online]. Available: https://arxiv.org/abs/2003.14414
- [60] X. Yi, Y. Zhou, M. Habermann, S. Shimada, V. Golyanik, C. Theobalt, and F. Xu, 'Physical inertial poser (pip): Physics-aware real-time human motion tracking from sparse inertial sensors,' 2022. [Online]. Available: https://arxiv.org/abs/2203.08528
- [61] Y. Yuan, S.-E. Wei, T. Simon, K. Kitani, and J. Saragih, 'Simpoe: Simulated character control for 3d human pose estimation,' 2021. [Online]. Available: https://arxiv.org/abs/2104.00683
- [62] Z. Luo, S. Iwase, Y. Yuan, and K. Kitani, 'Embodied sceneaware human pose estimation,' 2022. [Online]. Available: https: //arxiv.org/abs/2206.09106
- [63] Z. Luo, R. Hachiuma, Y. Yuan, and K. Kitani, 'Dynamics-regulated kinematic policy for egocentric pose estimation,' 2022. [Online]. Available: https://arxiv.org/abs/2106.05969
- [64] Y. Yuan, J. Song, U. Iqbal, A. Vahdat, and J. Kautz, 'Physdiff: Physicsguided human motion diffusion model,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 16 010-16 021.
- [65] H. Yao, Z. Song, Y. Zhou, T. Ao, B. Chen, and L. Liu, 'Moconvq: Unified physics-based motion control via scalable discrete

representations,' 2023. [Online]. Available: https://arxiv.org/abs/2310. 10198

- [66] N. Gillman, M. Freeman, D. Aggarwal, C.-H. Hsu, C. Luo, Y. Tian, and C. Sun, 'Self-correcting self-consuming loops for generative model training,' 2024.
- [67] Z. Li, M. Luo, R. Hou, X. Zhao, H. Liu, H. Chang, Z. Liu, and C. Li, 'Morph: A motion-free physics optimization framework for human motion generation,' 2025. [Online]. Available: https: //arxiv.org/abs/2411.14951
- [68] G. Tiwari, D. Antic, J. E. Lenssen, N. Sarafianos, T. Tung, and G. PonsMoll, 'Pose-ndf: Modeling human pose manifolds with neural distance fields,' in European Conference on Computer Vision (ECCV) , October 2022.
- [69] Y. He, G. Tiwari, T. Birdal, J. E. Lenssen, and G. Pons-Moll, 'Nrdf: Neural riemannian distance fields for learning articulated pose priors,' in Conference on Computer Vision and Pattern Recognition (CVPR) , 2024.
- [70] H. Wang, Y. Zhu, Q. Lai, Y. Zhang, G.-S. Xie, and X. Geng, 'Pamd: Plausibility-aware motion diffusion model for long dance generation,' 2025. [Online]. Available: https://arxiv.org/abs/2505.20056
- [71] Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, C. Chen, C. Olsson, C. Olah, D. Hernandez, D. Drain, D. Ganguli, D. Li, E. Tran-Johnson, E. Perez, J. Kerr, J. Mueller, J. Ladish, J. Landau, K. Ndousse, K. Lukosuite, L. Lovitt, M. Sellitto, N. Elhage, N. Schiefer, N. Mercado, N. DasSarma, R. Lasenby, R. Larson, S. Ringer, S. Johnston, S. Kravec, S. E. Showk, S. Fort, T. Lanham, T. Telleen-Lawton, T. Conerly, T. Henighan, T. Hume, S. R. Bowman, Z. Hatfield-Dodds, B. Mann, D. Amodei, N. Joseph, S. McCandlish, T. Brown, and J. Kaplan, 'Constitutional ai: Harmlessness from ai feedback,' 2022.
- [72] Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, N. Joseph, S. Kadavath, J. Kernion, T. Conerly, S. El-Showk, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, T. Hume, S. Johnston, S. Kravec, L. Lovitt, N. Nanda, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, B. Mann, and J. Kaplan, 'Training a helpful and harmless assistant with reinforcement learning from human feedback,' 2022.
- [73] H. Lee, S. Phatale, H. Mansoor, T. Mesnard, J. Ferret, K. Lu, C. Bishop, E. Hall, V. Carbune, A. Rastogi, and S. Prakash, 'Rlaif: Scaling reinforcement learning from human feedback with ai feedback,' 2023.
- [74] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, 'Training language models to follow instructions with human feedback,' 2022.
- [75] K. Lee, H. Liu, M. Ryu, O. Watkins, Y. Du, C. Boutilier, P. Abbeel, M. Ghavamzadeh, and S. S. Gu, 'Aligning text-to-image models using human feedback,' 2023.
- [76] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine, 'Training diffusion models with reinforcement learning,' 2024.
- [77] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee, 'Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models,' 2023.
- [78] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik, 'Diffusion model alignment using direct preference optimization,' 2023.
- [79] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, 'Direct preference optimization: Your language model is secretly a reward model,' 2023.
- [80] G. Han, M. Liang, J. Tang, Y. Cheng, W. Liu, and S. Huang, 'Reindiffuse: Crafting physically plausible motions with reinforced diffusion model,' in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) . IEEE, 2025, pp. 2218-2227.
- [81] H. Wang, W. Zhu, L. Miao, Y. Xu, F. Gao, Q. Tian, and Y. Wang, 'Aligning motion generation with human perceptions,' in International Conference on Learning Representations (ICLR) , 2025.
- [82] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, 'Smpl: A skinned multi-person linear model,' Seminal Graphics Papers: Pushing the Boundaries, Volume 2 , 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:5328073
- [83] R. J. Williams, 'Simple statistical gradient-following algorithms for connectionist reinforcement learning,' Machine learning , vol. 8, pp. 229-256, 1992.
- [84] J. Lin, R. Wang, J. Lu, Z. Huang, G. Song, A. Zeng, X. Liu, C. Wei, W. Yin, Q. Sun et al. , 'The quest for generalizable motion generation: Data, model, and evaluation,' arXiv preprint arXiv:2510.26794 , 2025.
- [85] P. Dhariwal and A. Nichol, 'Diffusion models beat gans on image synthesis,' 2021. [Online]. Available: https://arxiv.org/abs/2105.05233
- [86] J. Song, C. Meng, and S. Ermon, 'Denoising diffusion implicit models,' arXiv preprint arXiv:2010.02502 , 2020.
- [87] J. Ho and T. Salimans, 'Classifier-free diffusion guidance,' arXiv preprint arXiv:2207.12598 , 2022.
- [88] D. P. Kingma and J. Ba, 'Adam: A method for stochastic optimization,' 2017. [Online]. Available: https://arxiv.org/abs/1412.6980
- [89] Y. Zhao, Y. Wang, L. Wen, H. Zhang, and X. Qi, 'Freedance: Towards harmonic free-number group dance generation via a unified framework,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2025, pp. 10 560-10 569.
---

## Extracted Images

| # | File | Dimensions | Size |
|---|------|------------|------|
| 1 | figure_0032.png | 2065x752 | 418.6KB |
| 2 | figure_0068.png | 2066x1644 | 2197.0KB |
| 3 | figure_0076.png | 2068x551 | 629.3KB |
| 4 | figure_0112.png | 994x545 | 104.1KB |
| 5 | figure_0116.png | 1010x407 | 132.5KB |
