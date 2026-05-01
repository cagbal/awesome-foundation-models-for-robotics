# awesome-foundation-models-for-robotics
Curated database of foundation models for robotics

## Rules & Legend
- I just try to add my notes here. I can make mistakes. Please don't be offended if your work is not here, just open an issue or PR.
- NOW AI HELP ME ADD PAPERS. MISTAKES HAPPEN. PLEASE DOUBLE CHECK ALL INFO.
- Included models: fundamental works, open weight/source works, works I saw on X, YouTube, LinkedIn, works I trained, works I tried to train but couldn't.
- Actions means chunked, single, end effector, joint actions. Unfortunately, I cannot keep track of all of them for each work. Also most of the models can be adapted to different modalities. 

**Modality Legend:**
- `I`: Image | `Vid`: Video | `L`: Language/Text | `A`: Actions
- `P`: Proprioception | `T`: Tactile | `D`: Depth | `G`: Goal | `S`: State/Sensors | `M`: Memory | `F`: Force
- `A'`: Future Actions | `I'`: Future Images | `I_plan`: Image-Space Plan | `Vp`: Viewpoint
- `Val`: Value / Expected Reward | `Prog`: Progress Tracking

---

## NotebookLM - if you want to listen to this repo
**Notebook**: [Link](https://notebooklm.google.com/notebook/9692265a-7b38-4112-8aa5-6463f7a69a2d))

## Main list 👇

### 🚀 2026 Models

#### **LDA-1B**
*I, L → A (Image, Language → Actions)*

* **Website**: [pku-epic.github.io/LDA](https://pku-epic.github.io/LDA/)
* **Paper**: [LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion](https://arxiv.org/abs/2602.12215)
* **Code**: [PKU-EPIC/LDA](https://github.com/PKU-EPIC/LDA)
* **Notes**:
    *   Released Feb 2026.
    *   Jointly learns dynamics, policy, and visual forecasting.
    *   Assembled EI-30k, an embodied interaction dataset comprising over 30k hours of trajectories.
    *   Uses a structured DINO latent space for scalable dynamics learning.
    *   Employs a multi-modal diffusion transformer to handle asynchronous vision and action streams.
    *   Outperforms prior methods (like $\pi_{0.5}$) on contact-rich, dexterous, and long-horizon tasks.


#### **$\pi_{0.7}$ (Pi 0.7)**
*I, L → A (Image, Language → Actions)*

* **Website**: [pi.website/pi07](https://www.pi.website/pi07)
* **Paper**: [$\pi_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://www.pi.website/download/pi07.pdf)
* **Notes**:
    *   Released Apr 2026.
    *   A steerable generalist robot foundation model.
    *   Trained with diverse prompts that contain task description, detailed language, generated subgoal images, and episode metadata.
    *   Exhibits compositional generalization, recombining skills from various tasks to solve new problems.

#### **ManiDreams**
*S → A (State → Actions)*

* **Website**: [rice-robotpi-lab.github.io/ManiDreams](https://rice-robotpi-lab.github.io/ManiDreams/)
* **Paper**: [ManiDreams: An Open-Source Library for Robust Object Manipulation via Uncertainty-aware Task-specific Intuitive Physics](https://arxiv.org/abs/2603.18336)
* **Code**: [Rice-RobotPI-Lab/ManiDreams](https://github.com/Rice-RobotPI-Lab/ManiDreams)
* **Notes**:
    *   Released Mar 2026.
    *   A modular framework for uncertainty-aware manipulation planning over intuitive physics models.
    *   Maintains a time-varying constraint (cage) around target objects, sampling and evaluating candidate actions via parallel forward simulation.
    *   Supports simulation-based and learning-based backends.

#### **ForceVLA2**
*I, L → A (Image, Language → Actions)*

* **Website**: [sites.google.com/view/force-vla2/home](https://sites.google.com/view/force-vla2/home)
* **Paper**: [ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](https://arxiv.org/abs/2603.15169)
* **Notes**:
    *   Released Mar 2026.
    *   End-to-end vision-language-action framework for hybrid force-position control.
    *   Introduces force-based prompts to the VLM and uses a Cross-Scale Mixture-of-Experts (MoE) in the action expert to fuse task concepts with real-time interaction forces.
    *   Outperforms pi0 and pi0.5 in contact-rich manipulation tasks.

#### **LiLo-VLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [yy-gx.github.io/LiLo-VLA](https://yy-gx.github.io/LiLo-VLA/)
* **Paper**: [LiLo-VLA: Compositional Long-Horizon Manipulation via Linked Object-Centric Policies](https://arxiv.org/abs/2602.21531)
* **Notes**:
    *   Released Feb 2026.
    *   Linked Local VLA framework for compositional long-horizon manipulation.
    *   Decouples transport (global motion) from interaction (object-centric VLA).
    *   Zero-shot generalization to novel long-horizon tasks.
    *   Introduces **LIBERO-Long++** and **Ultra-Long** benchmarks.

#### **JALA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](https://arxiv.org/abs/2602.21736)
* **Notes**:
    *   Released Feb 2026.
    *   Pretraining framework that learns **Jointly-Aligned Latent Actions (JALA)**.
    *   Learns a predictive action embedding aligned with both inverse dynamics and real actions.
    *   Scales with **UniHand-Mix**, a 7.5M video corpus (>2,000 hours).
    *   Significantly improves downstream robot manipulation performance.

#### **Self-Correcting VLA (SC-VLA)**
*I, L → A (Image, Language → Actions)*

* **Paper**: [Self-Correcting VLA: Online Action Refinement via Sparse World Imagination](https://arxiv.org/abs/2602.21633)
* **Code**: [Kisaragi0/SC-VLA](https://github.com/Kisaragi0/SC-VLA)
* **Notes**:
    *   Released Feb 2026.
    *   Achieves self-improvement by intrinsically guiding action refinement through sparse imagination.
    *   Integrates auxiliary predictive heads to forecast current task progress and future trajectory trends.
    *   Introduces online action refinement to reshape progress-dependent dense rewards.
    *   Yields highest task throughput with 16% fewer steps and 9% higher success rate than baselines.

#### **HALO**
*I, L → A (Image, Language → Actions)*

* **Paper**: [HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning](https://arxiv.org/abs/2602.21157)
* **Notes**:
    *   Released Feb 2026.
    *   Unified VLA model for Embodied Multimodal Chain-of-Thought (EM-CoT) reasoning.
    *   Mixture-of-Transformers (MoT) architecture decoupling semantic reasoning, visual foresight, and action prediction.
    *   Surpasses baseline policy pi_0 by 34.1% on RoboTwin benchmark.
    *   Demonstrates strong generalization under aggressive unseen environmental randomization.

#### **AutoHorizon**
*I, L → A (Image, Language → Actions)*

* **Website**: [hatchetproject.github.io/autohorizon](https://hatchetproject.github.io/autohorizon/)
* **Paper**: [VLA Knows Its Limits](https://arxiv.org/abs/2602.21445)
* **Notes**:
    *   Released Feb 2026.
    *   Test-time method that dynamically estimates the execution horizon for each predicted action chunk.
    *   Analyzes self-attention weights in flow-based VLAs.
    *   Finds that intra-chunk actions attend invariantly to vision-language tokens.
    *   Incurs negligible computational overhead and generalizes across diverse tasks and flow-based models.

#### **TOPReward**
*Vid, L → Val (Video, Language → Value)*

* **Website**: [topreward.github.io/webpage](https://topreward.github.io/webpage/)
* **Paper**: [TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics](https://arxiv.org/abs/2602.19313)
* **Code**: [TOPReward/TOPReward](https://github.com/TOPReward/TOPReward)
* **Notes**:
    *   Released Feb 2026.
    *   A zero-shot progress estimator that interprets pretrained video VLM token likelihoods as temporal value functions.
    *   Avoids relying on numerical output, leveraging token probabilities (e.g., the token "True") for instruction satisfaction.
    *   Enables success detection and reward-aligned behavior cloning.

#### **VLANeXt**
*I, L → A (Image, Language → Actions)*

* **Website**: [dravenalg.github.io/VLANeXt](https://dravenalg.github.io/VLANeXt/)
* **Paper**: [VLANeXt: Recipes for Building Strong VLA Models](https://arxiv.org/abs/2602.18532)
* **Code**: [DravenALG/VLANeXt](https://github.com/DravenALG/VLANeXt)
* **Notes**:
    *   Released Feb 2026.
    *   Systematically explores the VLA design space under a unified framework to distill 12 key findings.
    *   Introduces VLANeXt, a simple yet effective model that outperforms prior state-of-the-art on LIBERO and LIBERO-plus benchmarks.
    *   Demonstrates strong generalization in real-world experiments.

#### **RoboGene**
*I, L → A (Image, Language → Actions)*

* **Website**: [robogene-boost-vla.github.io](https://robogene-boost-vla.github.io/)
* **Paper**: [RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation](https://arxiv.org/abs/2602.16444)
* **Notes**:
    *   Released Feb 2026.
    *   Agentic framework for automated generation of diverse manipulation tasks.
    *   Integrates diversity-driven sampling, self-reflection, and human-in-the-loop refinement.
    *   Collected 18k trajectories.
    *   VLA models pre-trained with RoboGene achieve higher success rates and superior generalization.

#### **DreamZero**
*I, L → A, Vid (Image, Language → Actions, Video)*

* **Website**: [dreamzero0.github.io](https://dreamzero0.github.io/)
* **Paper**: [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
* **Code**: [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero)
* **Weights**: [Hugging Face](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)
* **Notes**:
    *   Released Feb 2026.
    *   World Action Model (WAM) that jointly predicts actions and videos.
    *   Achieves strong zero-shot generalization to new tasks and environments (over 2x improvement vs VLAs).
    *   Demonstrates efficient cross-embodiment transfer (adapts to new robot with 30 mins of play data).
    *   Enables real-time closed-loop control at 7Hz via model and system optimizations (DreamZero-Flash).

#### **FUTURE-VLA**
*I, L → A, I' (Image, Language → Actions, Future Images)*

* **Paper**: [FUTURE-VLA: Forecasting Unified Trajectories Under Real-time Execution](https://arxiv.org/abs/2602.15882)
* **Notes**:
    *   Released Feb 2026.
    *   Unified architecture reformulating long-horizon control and future forecasting as a monolithic sequence-generation task.
    *   Leverages **Temporally Adaptive Compression** for high spatiotemporal information density.
    *   Performs **Latent-Space Autoregression** to align actionable dynamics with reviewable visual look-aheads.
    *   Enables prediction-guided Human-In-the-Loop mechanisms.
    *   Achieves 99.2% success on LIBERO.

#### **DM0**
*I, L → A (Image, Language → Actions)*

* **Paper**: [DM0: An Embodied-Native Vision-Language-Action Model towards Physical AI](https://arxiv.org/abs/2602.14974)
* **Code**: [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic)
* **Notes**:
    *   Released Feb 2026.
    *   Embodied-Native VLA framework designed for Physical AI.
    *   Unifies embodied manipulation and navigation by learning from heterogeneous data sources.
    *   Builds a flow-matching action expert atop the VLM.
    *   Uses Embodied Spatial Scaffolding for spatial CoT reasoning.
    *   Achieves SOTA performance on RoboChallenge benchmark.

#### **RynnBrain**
*I, L → A (Image, Language → Actions)*

* **Website**: [alibaba-damo-academy.github.io/RynnBrain.github.io](https://alibaba-damo-academy.github.io/RynnBrain.github.io)
* **Paper**: [RynnBrain: Open Embodied Foundation Models](https://arxiv.org/abs/2602.14979)
* **Notes**:
    *   Released Feb 2026.
    *   Open-source spatiotemporal foundation model for embodied intelligence.
    *   Strengthens egocentric understanding, localization, reasoning, and physics-aware planning.
    *   Family includes 2B, 8B, and 30B (MoE) models.
    *   Outperforms existing embodied foundation models on 20 benchmarks.

#### **APEX**
*I, P → A (Image, Proprioception → Actions)*

* **Website**: [apex-humanoid.github.io](https://apex-humanoid.github.io/)
* **Paper**: [APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots](https://arxiv.org/abs/2602.11143)
* **Notes**:
    *   Released Feb 2026.
    *   System for perceptive, climbing-based high-platform traversal for humanoids.
    *   Composes terrain-conditioned behaviors (climb-up, climb-down, walk, crawl).
    *   Uses a generalized ratchet progress reward for learning contact-rich maneuvers.
    *   Demonstrates zero-shot sim-to-real traversal of 0.8 meter platforms on Unitree G1.

#### **RISE**
*I, L → A (Image, Language → Actions)*

* **Website**: [opendrivelab.com/kai0-rl](https://opendrivelab.com/kai0-rl/)
* **Paper**: [RISE: Self-Improving Robot Policy with Compositional World Model](https://arxiv.org/abs/2602.11075)
* **Notes**:
    *   Released Feb 2026.
    *   Scalable framework for robotic reinforcement learning via imagination.
    *   **Compositional World Model**: predicts multi-view future via controllable dynamics model and evaluates outcomes.
    *   Enables continuous self-improvement in imaginary space without costly physical interaction.
    *   Achieves +35-45% improvement on real-world manipulation tasks.

#### **ContactGaussian-WM**
*Vid → I', Physics (Video → Future Images, Physics)*

* **Paper**: [ContactGaussian-WM: Learning Physics-Grounded World Model from Videos](https://arxiv.org/abs/2602.11021)
* **Notes**:
    *   Released Feb 2026.
    *   Differentiable physics-grounded rigid-body world model.
    *   Uses a unified **Gaussian representation** for visual appearance and collision geometry.
    *   Learns physical laws directly from sparse and contact-rich video data.
    *   Outperforms SOTA in learning complex scenarios and robust generalization.

#### **VISTA**
*I, L → A (Image, Language → Actions)*

* **Website**: [vista-wm.github.io](https://vista-wm.github.io/)
* **Paper**: [Scaling World Model for Hierarchical Manipulation Policies](https://arxiv.org/abs/2602.10983)
* **Notes**:
    *   Released Feb 2026.
    *   Hierarchical VLA framework using a world model for visual subgoal decomposition.
    *   High-level world model divides tasks into subtask sequences with **synthesized goal images**.
    *   Synthesized images provide visually and physically grounded details for the low-level policy.
    *   Boosts performance in novel scenarios from 14% to 69% with world model guidance.

#### **Say, Dream, and Act**
*I, L → A, Vid (Image, Language → Actions, Video)*

* **Paper**: [Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation](https://arxiv.org/abs/2602.10717)
* **Notes**:
    *   Released Feb 2026.
    *   Framework for fast and predictive video-conditioned action.
    *   Uses adversarial distillation for fast, few-step video generation ("Dreaming").
    *   Action model leverages both generated videos and real observations to correct spatial errors.
    *   Produces spatially accurate video predictions supporting precise manipulation.

#### **LAP**
*I, L → A (Image, Language → Actions)*

* **Website**: [lap-vla.github.io](https://lap-vla.github.io/)
* **Paper**: [LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer](https://arxiv.org/abs/2602.10556)
* **Notes**:
    *   Released Feb 2026.
    *   **Language-Action Pre-training**: represents low-level robot actions directly in natural language.
    *   Aligns action supervision with the pre-trained VLM's input-output distribution.
    *   **LAP-3B** achieves >50% average zero-shot success on novel robots without fine-tuning.
    *   Unifies action prediction and VQA in a shared language-action format.

#### **LocoVLM**
*I, L → A (Image, Language → Actions)*

* **Website**: [locovlm.github.io](https://locovlm.github.io/)
* **Paper**: [LocoVLM: Grounding Vision and Language for Adapting Versatile Legged Locomotion Policies](https://arxiv.org/abs/2602.10399)
* **Notes**:
    *   Released Feb 2026.
    *   Integrates high-level commonsense reasoning from foundation models into legged locomotion.
    *   Uses a VLM to extract environmental semantics and ground them in a skill database.
    *   Trains a style-conditioned policy for diverse locomotion skills.
    *   Achieves 87% instruction-following accuracy.

#### **ST4VLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [internrobotics.github.io/internvla-m1.github.io](https://internrobotics.github.io/internvla-m1.github.io/)
* **Paper**: [ST4VLA: Spatially Guided Training for Vision-Language-Action Models](https://arxiv.org/abs/2602.10109)
* **Notes**:
    *   Released Feb 2026.
    *   Dual-system VLA framework leveraging **Spatial Guided Training**.
    *   Stage 1: Spatial grounding pre-training (point, box, trajectory prediction).
    *   Stage 2: Spatially guided action post-training with spatial prompting.
    *   Substantial improvements on Google Robot and WidowX Robot tasks.

#### **DreamDojo**
*Vid, A → Vid' (Video, Actions → Future Video) [World Model]*

* **Website**: [dreamdojo-world.github.io](https://dreamdojo-world.github.io/)
* **Paper**: [DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos](https://arxiv.org/abs/2602.06949)
* **Code**: [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo)
* **Weights**: [nvidia/DreamDojo](https://huggingface.co/nvidia/DreamDojo)
* **Notes**:
    *   Released Feb 2026 by NVIDIA.
    *   Foundation world model learning diverse interactions and dexterous controls from 44k hours of egocentric human videos (`DreamDojo-HV` dataset).
    *   Introduces **continuous latent actions** as a hardware-agnostic proxy to extract control signals from unlabelled human video.
    *   Distillation pipeline accelerates autoregressive generation to real-time 10.81 FPS, enabling live teleoperation, policy evaluation, and model-based planning.

#### **EgoActor**
*I, L → A (Image, Language → Actions)*

* **Paper**: [EgoActor: Grounding Task Planning into Spatial-aware Egocentric Actions for Humanoid Robots via Visual-Language Models](https://arxiv.org/abs/2602.04515)
* **Notes**:
    *   Released Feb 2026.
    *   Unified and scalable VLM grounding high-level instructions into precise, spatially aware humanoid actions.
    *   Predicts **locomotion primitives** (walk, turn), head movements, and manipulation commands.
    *   Leverages broad supervision from real-world demos, spatial reasoning QA, and simulated demos.
    *   Inference under 1s with 4B and 8B parameter models.

#### **GeneralVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [GeneralVLA: Generalizable Vision-Language-Action Models with Knowledge-Guided Trajectory Planning](https://arxiv.org/abs/2602.04315)
* **Website**: [aigeeksgroup.github.io/GeneralVLA](https://aigeeksgroup.github.io/GeneralVLA)
* **Code**: [AIGeeksGroup/GeneralVLA](https://github.com/AIGeeksGroup/GeneralVLA)
* **Notes**:
    *   Released Feb 2026.
    *   Hierarchical VLA model enabling **zero-shot manipulation** without real-world robotic data collection.
    *   High-level **ASM** (Affordance Segmentation Module) perceives image keypoint affordances.
    *   Mid-level **3DAgent** carries out task understanding and trajectory planning.
    *   Low-level **3D-aware control policy** executes precise manipulation.

#### **SCALE**
*I, L → A (Image, Language → Actions)*

* **Paper**: [SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models](https://arxiv.org/abs/2602.04208)
* **Notes**:
    *   Released Feb 2026.
    *   Inference strategy that jointly modulates visual perception and action based on **'self-uncertainty'**.
    *   Inspired by **Active Inference** theory.
    *   Requires no additional training, no verifier, and only a single forward pass.
    *   Broadens exploration in perception and action under high uncertainty.

#### **DADP**
*I → A (Image → Actions)*

* **Paper**: [DADP: Domain Adaptive Diffusion Policy](https://arxiv.org/abs/2602.04037)
* **Website**: [outsider86.github.io/DomainAdaptiveDiffusionPolicy](https://outsider86.github.io/DomainAdaptiveDiffusionPolicy/)
* **Notes**:
    *   Released Feb 2026.
    *   Achieves robust adaptation through unsupervised disentanglement and domain-aware diffusion injection.
    *   Introduces **Lagged Context Dynamical Prediction** to filter out transient properties.
    *   Integrates learned domain representations directly into the generative process.

#### **SD-VLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement](https://arxiv.org/abs/2602.03983)
* **Notes**:
    *   Released Feb 2026.
    *   Disentangles visual inputs into multi-level **static and dynamic tokens**.
    *   Retains a single copy of static tokens (e.g., background) to significantly reduce context length.
    *   Reuses KV cache of static tokens via a lightweight **recache gate**.
    *   Delivers 2.26x inference speedup and improves long-horizon task performance.

#### **VLS**
*I, L → A (Image, Language → Actions)*

* **Paper**: [VLS: Steering Pretrained Robot Policies via Vision-Language Models](https://arxiv.org/abs/2602.03973)
* **Website**: [vision-language-steering.github.io/webpage](https://vision-language-steering.github.io/webpage/)
* **Notes**:
    *   Released Feb 2026.
    *   **Training-free framework** for inference-time adaptation of frozen generative robot policies (diffusion or flow-matching).
    *   Steers sampling process using VLMs to synthesize trajectory-differentiable reward functions.
    *   Addresses failures near obstacles, on shifted surfaces, or with mild clutter.

#### **DynamicVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [infinitescript.com/project/dynamic-vla](https://www.infinitescript.com/project/dynamic-vla/)
* **Paper**: [DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation](https://arxiv.org/abs/2601.22153)
* **Code**: [hzxie/DynamicVLA](https://github.com/hzxie/DynamicVLA)
* **Notes**:
    *   Released Jan 2026.
    *   Enables open-ended dynamic object manipulation by pairing a compact 0.4B VLM with low-latency Continuous Inference.
    *   Uses **Latent-Aware Action Streaming** to remove pauses and ensure seamless action transitions.
    *   Introduces the **Dynamic Object Manipulation (DOM)** benchmark with 2.8K scenes and 206 objects.
    *   Outperforms Pi0.5, SmolVLA, and VLASH in dynamic tasks.

#### **DeFM**
*D → S (Depth → Representations)*

* **Paper**: [DeFM: Learning Foundation Representations from Depth for Robotics](https://arxiv.org/abs/2601.18923)
* **Code**: [leggedrobotics/defm](https://github.com/leggedrobotics/defm)
* **Notes**:
    *   Released Jan 2026.
    *   Self-supervised foundation model trained on 60M depth images.
    *   Uses DINO-style self-distillation to learn metric-aware representations.
    *   Introduces a three-channel input normalization strategy to preserve metric depth.
    *   Distilled into compact models (as small as 3M params) for efficient policy learning.
    *   Achieves SOTA on depth-based navigation, locomotion, and manipulation benchmarks.

#### **SAM2Act & SAM2Act+**
*I, P → A (Image, Proprioception → Actions)*

* **Website**: [sam2act.github.io](https://sam2act.github.io)
* **Paper**: [SAM2Act: Integrating Visual Foundation Model with A Memory Architecture for Robotic Manipulation](https://arxiv.org/abs/2601.18564)
* **Code**: [sam2act/sam2act](https://github.com/sam2act/sam2act)
* **Notes**:
    *   Released Jan 2026.
    *   Integrates the SAM2 visual foundation model with a memory architecture for robotic manipulation.
    *   SAM2Act+ incorporates a memory bank and encoder for episodic recall, enabling spatial memory-dependent tasks.
    *   Achieves state-of-the-art performance on RLBench (86.8%) and robust generalization on The Colosseum.

#### **LingBot-VLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [technology.robbyant.com/lingbot-vla](https://technology.robbyant.com/lingbot-vla)
* **Paper**: [A Pragmatic VLA Foundation Model](https://arxiv.org/abs/2601.18692)
* **Code**: [robbyant/lingbot-vla](https://github.com/robbyant/lingbot-vla)
* **Notes**:
    *   Released Jan 2026.
    *   Pre-trained on 20,000 hours of real-world multi-embodiment robot data (9 dual-arm configurations).
    *   Achieves clear superiority on 100 real-world tasks across 3 platforms.
    *   Empirically validates **Scaling Laws** for VLAs: performance scales with data volume without saturation.
    *   Highly efficient training throughput.

#### **Cosmos Policy**
*I, P, L → A, I', Val (Image, Proprioception, Language → Actions, Future Images, Value)*

* **Website**: [research.nvidia.com/labs/dir/cosmos-policy](https://research.nvidia.com/labs/dir/cosmos-policy/)
* **Paper**: [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163)
* **Notes**:
    *   Released Jan 2026.
    *   Adapts `Cosmos-Predict2` (video generation model) into a robot policy via single-stage post-training.
    *   No architectural modifications to the base video model; actions are encoded as latent frames.
    *   Generates **future state images** and **values** (expected rewards) alongside actions, enabling **test-time planning**.
    *   Achieves state-of-the-art performance on LIBERO (98.5%) and RoboCasa (67.1%).
    *   Can learn from experience (policy rollout data) to refine its world model.

#### **EgoWM**
*I, A → I' (Image, Actions → Future Images)*

* **Website**: [egowm.github.io](https://egowm.github.io)
* **Paper**: [Walk through Paintings: Egocentric World Models from Internet Priors](https://arxiv.org/abs/2601.15284)
* **Code**: [miccooper9/egowm](https://github.com/miccooper9/egowm)
* **Notes**:
    *   Released Jan 2026.
    *   Transforms pretrained video diffusion models into action-conditioned world models.
    *   Injects motor commands through lightweight conditioning layers.
    *   Scales across embodiments, from 3-DoF mobile robots to 25-DoF humanoids.
    *   Introduces the Structural Consistency Score (SCS) to measure physical correctness.
    *   Generalizes to unseen environments, including paintings ("Walk through Paintings").

#### **BayesianVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries](https://arxiv.org/abs/2601.15197)
* **Notes**:
    *   Released Jan 2026.
    *   Addresses "Information Collapse" in goal-driven datasets where language is ignored.
    *   This collapse occurs because language instructions in existing datasets are often highly predictable from visual observations alone, causing the model to ignore language.
    *   Proposes a Bayesian decomposition framework with learnable Latent Action Queries.
    *   Maximizes conditional Pointwise Mutual Information (PMI) between actions and instructions.

#### **TIDAL**
*I, L → A (Image, Language → Actions)*

* **Paper**: [TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control](https://arxiv.org/abs/2601.14945)
* **Notes**:
    *   Released Jan 2026.
    *   Addresses high inference latency in large VLA models which causes execution blind spots.
    *   Proposes a hierarchical framework: low-frequency macro-intent loop caches semantic embeddings, high-frequency micro-control loop interleaves single-step flow integration.
    *   Enables ~9 Hz control on edge hardware (vs ~2.4 Hz baselines).
    *   Uses a temporally misaligned training strategy to learn predictive compensation.

#### **HumanoidVLM**
*I, L → A (Image, Language → Actions)*

* **Paper**: [HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation](https://arxiv.org/abs/2601.14874)
* **Notes**:
    *   Released Jan 2026.
    *   Enables humanoids (Unitree G1) to select task-appropriate impedance parameters from egocentric vision.
    *   Combines a VLM for semantic inference with a FAISS-based RAG module which retrieves experimentally validated stiffness-damping pairs for compliant manipulation.

#### **TwinBrainVLA**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Paper**: [TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers](https://arxiv.org/abs/2601.14133)
* **Notes**:
    *   Released Jan 2026.
    *   Resolves the tension between general semantic understanding and fine-grained motor skills.
    *   Features an **Asymmetric Mixture-of-Transformers (AsyMoT)** where the "Right Brain" (proprioception) can dynamically query the frozen "Left Brain" (VLM) for semantic knowledge, rather than just using standard fine-tuning.
    *   Uses a Flow-Matching Action Expert for precise control.

#### **DroneVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [DroneVLA: VLA based Aerial Manipulation](https://arxiv.org/abs/2601.13809)
* **Notes**:
    *   Released Jan 2026.
    *   Applies VLA models to autonomous aerial manipulation with a custom drone.
    *   Integrates Grounding DINO as a separate module for object localization and dynamic planning within the pipeline.
    *   Uses a human-centric controller for safe handovers.

#### **UniAct**
*I, L → A (Image, Language → Actions)*

* **Website**: [2toinf.github.io/UniAct](https://2toinf.github.io/UniAct/)
* **Paper**: [Universal Actions for Enhanced Embodied Foundation Models](https://arxiv.org/abs/2601.10105)
* **Code**: [2toinf/UniAct](https://github.com/2toinf/UniAct)
* **Notes**:
    *   Released Jan 2026.
    *   Operates in a Universal Action Space constructed as a vector-quantized (VQ) codebook.
    *   Learns universal actions capturing generic atomic behaviors shared across robots.
    *   Uses streamlined heterogeneous decoders to translate universal actions into embodiment-specific commands.
    *   0.5B model outperforms significantly larger models (14x larger).

#### **ActiveVLA**
*I, L, D → A, Vp (Image, Language, Depth → Actions, Viewpoint)*

* **Paper**: [ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation](https://arxiv.org/abs/2601.08325)
* **Notes**:
    *   Released Jan 2026.
    *   Injects active perception into VLA models to address limitations of static, end-effector-centric views.
    *   Adopts a **coarse-to-fine paradigm**: first localizes critical 3D regions, then optimizes active perception.
    *   Uses **Active View Selection** to choose viewpoints that maximize task relevance/diversity and minimize occlusion.
    *   Applies **Active 3D Zoom-in** to enhance resolution in key areas for fine-grained manipulation.
    *   Outperforms baselines on simulation benchmarks and transfers to real-world tasks.

#### **PointWorld**
*I, D, A → S (Image, Depth, Actions → 3D Point Flow)*

* **Website**: [point-world.github.io](https://point-world.github.io/)
* **Paper**: [PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation](https://arxiv.org/abs/2601.03782)
* **Code**: [huangwl18/PointWorld](https://github.com/huangwl18/PointWorld)
* **Notes**:
    *   Released Jan 2026.
    *   Large pre-trained 3D world model forecasting future states from single RGB-D images.
    *   Represent actions and state changes as **3D point flows** (per-pixel displacements in 3D space), enabling geometry-aware predictions.
    *   Unifies state and action in a shared 3D space, facilitating cross-embodiment learning.
    *   Trained on ~2M trajectories and 500 hours of real and simulated data.
    *   Enables diverse zero-shot manipulation skills (pushing, tool use) via MPC.

#### **VLM4VLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [cladernyjorn.github.io/VLM4VLA.github.io](https://cladernyjorn.github.io/VLM4VLA.github.io/)
* **Paper**: [VLM4VLA: Revisiting Vision-Language-Models in Vision-Language-Action Models](https://arxiv.org/abs/2601.03309)
* **Code**: [CladernyJorn/VLM4VLA](https://github.com/CladernyJorn/VLM4VLA)
* **Notes**:
    *   Released Jan 2026.
    *   Unified training and evaluation framework (VLM4VLA) for studying VLM backbones in VLAs.
    *   Reveals that **VLM general capabilities (VQA)** are poor predictors of downstream VLA performance.
    *   Identifies the **vision encoder** as the primary bottleneck; fine-tuning it is crucial (freezing it leads to degradation).
    *   Finds that fine-tuning on **auxiliary embodied tasks** (e.g., embodied QA, visual pointing) does not guarantee better control performance.

#### **1X World Model (1XWM)**
*I, L → Vid, A (Image, Language → Video, Actions)*

* **Website**: [1x.tech/ai](https://www.1x.tech/ai)
* **Notes**:
    *   Released Jan 2026.
    *   Video-pretrained world model serving as NEO's cognitive core.
    *   Derives robot actions from text-conditioned video generation (14B parameter backbone).
    *   Uses a two-stage process: generates future video frames (World Model), then extracts actions via an Inverse Dynamics Model (IDM).
    *   Trained on web-scale video, 900 hours of egocentric human video, and fine-tuned on 70 hours of robot data.
    *   Explicitly functions as a **World Model**, predicting/hallucinating outcomes before execution.
    *   Generalizes to novel objects and tasks without teleoperation data.

#### **Nvidia Isaac GR00T N1.6**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [developer.nvidia.com/isaac/gr00t](https://developer.nvidia.com/isaac/gr00t)
* **Research Page**: [research.nvidia.com/labs/gear/gr00t-n1_6/](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
* **Code**: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
* **Weights**: [Hugging Face](https://huggingface.co/nvidia/GR00T-N1.6-3B)
* **Notes**:
    *   Released Jan 2026.
    *   Reasoning VLA model for generalist humanoid robots.
    *   Integrates `NVIDIA Cosmos Reason` for high-level reasoning and contextual understanding.
    *   Unlocks full-body control for simultaneous moving and manipulation.

#### **Rho-alpha (ρα)**
*I, L, T → A (Image, Language, Tactile → Actions)*

* **Website**: [microsoft.com/en-us/research/story/advancing-ai-for-the-physical-world/](https://www.microsoft.com/en-us/research/story/advancing-ai-for-the-physical-world/)
* **Notes**:
    *   Released Jan 2026.
    *   The first robotics model derived from Microsoft's **Phi** series.
    *   **VLA+ Model**: Integrates tactile sensing directly into the decision-making process.
    *   Uses a split architecture: a VLM for high-level reasoning and a specialized **action expert** for high-frequency control.
    *   Trained using physical demonstrations and simulation (Isaac Sim).

#### **$\pi^*_0.6$ (Pi-Star 0.6)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [physicalintelligence.company/blog/pistar06](https://www.physicalintelligence.company/blog/pistar06)
* **Notes**:
    *   Released early 2026.
    *   Introduces Reinforcement Learning (RL) to the VLA training pipeline.
    *   Allows the model to learn from experience, significantly improving success rates and throughput on real-world tasks.
    *   **Personal Note**: *I tried to train this locally but couldn't get the RL pipeline to converge due to limited VRAM scaling on my setup.*

---

### 📆 2025 Models

#### **Dream-VLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone](https://arxiv.org/abs/2512.22615)
* **Notes**:
    *   Released Dec 2025.
    *   Diffusion LLM-based VLA (dVLA) developed through continuous pre-training on open robotic datasets.
    *   Natively bidirectional diffusion backbone is inherently suited for action chunking and parallel generation.
    *   Demonstrates superior performance on VLA tasks compared to autoregressive baselines.

#### **VLA-Motion**
*I, L → A (Image, Language → Actions)*

* **Website**: [vla-motion.github.io](https://vla-motion.github.io/)
* **Paper**: [Robotic VLA Benefits from Joint Learning with Motion Image Diffusion](https://arxiv.org/abs/2512.18007)
* **Notes**:
    *   Released Dec 2025.
    *   Enhances VLAs with motion reasoning by jointly training with a motion image diffusion head (optical flow).
    *   The motion head acts as an auxiliary task, improving the shared representation.
    *   Improves success rates on LIBERO (97.5%) and real-world tasks (23% gain).
    *   No additional inference latency.

#### **FASTerVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [FASTer: Toward Efficient Autoregressive Vision Language Action Modeling via Neural Action Tokenization](https://arxiv.org/abs/2512.04952)
* **Notes**:
    *   Released Dec 2025.
    *   Builds on the FAST tokenizer with block-wise autoregressive decoding and a lightweight action expert.
    *   Uses a learnable action tokenizer (FASTerVQ) that encodes action chunks as single-channel images.
    *   Achieves faster inference and higher task performance compared to diffusion VLAs.

#### **ManualVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [ManualVLA: A Unified VLA Model for Chain-of-Thought Manual Generation and Robotic Manipulation](https://arxiv.org/abs/2512.02013)
* **Notes**:
    *   Released Dec 2025.
    *   Unified VLA framework with Mixture-of-Transformers (MoT).
    *   Generates intermediate "manuals" (images, position prompts, textual instructions) via a planning expert.
    *   Uses a Manual Chain-of-Thought (ManualCoT) reasoning process.
    *   Achieves 32% higher success rate on long-horizon tasks like LEGO assembly.

#### **GR-RL**
*I, L → A (Image, Language → Actions)*

* **Website**: [seed.bytedance.com/gr_rl](https://seed.bytedance.com/gr_rl)
* **Paper**: [GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2512.01801)
* **Notes**:
    *   Released Dec 2025.
    *   Turns a generalist VLA policy into a specialist for long-horizon dexterous manipulation.
    *   Uses a multi-stage training pipeline (filtering, augmentation, online RL).
    *   The online RL component learns a latent space noise predictor to align the policy with deployment behaviors.
    *   Can autonomously lace up a shoe (83.3% success rate), requiring millimeter-level precision.

#### **SONIC**
*S, P → A (State, Proprioception → Actions)*

* **Website**: [nvlabs.github.io/SONIC](https://nvlabs.github.io/SONIC/)
* **Paper**: [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://arxiv.org/abs/2511.07820)
* **Code**: [huggingface/trl](https://github.com/huggingface/trl) (Related)
* **Notes**:
    *   Released Nov 2025.
    *   Addresses the challenge of diverse human motion data scarcity by extracting an expansive motion dataset (OmniHuman) containing diverse skills and realistic movements.
    *   Introduces SONIC, a large-scale neural tracking policy demonstrating natural humanoid motions with up to 10.7x lower tracking error.
    *   Validates zero-shot deployment in real-world scenarios for expressive humanoid movements.

#### **EveryDayVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [everydayvla.github.io](https://everydayvla.github.io/)
* **Paper**: [EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation](https://arxiv.org/abs/2511.05397)
* **Notes**:
    *   Released Nov 2025.
    *   Aims to democratize robotic manipulation with affordable hardware ($300 6-DOF arm).
    *   Unified model jointly outputting discrete and continuous actions.
    *   Features an adaptive-horizon ensemble to monitor motion uncertainty and trigger on-the-fly re-planning.
    *   Matches SOTA on LIBERO benchmark.

#### **XR-1**
*I, L → A (Image, Language → Actions)*

* **Paper**: [XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations](https://arxiv.org/abs/2511.02776)
* **Notes**:
    *   Released Nov 2025.
    *   Introduces Unified Vision-Motion Codes (UVMC), a discrete latent representation for visual dynamics and robotic motion.
    *   Uses a dual-branch VQ-VAE to jointly encode vision and motion.
    *   Demonstrates strong cross-task and cross-embodiment generalization in real-world experiments.

#### **Unified Diffusion VLA**
*I, L → A, I' (Image, Language → Actions, Future Images)*

* **Paper**: [Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process](https://arxiv.org/abs/2511.01718)
* **Notes**:
    *   Released Nov 2025.
    *   Jointly understands, generates future images, and acts using a synchronous denoising process.
    *   Integrates multiple modalities into a single denoising trajectory (JD3P).
    *   Achieves 4x faster inference than autoregressive methods on benchmarks like CALVIN and LIBERO.

#### **RL-100**
*I, P → A (Image, Proprioception → Actions)*

* **Website**: [lei-kun.github.io/RL-100](https://lei-kun.github.io/RL-100/)
* **Paper**: [RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning](https://arxiv.org/abs/2510.14830)
* **Code**: [Lei-Kun/Uni-o4](https://github.com/Lei-Kun/Uni-o4)
* **Notes**:
    *   Released Oct 2025.
    *   Tackles the efficiency issues of traditional deep RL in real-world environments.
    *   Introduces an algorithmic framework built on a cross-modal transformer backbone designed to learn directly on real robots.
    *   Achieves rapid acquisition of complex manipulation skills (within 1-2 hours) with high success rates across 100+ tasks in a single day.

#### **ManiAgent**
*I, L → A (Image, Language → Actions)*

* **Paper**: [ManiAgent: An Agentic Framework for General Robotic Manipulation](https://arxiv.org/abs/2510.11660)
* **Notes**:
    *   Released Oct 2025.
    *   Agentic architecture for general manipulation tasks.
    *   Uses multi-agent communication for perception, sub-task decomposition, and action generation.
    *   Achieves 95.8% success rate on real-world pick-and-place tasks.

#### **X-VLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model](https://arxiv.org/abs/2510.10274)
* **Notes**:
    *   Released Oct 2025.
    *   Uses "soft prompts" (learnable embeddings) to adapt to different robot embodiments and datasets.
    *   Treats each hardware setup as a distinct "task" guided by these prompts.
    *   Built on a flow-matching-based VLA architecture.

#### **IntentionVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [IntentionVLA: Generalizable and Efficient Embodied Intention Reasoning for Human-Robot Interaction](https://arxiv.org/abs/2510.07778)
* **Notes**:
    *   Released Oct 2025.
    *   Focuses on implicit human intention reasoning for complex interactions.
    *   Uses a curriculum training paradigm combining intention inference, spatial grounding, and embodied reasoning.
    *   Significantly outperforms baselines on out-of-distribution intention tasks.

#### **Gemini Robotics 1.5 & ER 1.5**
*I, Vid, L → A, Val (Image, Video, Language → Actions, Reasoning/Value)*

* **Website**: [deepmind.google/models/gemini-robotics/](https://deepmind.google/models/gemini-robotics/)
* **Paper**: [Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer](https://arxiv.org/abs/2510.03342)
* **Notes**:
    *   Released Oct 2025.
    *   A dual-model system: VLA for low-level control and Embodied Reasoning (ER) for high-level planning.
    *   Interleaves actions with a natural language "thinking" process to decompose complex tasks.
    *   Demonstrates motion transfer, allowing policies to adapt across different robot embodiments (e.g., Aloha to Apollo).

#### **CLAP**
*I, L → A (Image, Language → Actions)*

* **Paper**: [CLAP: A Closed-Loop Diffusion Transformer Action Foundation Model for Robotic Manipulation](https://ieeexplore.ieee.org/document/11246478)
* **Notes**:
    *   Presented at IROS 2025 (October).
    *   Componentized VLA architecture with a specialized **action module** and a **critic module**.
    *   Uses **diffusion action transformers** for modeling continuous temporal actions.
    *   The **critic module** enables closed-loop inference by refining actions based on feedback.
    *   Outperforms methods that use simple action quantization, handling complex, high-precision tasks and generalizing to unseen objects.

#### **MLA**
*I, P, T, L → A (Image, Proprioception, Tactile, Language → Actions)*

* **Website**: [sites.google.com/view/open-mla](https://sites.google.com/view/open-mla)
* **Paper**: [MLA: A Multisensory Language-Action Model for Multimodal Understanding and Forecasting in Robotic Manipulation](https://arxiv.org/abs/2509.26642)
* **Notes**:
    *   Released Sep 2025.
    *   Integrates 2D visual, 3D geometric, and tactile cues.
    *   Repurposes the LLM itself as a perception module (encoder-free alignment).
    *   Predicts future multisensory objectives to facilitate physical world modeling.

#### **SARM**
*Vid, L → Prog (Video, Language → Progress)*

* **Website**: [qianzhong-chen.github.io/sarm.github.io](https://qianzhong-chen.github.io/sarm.github.io/)
* **Paper**: [SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation](https://arxiv.org/abs/2509.25358)
* **Code**: [Qianzhong-Chen/SARM](https://github.com/Qianzhong-Chen/SARM)
* **Notes**:
    *   Released Sep 2025.
    *   **Stage-Aware Reward Modeling** framework for long-horizon robot manipulation.
    *   Jointly predicts the high-level task stage and fine-grained progress within each stage from video frames.
    *   Uses natural language subtask annotations to derive consistent progress labels.
    *   Enables **Reward-Aligned Behavior Cloning (RA-BC)**, weighting training samples based on predicted progress.

#### **MotoVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [motovla.github.io](https://motovla.github.io/)
* **Paper**: [Generalist Robot Manipulation beyond Action Labeled Data](https://arxiv.org/abs/2509.19958)
* **Notes**:
    *   Released Sep 2025.
    *   Leverages motion data (without explicit action labels) to train generalist policies.
    *   Introduces a Motion Tokenizer to learn discrete motion representations.
    *   Enables scaling up training data by utilizing large-scale video datasets.

#### **Behavior Foundation Model (BFM)**
*G, P → A (Goal/Objectives, Proprioception → Actions)*

* **Paper**: [Behavior Foundation Model for Humanoid Robots](https://arxiv.org/abs/2509.13780)
* **Notes**:
    *   Released Sep 2025.
    *   Generative model pretrained on large-scale behavioral datasets for humanoid robots.
    *   Models the distribution of full-body behavioral trajectories conditioned on goals and proprioception.
    *   Integrates masked online distillation with CVAE.
    *   Enables flexible operation across diverse control modes (velocity, motion tracking, teleop) and generalizes robustly.

#### **NavFoM**
*I, L → A (Image, Language → Actions)*

* **Website**: [pku-epic.github.io/NavFoM-Web](https://pku-epic.github.io/NavFoM-Web/)
* **Paper**: [Embodied Navigation Foundation Model](https://arxiv.org/abs/2509.12129)
* **Notes**:
    *   Released Sep 2025.
    *   Cross-embodiment and cross-task navigation foundation model.
    *   Trained on 8 million navigation samples (quadrupeds, drones, wheeled robots, vehicles).
    *   Unified architecture handling diverse camera setups and temporal horizons.

#### **FLOWER**
*I, L → A (Image, Language → Actions)*

* **Paper**: [FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies](https://arxiv.org/abs/2509.04996)
* **Notes**:
    *   Released Sep 2025.
    *   Proposes Vision-Language-Flow (VLF) models to make generalist policies more efficient.
    *   Achieves 3x faster inference speed compared to diffusion-based VLAs.
    *   Demonstrates strong performance on CALVIN and real-world tasks.

#### **ManiFlow**
*I, L → A (Image, Language → Actions)*

* **Website**: [maniflow-policy.github.io](https://maniflow-policy.github.io/)
* **Paper**: [ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training](https://arxiv.org/abs/2509.01819)
* **Notes**:
    *   Released Sep 2025.
    *   Uses consistency-based flow matching for efficient action generation.
    *   Trained on large-scale open-source datasets (Open-X).
    *   Outperforms OpenVLA and other baselines in simulation and real-world experiments.

#### **UnifoLM-WMA-0**
*I, A → I', A' (Image, Actions → Future Images, Future Actions)*

* **Website**: [unigen-x.github.io/unifolm-world-model-action.github.io](https://unigen-x.github.io/unifolm-world-model-action.github.io/)
* **Code**: [unitreerobotics/unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)
* **Notes**:
    *   Released Sep 2025.
    *   Unitree's open-source world-model-action architecture for general-purpose robot learning.
    *   Functions as both a Simulation Engine (generating synthetic data) and Policy Enhancement (predicting future interactions).
    *   Trained on Unitree's open-source datasets and fine-tuned on Open-X.

#### **Discrete Diffusion VLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](https://arxiv.org/abs/2508.20072)
* **Code**: [Liang-ZX/DiscreteDiffusionVLA](https://github.com/Liang-ZX/DiscreteDiffusionVLA)
* **Notes**:
    *   Released Aug 2025.
    *   Discretizes continuous action spaces and uses discrete diffusion for action decoding.
    *   Unified transformer framework compatible with standard VLM token interfaces.
    *   Achieves 96.3% success rate on LIBERO and outperforms continuous diffusion baselines.

#### **Long-VLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [long-vla.github.io](https://long-vla.github.io/)
* **Paper**: [Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation](https://arxiv.org/abs/2508.19958)
* **Notes**:
    *   Released Aug 2025.
    *   Addresses the limitation of current VLAs in long-horizon tasks.
    *   Incorporates a hierarchical planning mechanism within the VLA framework.
    *   Significantly improves success rates on multi-stage manipulation tasks.

#### **Embodied-R1**
*I, L → A (Image, Language → Actions)*

* **Website**: [embodied-r1.github.io](https://embodied-r1.github.io/)
* **Paper**: [Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation](https://arxiv.org/abs/2508.13998)
* **Code**: [pickxiguapi/Embodied-R1](https://github.com/pickxiguapi/Embodied-R1)
* **Notes**:
    *   Released Aug 2025.
    *   3B Vision-Language Model designed for embodied reasoning and "pointing".
    *   Uses "pointing" as a unified intermediate representation (similar concept to Molmo).
    *   Trained with Reinforced Fine-tuning (RFT) with multi-task reward design.
    *   Demonstrates robust zero-shot generalization (e.g., 56.2% success in SIMPLEREnv).

#### **DeepFleet**
*P, G → A (Proprioception, Goal → Actions)*

* **Website**: [amazon.science/blog/amazon-builds-first-foundation-model-for-multirobot-coordination](https://www.amazon.science/blog/amazon-builds-first-foundation-model-for-multirobot-coordination)
* **Paper**: [DeepFleet: Multi-Agent Foundation Models for Mobile Robots](https://arxiv.org/abs/2508.08574)
* **Notes**:
    *   Released Aug 2025.
    *   A suite of foundation models for coordinating large-scale mobile robot fleets.
    *   Trained on fleet movement data from hundreds of thousands of robots in Amazon warehouses.
    *   Explores four architectures, with Robot-Centric (RC) and Graph-Floor (GF) showing the most promise for scaling.
    *   Enables proactive planning to avoid congestion and deadlocks in complex multi-agent environments.

#### **MolmoAct**
*I, L → D, I_plan, A (Image, Language → Depth Tokens, Image-Space Plan, Actions)*

* **Website**: [allenai.org/blog/molmoact](https://allenai.org/blog/molmoact)
* **Paper**: [MolmoAct: Action Reasoning Models that can Reason in Space](https://arxiv.org/abs/2508.07917)
* **Weights**: [Hugging Face](https://huggingface.co/allenai/MolmoAct-7B-D-0812)
* **Notes**:
    *   Released Aug 2025.
    *   A very interesting and large model with a unique reasoning process.
    *   It first estimates depth tokens, then plans a trajectory in the image space (independent of the robot's body), and finally generates the actions.
    *   Because the image trace can be modified by a user, the resulting actions are steerable.

#### **RICL**
*I, L → A (Image, Language → Actions)*

* **Website**: [ricl-vla.github.io](https://ricl-vla.github.io/)
* **Paper**: [RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models](https://arxiv.org/abs/2508.02062)
* **Notes**:
    *   Released Aug 2025.
    *   Enables VLA models to adapt to new tasks via in-context learning (ICL).
    *   Uses a retrieval-based mechanism to fetch relevant demonstrations.
    *   Avoids the need for expensive fine-tuning for every new task.

#### **Digit's Motor Cortex**
*G, P → A (Goal/Objectives, Proprioception → Actions)*

* **Website**: [agilityrobotics.com/content/training-a-whole-body-control-foundation-model](https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model)
* **Notes**:
    *   Released Aug 2025.
    *   A whole-body control foundation model trained purely in simulation (Isaac Sim).
    *   Uses a small LSTM (<1M params) to handle balance, locomotion, and disturbance recovery.
    *   Functions as a "motor cortex," taking end-effector objectives and handling the low-level dynamics.

#### **InstructVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation](https://arxiv.org/abs/2507.17520)
* **Notes**:
    *   Released Jul 2025.
    *   Two-stage pipeline: pretrains an action expert/latent interface, then instruction-tunes a VLM.
    *   Uses an MoE-adapted VLM to switch between textual reasoning and latent action generation.
    *   Focuses on preserving multimodal reasoning while adding precise manipulation capabilities.

#### **GR-3**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Paper**: [GR-3 Technical Report](https://arxiv.org/abs/2507.15493)
* **Notes**:
    *   Released Jul 2025.
    *   Trained on three diverse data types: internet-scale vision-language data, human hand tracking data, and robot trajectories.
    *   The architecture is a VLM + DiT, similar to other leading models.
    *   Employs compliance control during teleoperation, which is beneficial for contact-rich tasks.
    *   Showed that it can learn new tasks from only 10 human trajectory demonstrations.

#### **Large Behavior Model (LBM)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [toyotaresearchinstitute.github.io/lbm1/](https://toyotaresearchinstitute.github.io/lbm1/)
* **Paper**: [A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation](https://arxiv.org/abs/2507.05331)
* **Notes**:
    *   Released Jul 2025.
    *   Uses a Diffusion Transformer (DiT) with Image and Text Encoders.
    *   Demonstrated for complex bimanual manipulation tasks.
    *   Has been implemented on a Boston Dynamics humanoid robot.

#### **Unified VLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [Unified Vision-Language-Action Model](https://arxiv.org/abs/2506.19850)
* **Notes**:
    *   Released Jun 2025.
    *   Autoregressively models vision, language, and actions as a single interleaved stream of discrete tokens.
    *   Incorporates world modeling during post-training to capture causal dynamics.
    *   Achieves strong results on CALVIN and LIBERO benchmarks.

#### **RoboMonkey**
*I, L → A (Image, Language → Actions)*

* **Website**: [robomonkey-vla.github.io](https://robomonkey-vla.github.io/)
* **Paper**: [RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models](https://arxiv.org/abs/2506.17811)
* **Notes**:
    *   Released Jun 2025.
    *   Focuses on test-time compute scaling for VLAs.
    *   Uses a learned verifier (value function) to sample and select the best actions during inference.
    *   Demonstrates that scaling test-time compute can rival training-time scaling.

#### **ControlVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [ControlVLA: Few-shot Object-centric Adaptation for Pre-trained Vision-Language-Action Models](https://arxiv.org/abs/2506.16211)
* **Notes**:
    *   Released Jun 2025.
    *   Adapts pre-trained VLAs to new objects and tasks using few-shot learning.
    *   Employs object-centric representations via a ControlNet-style adapter to preserve pre-trained knowledge.
    *   Achieves efficient adaptation with minimal data.

#### **V-JEPA 2 & V-JEPA 2-AC**
*Vid → S, A (Video → Embeddings, Actions)*

* **Website**: [ai.meta.com/vjepa/](https://ai.meta.com/vjepa/)
* **Paper**: [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
* **Code**: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
* **Notes**:
    *   Released Jun 2025.
    *   A spatially capable vision encoder trained entirely with self-supervision.
    *   Capable of next-state prediction, functioning as a world model.
    *   The V-JEPA 2-AC version is post-trained with an "action-conditioned probe" to generate robot actions.

#### **Waymo Motion FM**
*S, M → A (State, Map → Trajectory)*

* **Website**: [waymo.com/research/scaling-laws-of-motion-forecasting-and-planning](https://waymo.com/research/scaling-laws-of-motion-forecasting-and-planning/)
* **Paper**: [Scaling Laws of Motion Forecasting and Planning -- Technical Report](https://arxiv.org/abs/2506.08228)
* **Notes**:
    *   Released Jun 2025.
    *   Demonstrates that motion forecasting and planning models follow scaling laws similar to LLMs.
    *   Trained on a massive dataset of 500,000 hours of driving data.
    *   Uses an encoder-decoder autoregressive transformer architecture.
    *   Shows that increasing compute and data predictably improves both open-loop and closed-loop performance.

#### **Genie Envisioner**
*I, L → Vid, A (Image, Language → Video, Actions)*

* **Paper**: [Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation](https://arxiv.org/abs/2506.05635)
* **Notes**:
    *   Released Jun 2025 (Presented at RSS 2025 as Best Systems Paper finalist).
    *   Unified platform collapsing robot sensing, policy learning, and evaluation into a single closed-loop video generative world model.
    *   Trained on ~3,000 hours of video-language paired data (AgiBot-World-Beta).

#### **Fast-in-Slow (FiS)**
*I, L → A (Image, Language → Actions)*

* **Paper**: [Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning](https://arxiv.org/abs/2506.01953)
* **Notes**:
    *   Released Jun 2025.
    *   Dual-system VLA embedding a fast execution module (System 1) within a slow reasoning VLM (System 2).
    *   System 1 shares parameters with System 2 but operates at higher frequency.
    *   Uses a dual-aware co-training strategy to jointly fine-tune both systems.
    *   Addresses the trade-off between reasoning capability and execution speed.

#### **Feel the Force (FTF)**
*I, T → A (Image, Tactile → Actions)*

* **Website**: [feel-the-force-ftf.github.io](https://feel-the-force-ftf.github.io)
* **Paper**: [Feel the Force: Contact-Driven Learning from Humans](https://arxiv.org/abs/2506.01944)
* **Notes**:
    *   Released Jun 2025.
    *   A robot learning system that models human tactile behavior to learn force-sensitive manipulation.
    *   Uses a tactile glove to collect human demonstrations with precise contact forces.
    *   Achieves robust force-aware control by continuously predicting the forces needed for manipulation.

#### **Agentic Robot**
*I, L → A (Image, Language → Actions)*

* **Website**: [agentic-robot.github.io](https://agentic-robot.github.io/)
* **Paper**: [Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents](https://arxiv.org/abs/2505.23450)
* **Notes**:
    *   Released May 2025.
    *   A brain-inspired framework that uses a Large Reasoning Model (LRM) to decompose tasks into subgoals (Standardized Action Procedure).
    *   Features a VLA executor for low-level control and a temporal verifier for error recovery.
    *   Achieves state-of-the-art performance on the LIBERO benchmark.

#### **TrackVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [pku-epic.github.io/TrackVLA-web](https://pku-epic.github.io/TrackVLA-web/)
* **Paper**: [TrackVLA: Embodied Visual Tracking in the Wild](https://arxiv.org/abs/2505.23189)
* **Notes**:
    *   Released May 2025.
    *   Integrates visual tracking capabilities into a VLA architecture.
    *   Enables robots to track and interact with moving targets in dynamic environments.
    *   Trained on a diverse dataset of tracking scenarios.

#### **3DLLM-Mem**
*I, L, M → A (Image, Language, Memory → Actions)*

* **Website**: [3dllm-mem.github.io](https://3dllm-mem.github.io/)
* **Paper**: [3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model](https://arxiv.org/abs/2505.22657)
* **Notes**:
    *   Released May 2025.
    *   Introduces a dynamic memory management system for Embodied 3D Large Language Models.
    *   Uses working memory tokens to selectively attend to episodic memory, enabling long-term spatial-temporal reasoning.
    *   Outperforms strong baselines by 16.5% on challenging in-the-wild embodied tasks (3DMem-Bench).

#### **UniSkill**
*I, Vid → A (Image, Video → Actions)*

* **Website**: [kimhanjung.github.io/UniSkill](https://kimhanjung.github.io/UniSkill/)
* **Paper**: [UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations](https://arxiv.org/abs/2505.08787)
* **Notes**:
    *   Released May 2025.
    *   Learns skill representations from large-scale human videos.
    *   Uses Inverse Skill Dynamics (ISD) to extract motion patterns and Forward Skill Dynamics (FSD) for future prediction.
    *   Transfers these skills to robot embodiments using a cross-embodiment interface.
    *   Enables learning from observing humans without explicit teleoperation data.

#### **UniVLA**
*I, L → A (Image, Language → Actions)*

* **Paper**: [UniVLA: Learning to Act Anywhere with Task-centric Latent Actions](https://arxiv.org/abs/2505.06111)
* **Notes**:
    *   Released May 2025.
    *   Learns task-centric action representations from videos using a latent action model (within DINO feature space).
    *   Can leverage data from arbitrary embodiments and perspectives without explicit action labels.
    *   Allows deploying generalist policies to various robots via efficient latent action decoding.

#### **OpenHelix**
*I, L → A (Image, Language → Actions)*

* **Website**: [openhelix-robot.github.io](https://openhelix-robot.github.io/)
* **Paper**: [OpenHelix: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation](https://arxiv.org/abs/2505.03912)
* **Notes**:
    *   Released May 2025.
    *   Open-source Dual-System VLA (Vision-Language-Action) model.
    *   Provides systematic empirical evaluations on dual-system architectures (System 1 for fast execution, System 2 for reasoning).
    *   Highlights a "prompt tuning" paradigm: adding a new `<ACT>` token and only training the `lm-head` preserves generalization.

#### **RobotxR1**
*I, L → A (Image, Language → Actions)*

* **Paper**: [RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning](https://arxiv.org/abs/2505.03238)
* **Notes**:
    *   Released May 2025.
    *   Extends R1-Zero approach to robotics via closed-loop Reinforcement Learning.
    *   Enables small-scale LLMs (e.g., Qwen2.5-3B) to achieve effective reasoning and control.
    *   Demonstrated on autonomous driving tasks.

#### **GraspVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [pku-epic.github.io/GraspVLA-web](https://pku-epic.github.io/GraspVLA-web/)
* **Paper**: [GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data](https://arxiv.org/abs/2505.03233)
* **Notes**:
    *   Released May 2025.
    *   A specialist foundation model for grasping.
    *   Pre-trained on a massive synthetic dataset (billion-scale) of grasping actions.
    *   Demonstrates zero-shot transfer to real-world objects.

#### **Uncertainty-Aware RWM (RWM-U)**
*I, P, A → I', S (Image, Proprioception, Actions → Future Images, Uncertainty)*

* **Website**: [sites.google.com/view/uncertainty-aware-rwm](https://sites.google.com/view/uncertainty-aware-rwm)
* **Paper**: [Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots](https://arxiv.org/abs/2504.16680)
* **Code**: [leggedrobotics/robotic_world_model_lite](https://github.com/leggedrobotics/robotic_world_model_lite)
* **Notes**:
    *   Released Apr 2025.
    *   Extends Robotic World Model (RWM) with ensemble-based epistemic uncertainty estimation.
    *   Enables fully offline model-based reinforcement learning (MBRL) on real robots by penalizing high-risk imagined transitions (MOPO-PPO).
    *   Evaluated on real quadruped and humanoid robots for manipulation and locomotion.

#### **Nvidia Isaac GR00T N1.5**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [developer.nvidia.com/isaac/gr00t](https://developer.nvidia.com/isaac/gr00t)
* **Paper**: [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)
* **Code**: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
* **Notes**:
    *   Released Mar 2025.
    *   Combines a Vision-Language Model (VLM) with a Diffusion Transformer (DiT).
    *   **Personal Note**: *A very nice codebase that is highly compatible with `lerobot`. I found the client/server inference utilities quite handy to experiment with.*

#### **HybridVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [hybrid-vla.github.io](https://hybrid-vla.github.io/)
* **Paper**: [HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model](https://arxiv.org/abs/2503.10631)
* **Code**: [PKU-HMI-Lab/Hybrid-VLA](https://github.com/PKU-HMI-Lab/Hybrid-VLA)
* **Notes**:
    *   Released Mar 2025.
    *   Unified framework integrating autoregressive reasoning and diffusion-based action prediction.
    *   Uses a collaborative action ensemble mechanism to fuse predictions from both paradigms.
    *   Outperforms previous SOTA VLA methods by 14% in simulation and 19% in real-world tasks.

#### **Magma**
*I, Vid, L → A (Image, Video, Language → Actions)*

* **Website**: [microsoft.github.io/Magma](https://microsoft.github.io/Magma/)
* **Paper**: [Magma: A Foundation Model for Multimodal AI Agents](https://arxiv.org/abs/2502.13130)
* **Code**: [microsoft/Magma](https://github.com/microsoft/Magma)
* **Notes**:
    *   Released Feb 2025.
    *   Multimodal foundation model for agentic tasks in digital and physical worlds.
    *   Uses Set-of-Mark (SoM) for action grounding and Trace-of-Mark (ToM) for action planning.
    *   State-of-the-art on UI navigation and robotic manipulation.

#### **DexVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [dex-vla.github.io](https://dex-vla.github.io/)
* **Paper**: [DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control](https://arxiv.org/abs/2502.05855)
* **Notes**:
    *   Released Feb 2025.
    *   Combines a VLM for high-level reasoning with a diffusion expert for low-level control.
    *   The diffusion expert is "plug-in", allowing modular upgrades.
    *   Focused on dexterous manipulation tasks.

#### **Nvidia Cosmos**
*Vid, L, A → Vid, L (Video, Language, Control → Video, Language/Reasoning)*

* **Website**: [nvidia.com/en-us/ai/cosmos](https://www.nvidia.com/en-us/ai/cosmos/)
* **Paper**: [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575)
* **Code**: [nvidia-cosmos](https://github.com/nvidia-cosmos)
* **Notes**:
    *   Released Jan 2025.
    *   A comprehensive world foundation model platform for Physical AI.
    *   Includes `cosmos-predict` (video generation), `cosmos-transfer` (control-to-video), and `cosmos-reason` (reasoning VLM).
    *   Models are open-weight and designed for robotics and autonomous vehicle simulation.

#### **$\pi_{0.5}$ (Pi 0.5)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [physicalintelligence.company/blog/pi05](https://www.physicalintelligence.company/blog/pi05)
* **Notes**:
    *   Released Early 2025.
    *   An evolution of π0 focused on open-world generalization.
    *   Capable of controlling mobile manipulators to perform tasks in entirely unseen environments like kitchens and bedrooms.

#### **SmolVLA**
*I, Vid, L → A (Image, Video, Language → Actions)*

* **Website**: [smolvla.net](https://smolvla.net/index_en.html)
* **Blog**: [huggingface.co/blog/smolvla](https://huggingface.co/blog/smolvla)
* **Notes**:
    *   Released Early 2025.
    *   A compact (~450M parameter) Vision-Language-Action model designed for efficiency.
    *   Optimized for running on consumer-grade GPUs and edge devices.
    *   Trained on the LeRobot community datasets.
    *   **Personal Note**: *Tried to train this and successfully ran inference on a consumer GPU. Very fast and lightweight.*

#### **Genie 3**
*I, L → Vid (Image, Language → Interactive World Video)*

* **Website**: [deepmind.google/blog/genie-3-a-new-frontier-for-world-models/](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
* **Notes**:
    *   Released Early 2025.
    *   A general-purpose world model capable of generating interactive environments at 24fps.
    *   Used to train embodied agents (like SIMA) in rich, simulated worlds.
    *   Maintains environmental consistency over long horizons (minutes) and allows promptable world events.

#### **RDT-2**
*I, L → A (Image, Language → Actions)*

* **Website**: [rdt-robotics.github.io/rdt2/](https://rdt-robotics.github.io/rdt2/)
* **Code**: [thu-ml/RDT2](https://github.com/thu-ml/RDT2)
* **Weights**: [Hugging Face](https://huggingface.co/collections/robotics-diffusion-transformer/rdt-2)
* **Notes**:
    *   Released Early 2025.
    *   The sequel to RDT-1B, designed for zero-shot cross-embodiment generalization.
    *   **RDT2-VQ**: A 7B VLA adapted from Qwen2.5-VL-7B, using Residual VQ for action tokenization.
    *   **RDT2-FM**: Uses a Flow-Matching action expert for lower latency control.
    *   Trained on 10,000+ hours of human manipulation videos across 100+ scenes (UMI data).

#### **ELLMER**
*I, L, F → A (Image, Language, Force → Actions)*

* **Paper**: [Embodied large language models enable robots to complete complex tasks in unpredictable environments](https://www.nature.com/articles/s42256-025-01000-0)
* **Code**: [ruaridhmon/ELLMER](https://github.com/ruaridhmon/ELLMER)
* **Notes**:
    *   Released Early 2025.
    *   Embodied Large-Language-Model-Enabled Robot framework.
    *   Uses GPT-4 and Retrieval-Augmented Generation (RAG) to extract relevant code examples from a knowledge base.
    *   Generates action plans that incorporate real-time force and visual feedback to adapt to unpredictable environments.
    *   Enables robots to complete long-horizon tasks like coffee making.

#### **LiReN**
*I, G → A (Image, Goal → Actions)*

* **Website**: [kylestach.github.io/lifelong-nav-rl](https://kylestach.github.io/lifelong-nav-rl/)
* **Paper**: [Lifelong Autonomous Improvement of Navigation Foundation Models in the Wild](https://proceedings.mlr.press/v270/stachowicz25a.html)
* **Code**: [kylestach/lifelong-nav-rl](https://github.com/kylestach/lifelong-nav-rl)
* **Weights**: [Hugging Face](https://huggingface.co/rail-berkeley/liren-base)
* **Notes**:
    *   Released Early 2025.
    *   The first navigation foundation model capable of autonomous fine-tuning in the wild.
    *   Combines offline RL pretraining with online RL for continuous improvement.
    *   Robust to new environments and embodiments.

#### **LAC-WM**
*I, A → I' (Image, Actions → Predicted Image)*

* **Paper**: [Latent Action Robot Foundation World Models for Cross-Embodiment Adaptation](https://openreview.net/forum?id=vEZgPr1deb)
* **Notes**:
    *   Released Early 2025.
    *   Learns a unified latent action space to handle diverse robot embodiments.
    *   Achieves significant performance improvements (up to 46.7%) over models with explicit motion labels.
    *   Enables efficient cross-embodiment learning and generalization.

#### **Lift3D Policy**
*I, P → A (Image, Proprioception → Actions)*

* **Paper**: [Lift3D Policy: Lifting 2D Foundation Models for Robust 3D Robotic Manipulation](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Lift3D_Policy_Lifting_2D_Foundation_Models_for_Robust_3D_Robotic_CVPR_2025_paper.pdf)
* **Code**: [PKU-HMI-Lab/LIFT3D](https://github.com/PKU-HMI-Lab/LIFT3D)
* **Notes**:
    *   Released Early 2025.
    *   Lifts 2D foundation models to construct robust 3D manipulation policies.
    *   Uses a task-aware masked autoencoder to enhance implicit 3D representations.
    *   Establishes positional mapping between 3D points and 2D model embeddings.

#### **3DS-VLA**
*I, L, D → A (Image, Language, Depth → Actions)*

* **Paper**: [3DS-VLA: A 3D Spatial-Aware Vision Language Action Model for Robust Multi-Task Manipulation](https://proceedings.mlr.press/v305/li25g.html)
* **Notes**:
    *   Released Early 2025.
    *   Enhances 2D VLAs with explicit 3D spatial awareness.
    *   Uses a 2D-to-3D positional alignment mechanism to encode spatial observations.
    *   Outperforms state-of-the-art 2D and 3D policies on RLBench and real-world tasks.

#### **DYNA-1**
*I, L → A (Image, Language → Actions)*

* **Website**: [dyna.co](https://www.dyna.co/)
* **Notes**:
    *   Released Early 2025.
    *   Production-ready foundation model built for autonomy at scale.
    *   Achieved >99% success rate in 24-hour non-stop operation.
    *   Deployed in commercial settings like hotels and gyms.

---

### 🏛️ 2024 & Older

#### **$\pi_0$ (Pi 0)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [physicalintelligence.company/blog/pi0](https://www.physicalintelligence.company/blog/pi0)
* **Paper**: [π0: A vision-language-action flow model for general robot control](https://arxiv.org/abs/2410.24164)
* **Code**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
* **Weights**: [Hugging Face](https://huggingface.co/lerobot/pi0)
* **Notes**:
    *   Released Oct 2024.
    *   Showcased in incredible bimanual and mobile robot demonstrations.
    *   Architecture consists of a pretrained Vision-Language Model (VLM) combined with an action expert.
    *   The pretrained VLM used is Paligemma.
    *   **Personal Note**: *I attempted to train this locally on my hardware but couldn't get the flow-matching loss to converge properly. However, it showcased incredible dexterous zero-shot capabilities out-of-the-box.*

#### **OpenVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [openvla.github.io](https://openvla.github.io/)
* **Paper**: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
* **Code**: [openvla/openvla](https://github.com/openvla/openvla)
* **Weights**: [Hugging Face](https://huggingface.co/openvla/openvla-7b)
* **Notes**:
    *   Released Jun 2024.
    *   Considered a fundamental work in open-source Vision-Language-Action models.
    *   Built with a Llama transformer backbone. Uses SigLIP + DINO for its vision component.
    *   **Personal Note**: *I fine-tuned this 7B model locally. It was very VRAM-hungry but provided an excellent baseline!*

#### **RoboCat (2023)**
*I, P, G → A (Image, Proprioception, Goal Image → Actions)*

* **Website**: [deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/)
* **Paper**: [RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation](https://arxiv.org/abs/2306.11706)
* **Notes**:
    *   Released Jun 2023.
    *   A multi-task, multi-embodiment generalist agent based on a decision transformer architecture (Gato).
    *   Demonstrates a self-improvement loop: a trained model is fine-tuned for a new task, generates more data for that task, and this new data is used to train the next, more capable version of the generalist agent.
    *   Can adapt to new tasks, objects, and even entirely new robot embodiments (e.g., KUKA arm) with only 100-1000 demonstration examples.

---

## 🤖 Noteworthy Benchmarks / Auxiliary Frameworks

### **KinDER**
* **Website**: [kinder-site](https://prpl-group.com/kinder-site/)
* **Paper**: [KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning](https://arxiv.org/abs/2604.25788)
* **Code**: [Princeton-Robot-Planning-and-Learning/kindergarden](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden)
* **Notes**:
    *   Released Apr 2026.
    *   A physical reasoning benchmark for robot learning and planning.
    *   Comprises 25 procedurally generated environments testing spatial relations, nonprehensile multi-object manipulation, tool use, combinatorial geometric constraints, and dynamic constraints.
    *   Includes a Gymnasium-compatible Python library with parameterized skills and demonstrations.
    *   Provides a standardized evaluation suite with 13 baselines spanning TAMP, imitation learning, RL, and foundation-model-based approaches.

---

### **RBench & RoVid-X**
* **Paper**: [Rethinking Video Generation Model for the Embodied World](https://arxiv.org/abs/2601.15282)
* **Notes**:
    *   Released Jan 2026.
    *   Introduces **RBench**, a comprehensive robotics benchmark for video generation.
    *   Presents **RoVid-X**, a large-scale high-quality robotic dataset for training video generation models.
    *   Evaluation results on 25 video models show high agreement with human assessments.

---

### **CRISP**
* **Website**: [utiasDSL.github.io/crisp_controllers](https://utiasDSL.github.io/crisp_controllers)
* **Paper**: [CRISP -- Compliant ROS2 Controllers for Learning-Based Manipulation Policies and Teleoperation](https://arxiv.org/abs/2509.06819)
* **Notes**:
    *   Released Sep 2025.
    *   A lightweight C++ implementation of compliant Cartesian and joint-space controllers for the ROS2 control standard.
    *   Designed for seamless integration with high-level learning-based policies as well as teleoperation.

---

### **SafeDec: Constrained Decoding for Robotics Foundation Models**
* **Website**: [constrained-robot-fms.github.io](https://constrained-robot-fms.github.io)
* **Paper**: [Constrained Decoding for Robotics Foundation Models](https://arxiv.org/abs/2509.01728)
* **Notes**:
    *   Released Sep 2025.
    *   A constrained decoding framework for autoregressive robot foundation models.
    *   Enforces task-specific safety rules (Signal Temporal Logic) at inference time without retraining.
    *   Compatible with state-of-the-art policies like SPOC and PoliFormer.

---

### **Risk-Guided Diffusion**
* **Paper**: [Risk-Guided Diffusion: Toward Deploying Robot Foundation Models in Space, Where Failure Is Not An Option](https://arxiv.org/abs/2506.17601)
* **Notes**:
    *   Released Jun 2025.
    *   Proposes a risk-guided diffusion framework fusing a fast "System-1" with a slow, physics-based "System-2".
    *   Addresses safety for deploying foundation models in space exploration.
    *   Reduces failure rates by up to 4x while matching goal-reaching performance.

---

### **Adapt3R: Adaptive 3D Scene Representation for Domain Transfer**
* **Paper**: [Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning](https://arxiv.org/abs/2503.04877)
* **Code**: [pairlab/Adapt3R](https://github.com/pairlab/Adapt3R)
* **Notes**:
    *   Released Mar 2025.
    *   Focuses on RGB-D based, viewpoint-invariant learning for imitation across domain gaps.
    *   Provides a well-presented analysis of the limitations of current methods.

---

### **Towards Safe Robot Foundation Models**
* **Paper**: [Towards Safe Robot Foundation Models](https://arxiv.org/abs/2503.07404)
* **Notes**:
    *   Released Mar 2025.
    *   Introduces a safety layer to constrain the action space of any generalist policy.
    *   Uses **ATACOM**, a safe reinforcement learning algorithm, to create a safe action space and ensure safe state transitions.
    *   Facilitates deployment in safety-critical scenarios without requiring specific safety fine-tuning.
    *   Demonstrated effectiveness in avoiding collisions in dynamic environments (e.g., air hockey).

---

## 📚 Influential Posts & Videos

### **Vision-Language-Action Models and the Search for a Generalist Robot Policy**
* **Link**: [Substack Post by Chris Paxton](https://substack.com/@cpaxton/p-166350114)
* **Notes**:
    *   A general overview of VLAs in the real world, with an excellent section on common failures.
    *   Full of great insights and references.

### **Where's RobotGPT?**
* **Link**: [YouTube Video by Dieter Fox](https://www.youtube.com/watch?v=OAZrBYCLnaA)
* **Notes**:
    *   This talk exists in many video forms; it's best to find the most recent version.
    *   Focuses on the current state of robotics models and what is needed to achieve LLM-level general intelligence in robots.
