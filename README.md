# awesome-foundation-models-for-robotics
Curated database of foundation models for robotics

## Rules: 
- I just try to add my notes here. I can make mistake. Please don't be offended if your work is not here, just open an issue or PR.
- NOW AI HELP ME ADD PAPERS. MISTAKES HAPPEN. PLEASE DOUBLE CHECK ALL INFO.
- Included models: fundamental works, open weight/source works, works I saw on X, YouTube, LinkedIn, works I trained, works I tried to train but couldn't
- Actions means chunked, single, end effector, joint actions. Unfortunately, I cannot keep track of all of them for each work. Also most of the models can be adapted to different modalities. 

## Main list 👇

### **π0.6 (pi0.6)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [physicalintelligence.company/blog/pistar06](https://www.physicalintelligence.company/blog/pistar06)
* **Notes**:
    *   Introduces Reinforcement Learning (RL) to the VLA training pipeline.
    *   Allows the model to learn from experience, significantly improving success rates and throughput on real-world tasks.

---

### **Gemini Robotics 1.5 & ER 1.5**
*I, V, L → A, R (Image, Video, Language → Actions, Reasoning)*

* **Website**: [deepmind.google/models/gemini-robotics/](https://deepmind.google/models/gemini-robotics/)
* **Paper**: [Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots...](https://arxiv.org/abs/2510.03342)
* **Notes**:
    *   A dual-model system: VLA for low-level control and Embodied Reasoning (ER) for high-level planning.
    *   Interleaves actions with a natural language "thinking" process to decompose complex tasks.
    *   Demonstrates motion transfer, allowing policies to adapt across different robot embodiments (e.g., Aloha to Apollo).

---

### **DeepFleet**
*P, G → A (Proprioception, Goal → Actions)*

* **Website**: [amazon.science/blog/amazon-builds-first-foundation-model-for-multirobot-coordination](https://www.amazon.science/blog/amazon-builds-first-foundation-model-for-multirobot-coordination)
* **Paper**: [DeepFleet: Multi-Agent Foundation Models for Mobile Robots](https://arxiv.org/abs/2508.08574)
* **Notes**:
    *   A suite of foundation models for coordinating large-scale mobile robot fleets.
    *   Trained on fleet movement data from hundreds of thousands of robots in Amazon warehouses.
    *   Explores four architectures, with Robot-Centric (RC) and Graph-Floor (GF) showing the most promise for scaling.
    *   Enables proactive planning to avoid congestion and deadlocks in complex multi-agent environments.

---

### **Genie 3**
*Text/Image → Interactive World Video*

* **Website**: [deepmind.google/blog/genie-3-a-new-frontier-for-world-models/](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
* **Notes**:
    *   A general-purpose world model capable of generating interactive environments at 24fps.
    *   Used to train embodied agents (like SIMA) in rich, simulated worlds.
    *   Maintains environmental consistency over long horizons (minutes) and allows promptable world events.

---

### **Digit's Motor Cortex**
*O, P → A (Objectives, Proprioception → Actions)*

* **Website**: [agilityrobotics.com/content/training-a-whole-body-control-foundation-model](https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model)
* **Notes**:
    *   A whole-body control foundation model trained purely in simulation (Isaac Sim).
    *   Uses a small LSTM (<1M params) to handle balance, locomotion, and disturbance recovery.
    *   Functions as a "motor cortex," taking end-effector objectives and handling the low-level dynamics.

---

### **π0.5 (pi0.5)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [physicalintelligence.company/blog/pi05](https://www.physicalintelligence.company/blog/pi05)
* **Notes**:
    *   An evolution of π0 focused on open-world generalization.
    *   Capable of controlling mobile manipulators to perform tasks in entirely unseen environments like kitchens and bedrooms.

---

### **Large Behavior Model (LBM)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [toyotaresearchinstitute.github.io/lbm1/](https://toyotaresearchinstitute.github.io/lbm1/)
* **Paper**: [A careful examination of large behavior models...](https://arxiv.org/abs/2507.05331)
* **Code**: [lucidrains/TRI-LBM](https://github.com/lucidrains/TRI-LBM)
* **Notes**:
    * Uses a Diffusion Transformer (DiT) with Image and Text Encoders.
    * Demonstrated for complex bimanual manipulation tasks.
    * Has been implemented on a Boston Dynamics humanoid robot.

---

### **SAM2Act & SAM2Act+**
*I, P → A (Image, Proprioception → Actions)*

* **Website**: [sam2act.github.io](https://sam2act.github.io)
* **Paper**: [SAM2Act: Integrating Visual Foundation Model with A Memory Architecture...](https://arxiv.org/abs/2501.18564)
* **Code**: [sam2act/sam2act](https://github.com/sam2act/sam2act)
* **Notes**:
    *   Integrates the SAM2 visual foundation model with a memory architecture for robotic manipulation.
    *   SAM2Act+ incorporates a memory bank and encoder for episodic recall, enabling spatial memory-dependent tasks.
    *   Achieves state-of-the-art performance on RLBench (86.8%) and robust generalization on The Colosseum.

---

### **Waymo Motion FM**
*S, Map → Trajectory*

* **Website**: [waymo.com/research/scaling-laws-of-motion-forecasting-and-planning](https://waymo.com/research/scaling-laws-of-motion-forecasting-and-planning/)
* **Paper**: [Scaling Laws of Motion Forecasting and Planning](https://arxiv.org/abs/2506.08228)
* **Notes**:
    *   Demonstrates that motion forecasting and planning models follow scaling laws similar to LLMs.
    *   Trained on a massive dataset of 500,000 hours of driving data.
    *   Uses an encoder-decoder autoregressive transformer architecture.
    *   Shows that increasing compute and data predictably improves both open-loop and closed-loop performance.

---

### **3DLLM-Mem**
*I, L, M → A (Image, Language, Memory → Actions)*

* **Website**: [3dllm-mem.github.io](https://3dllm-mem.github.io/)
* **Paper**: [3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D LLM](https://arxiv.org/abs/2505.22657)
* **Notes**:
    *   Introduces a dynamic memory management system for Embodied 3D Large Language Models.
    *   Uses working memory tokens to selectively attend to episodic memory, enabling long-term spatial-temporal reasoning.
    *   Outperforms strong baselines by 16.5% on challenging in-the-wild embodied tasks (3DMem-Bench).
    *   NeurIPS 2025.

---

### **Agentic Robot**
*I, L → A (Image, Language → Actions)*

* **Website**: [agentic-robot.github.io](https://agentic-robot.github.io/)
* **Paper**: [Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models...](https://arxiv.org/abs/2505.23450)
* **Notes**:
    *   A brain-inspired framework that uses a Large Reasoning Model (LRM) to decompose tasks into subgoals (Standardized Action Procedure).
    *   Features a VLA executor for low-level control and a temporal verifier for error recovery.
    *   Achieves state-of-the-art performance on the LIBERO benchmark.

---

### **OpenVLA**
*I, L → A (Image, Language → Actions)*

* **Website**: [openvla.github.io](https://openvla.github.io/)
* **Paper**: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
* **Code**: [Official Repo](https://github.com/openvla/openvla)
* **Weights**: [Hugging Face](https://huggingface.co/openvla/openvla-7b)
* **Notes**:
    * Considered a fundamental work in open-source Vision-Language-Action models.
    * Built with a Llama transformer backbone.
    * Uses SigLIP + DINO for its vision component.

---

### **π0 (pi0)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [physicalintelligence.company/blog/pi0](https://www.physicalintelligence.company/blog/pi0)
* **Paper**: [π0: A vision-language-action flow model for general robot control](https://arxiv.org/abs/2410.24164)
* **Code**: [openpi on GitHub](https://github.com/Physical-Intelligence/openpi)
* **Weights**: [lerobot on Hugging Face](https://huggingface.co/lerobot/pi0)
* **Notes**:
    * Showcased in incredible bimanual and mobile robot demonstrations.
    * Architecture consists of a pretrained Vision-Language Model (VLM) combined with an action expert.
    * The pretrained VLM used is Paligemma.

---

### **π0-FAST (pi0-FAST)**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [physicalintelligence.company/research/fast](https://www.physicalintelligence.company/research/fast)
* **Paper**: [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)
* **Code**: [openpi on GitHub](https://github.com/Physical-Intelligence/openpi)
* **Notes**:
    *   An autoregressive variant of π0 using the FAST action tokenizer.
    *   Training is up to 5x faster than diffusion-based counterparts.
    *   Released open-source alongside π0.

---

### **SmolVLA**
*I, V, L → A (Image, Video, Language → Actions)*

* **Website**: [smolvla.net](https://smolvla.net/index_en.html)
* **Blog**: [huggingface.co/blog/smolvla](https://huggingface.co/blog/smolvla)
* **Notes**:
    *   A compact (~450M parameter) Vision-Language-Action model designed for efficiency.
    *   Optimized for running on consumer-grade GPUs and edge devices.
    *   Trained on the LeRobot community datasets.

---

### **Nvidia Isaac GR00T N1.5**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Website**: [developer.nvidia.com/isaac/gr00t](https://developer.nvidia.com/isaac/gr00t)
* **Paper**: [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)
* **Code**: [Official NVIDIA Repo](https://github.com/NVIDIA/Isaac-GR00T)
* **Notes**:
    * Combines a Vision-Language Model (VLM) with a Diffusion Transformer (DiT).
    * Features a very nice codebase that is compatible with `lerobot` with minor edits.
    * Includes utilities for inference servers and clients, making fine-tuning straightforward.

---

### **MolmoAct**
*Image → Depth Tokens, Image-Space Plan, Actions*

* **Website**: [allenai.org/blog/molmoact](https://allenai.org/blog/molmoact)
* **Paper**: [MolmoAct: Action Reasoning Models that can Reason in Space](https://arxiv.org/abs/2508.07917)
* **Weights**: [Hugging Face](https://huggingface.co/allenai/MolmoAct-7B-D-0812)
* **Notes**:
    * A very interesting and large model with a unique reasoning process.
    * It first estimates depth tokens, then plans a trajectory in the image space (independent of the robot's body), and finally generates the actions.
    * Because the image trace can be modified by a user, the resulting actions are steerable.

---

### **GR-3**
*I, P, L → A (Image, Proprioception, Language → Actions)*

* **Paper**: [GR-3: Foundation Models for Generalist Robots](https://arxiv.org/abs/2507.15493)
* **Code**: None available.
* **Notes**:
    * Trained on three diverse data types: internet-scale vision-language data, human hand tracking data, and robot trajectories.
    * The architecture is a VLM + DiT, similar to other leading models.
    * Employs compliance control during teleoperation, which is beneficial for contact-rich tasks.
    * Showed that it can learn new tasks from only 10 human trajectory demonstrations.

---

### **V-JEPA 2 & V-JEPA 2-AC**
*Video → Embeddings (V-JEPA 2) / Actions (V-JEPA 2-AC)*

* **Website**: [ai.meta.com/vjepa/](https://ai.meta.com/vjepa/)
* **Paper**: [V-JEPA 2: Self-supervised video models enable understanding...](https://arxiv.org/abs/2506.09985)
* **Code**: [Official facebookresearch Repo](https://github.com/facebookresearch/vjepa2)
* **Notes**:
    * A spatially capable vision encoder trained entirely with self-supervision.
    * Capable of next-state prediction, functioning as a world model.
    * The V-JEPA 2-AC version is post-trained with an "action-conditioned probe" to generate robot actions.

---

### **Nvidia Cosmos**
*Video, Text, Control → Video, Text (Reasoning)*

* **Website**: [nvidia.com/en-us/ai/cosmos](https://www.nvidia.com/en-us/ai/cosmos/)
* **Paper**: [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575)
* **Code**: [Official Repo](https://github.com/nvidia-cosmos)
* **Notes**:
    * A comprehensive world foundation model platform for Physical AI.
    * Includes `cosmos-predict` (video generation), `cosmos-transfer` (control-to-video), and `cosmos-reason` (reasoning VLM).
    * Models are open-weight and designed for robotics and autonomous vehicle simulation.

---

### **LiReN**
*I, G → A (Image, Goal → Actions)*

* **Website**: [kylestach.github.io/lifelong-nav-rl](https://kylestach.github.io/lifelong-nav-rl/)
* **Paper**: [Lifelong Autonomous Improvement of Navigation Foundation Models in the Wild](https://proceedings.mlr.press/v270/stachowicz25a.html)
* **Code**: [Official Repo](https://github.com/kylestach/lifelong-nav-rl)
* **Weights**: [Hugging Face](https://huggingface.co/rail-berkeley/liren-base)
* **Notes**:
    * The first navigation foundation model capable of autonomous fine-tuning in the wild.
    * Combines offline RL pretraining with online RL for continuous improvement.
    * Robust to new environments and embodiments.

---

### **LAC-WM**
*I, A → I' (Image, Actions → Predicted Image)*

* **Paper**: [Latent Action Robot Foundation World Models for Cross-Embodiment Adaptation](https://openreview.net/forum?id=vEZgPr1deb)
* **Notes**:
    * Learns a unified latent action space to handle diverse robot embodiments.
    * Achieves significant performance improvements (up to 46.7%) over models with explicit motion labels.
    * Enables efficient cross-embodiment learning and generalization.

***

### **RoboCat (2023)**
*I, P, G → A (Image, Proprioception, Goal Image → Actions)*

- **Website**: [Google DeepMind Blog Post](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/)
- **Paper**: [RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation](https://arxiv.org/abs/2306.11706)r
- **Notes**:
  - A multi-task, multi-embodiment generalist agent based on a decision transformer architecture (Gato).
  - Demonstrates a self-improvement loop: a trained model is fine-tuned for a new task, generates more data for that task, and this new data is used to train the next, more capable version of the generalist agent.
  - Can adapt to new tasks, objects, and even entirely new robot embodiments (e.g., KUKA arm) with only 100-1000 demonstration examples.
  - Tasks are specified using a visual goal image, not language.

## 🤖 Noteworthy Papers

### **Adapt3R: Adaptive 3D Scene Representation for Domain Transfer**
* **Paper**: [Wilcox, Albert, et al.](https://arxiv.org/abs/2503.04877)
* **Code**: [Official pairlab Repo](https://github.com/pairlab/Adapt3R)
* **Notes**:
    * Focuses on RGB-D based, viewpoint-invariant learning for imitation.
    * Provides a well-presented analysis of the limitations of current methods.

***

## 📚 Influential Posts & Videos

### **Vision-Language-Action Models and the Search for a Generalist Robot Policy**
* **Link**: [Substack Post by Chris Paxton](https://substack.com/@cpaxton/p-166350114)
* **Notes**:
    * A general overview of VLAs in the real world, with an excellent section on common failures.
    * Full of great insights and references.

### **Where's RobotGPT?**
* **Link**: [YouTube Video by Dieter Fox](https://www.youtube.com/watch?v=OAZrBYCLnaA)
* **Notes**:
    * This talk exists in many video forms; it's best to find the most recent version.
    * Focuses on the current state of robotics models and what is needed to achieve LLM-level general intelligence in robots.
