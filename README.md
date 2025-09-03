# awesome-foundation-models-for-robotics
Curated database of foundation models for robotics

## Rules: 
- I just try to add my notes here. I can make mistake. Please don't be offended if your work is not here, just open an issue or PR.
- Included models: fundamental works, open weight/source works, works I saw on X, YouTube, LinkedIn, works I trained, works I tried to train but couldn't
- Actions means chunked, single, end effector, joint actions. Unfortunately, I cannot keep track of all of them for each work. Also most of the models can be adapted to different modalities. 

## Main list

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

***

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
