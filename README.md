# awesome-foundation-models-for-robotics
Curated database of foundation models for robotics

## Rules: 
- I just try to add my notes here. I can make mistake. Please don't be offended if your work is not here, just open an issue or PR.
- Included models: fundamental works, open weight/source works, works I saw on X, YouTube, LinkedIn, works I trained, works I tried to train but couldn't
- Actions means chunked, single, end effector, joint actions. Unfortunately, I cannot keep track of all of them for each work. Also most of the models can be adapted to different modalities. 

## Main list
| name | website | notes | implementations | input | output | ref |
|---|---|---|---|---|---|---|
| Large Behavior Model | [Link](https://toyotaresearchinstitute.github.io/lbm1/) | - DiT with Image and Text Encoder<br>- Demonstrated for bimanual manipulation tasks <br>- It is also implemented on Boston Dynamics humanoid<br>  | [lucidrains/TRI-LBM](https://github.com/lucidrains/TRI-LBM) | image, proprioception, language | actions | Barreiros, Jose, et al. "A careful examination of large behavior models for multitask dexterous manipulation." arXiv preprint arXiv:2507.05331 (2025). |
| OpenVLA |[Link](https://openvla.github.io/)|- One of the fundamental works <br>- llama is the main transformer<br>- siglip + dino for vision| - [Official](https://github.com/openvla/openvla) <br>- [Weights](https://huggingface.co/openvla/openvla-7b) <br>- there is no shortage of OpenVLA implementations |image, language|actions|Kim, Moo Jin, et al. "Openvla: An open-source vision-language-action model." arXiv preprint arXiv:2406.09246 (2024).|
|pi0|[Link](https://www.physicalintelligence.company/blog/pi0)|-Incredible bimanual, mobile robot demos<br>- Pretrained VLM + Action expert<br>- Pretrained VLM is Paligemma|- [openpi](https://github.com/Physical-Intelligence/openpi) <br>- [lerobot](https://huggingface.co/lerobot/pi0) |image, proprioception, language|actions|Black, Kevin, et al. "π0: A vision-language-action flow model for general robot control. CoRR, abs/2410.24164, 2024. doi: 10.48550." arXiv preprint ARXIV.2410.24164.|
|Nvidia Isaac Gr00t N1.5|[Link](https://developer.nvidia.com/isaac/gr00t)|- VLM + DiT <br>- Very nice codebase<br>- Compatible with lerobot with small editions<br>- Utilities for inference server, clients<br>- Finetuning is super easy|[Official](https://github.com/NVIDIA/Isaac-GR00T)|image, proprioception, language|actions|Bjorck, Johan, et al. "Gr00t n1: An open foundation model for generalist humanoid robots." arXiv preprint arXiv:2503.14734 (2025).|


## Also not directly Foundation models, but epic recent Robotics papers:
| ref | notes | implementations | 
|---|---|---|
|Wilcox, Albert, et al. "Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning." arXiv preprint arXiv:2503.04877 (2025).|- RGBD based viewpoint invariant learning.<br>- Nicely presented the limitations.| [Official](https://github.com/pairlab/Adapt3R)
    
