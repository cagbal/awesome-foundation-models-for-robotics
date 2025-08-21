# awesome-foundation-models-for-robotics
Curated database of foundation models for robotics

**Rules:** 
- I just try to add my notes here. I can make mistake. Please don't be offended if your work is not here, just open an issue or PR.
- Included models: fundamental works, open weight/source works, works I saw on X, YouTube, LinkedIn.
- Actions means chunked, single, end effector, joint actions. Unfortunately, I cannot keep track of all of them for each work. Also most of the models can be adapted to different modalities. 

| name | website | notes | implementations | input | output | ref |
|---|---|---|---|---|---|---|
| Large Behavior Model | [Link](https://toyotaresearchinstitute.github.io/lbm1/) | - DiT with Image and Text Encoder<br>- Demonstrated for bimanual manipulation tasks <br>- It is also implemented on Boston Dynamics humanoid<br>  | https://github.com/lucidrains/TRI-LBM | image, proprioception, language | actions | Barreiros, Jose, et al. "A careful examination of large behavior models for multitask dexterous manipulation." arXiv preprint arXiv:2507.05331 (2025).
| OpenVLA |[Link](https://openvla.github.io/)|- One of the fundamental works <br>- llama is the main transformer<br>- siglip + dino for vision| - [Official](https://github.com/openvla/openvla) <br>- [Weights](https://huggingface.co/openvla/openvla-7b) <br>- there is no shortage of OpenVLA implementations |image, language|actions|Kim, Moo Jin, et al. "Openvla: An open-source vision-language-action model." arXiv preprint arXiv:2406.09246 (2024).|
