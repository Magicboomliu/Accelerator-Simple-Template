# Accelerator-Simple-Template

This an example of training the [Marigold Depth Estimation](https://huggingface.co/spaces/toshas/marigold) using accelerator using the sceneflow dataset. Since the original training code is not open source, only the inference pipeline is released, so the performance is not guaranteed. BTW, Any other dataset is fine, just change the dataloader.  

Reference Code: [Marigold-ETH](https://github.com/prs-eth/marigold)   

Reference Paper: [Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://arxiv.org/abs/2312.02145)


#### Run the Inference of Monodepth estimation: 

```
cd scripts
sh inference.sh
``` 

#### Run the Inference of Monodepth Training, Using SceneFlow as an example:
```
cd scripts
sh train.sh
``` 

Note the training at least takes 21 VRAM even the batch size is set to 1.