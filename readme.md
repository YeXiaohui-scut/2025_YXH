# LaWa: Using Latent Space for In-Generation Image Watermarking

This repo contains the implementation of the paper 'LaWa: Using Latent Space for In-Generation Image Watermarking', published at ECCV2024, **plus a major upgrade to Semantic Watermarking**.

Link to arXiv paper: https://arxiv.org/abs/2408.05868

Link to Huawei's AI Gallery Notebook: https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=03ccae2a-4fa8-4739-a75b-659a3abcc690

<p align="center">
<center>
<img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/lawa/framework.png" alt="alt text" width="1000">
</center>
</p>

## 🆕 Semantic Watermarking Upgrade

We've upgraded LaWa from binary watermarks to **semantic watermarks** based on LatentSeal, SWIFT, and MetaSeal approaches. Key improvements:

- **Semantic Vectors**: Embed high-dimensional semantic vectors (512 dims) instead of 48-bit binary codes
- **Cryptographic Security**: Rotation matrix encryption for enhanced security
- **Rich Metadata**: Embed model version, user ID, timestamps, and more
- **Prompt Integration**: Watermark is semantically bound to the text prompt
- **U-Net Based Embedding**: Content-adaptive perturbation generation

**📖 [Read the full documentation](SEMANTIC_WATERMARKING.md)**

### Quick Start with Semantic Watermarking

```bash
# Run demo
python examples/semantic_watermarking_demo.py

# Run inference with semantic watermarking
python inference_semantic.py \
    --config_sd stable-diffusion/configs/stable-diffusion/v1-inference.yaml \
    --ckpt_sd weights/stable-diffusion-v1/model.ckpt \
    --config_lawa configs/SD14_SemanticLaWa_inference.yaml \
    --prompt "A photo of an astronaut riding a horse on the moon" \
    --add_metadata \
    --model_version "v1.0" \
    --outdir results/semantic_watermarking
```

--- 

## Install Required Packages

We have tested our code with python 3.8.17, pytorch 2.0.1, torchvision 0.15.2, and cuda 11.3. You can reproduce the environment using conda by running


```python
!conda env create -f environment.yml
!conda activate LaWa
```

## Inference
Run the following script to download our pretrained modified decoder as well as the original decoder. These weights correspond to the KL-f8 auto-encoder model and 48-bit watermarks.



```python
!bash download.sh
```

Model weights will be saved to `weights/LaWa/last.ckpt` and `weights/first_stage_models/first_stage_KL-f8.ckpt`.  

Furthermore, download weights of Stable Diffusion v1.4 model from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) and save it to `weights/stable-diffusion-v1/model.ckpt`.

To generate watermarked images using Stable Diffusion and LaWa, run:


```python
!python inference_AIGC.py --config configs/SD14_LaWa_inference.yaml --prompt "A white plate of food on a dining table" --message_len 48 --message '110111001110110001000000011101000110011100110101' --outdir results/SD14_LaWa/txt2img-samples
```

This will save the generated original and watermarked images as well as the difference image in `results/SD14_LaWa/txt2img-samples`. Also, `results/SD14_LaWa/test_results_quality.csv` and `results/SD14_LaWa/test_results_attacks.csv` are generated, which contain a summary of the visual quality of the watermarked image as well as its robustness to attacks.

## Train your own model
### Data Preparation
Download the MIRFlickR dataset from the official website. `data/train_100k.csv` contains the list of images we have used for training. In the config file `configs/SD14_LaWa.yaml`, adjust the path to images folder of the dataset under the data_dir of train and validation datasets.
### Train
You can train your modified decoder using:


```python
!python train.py --message_len 48 --config configs/SD14_LaWa.yaml --batch_size 8 --max_epochs 40 --learning_rate 0.00006
```

Batch size 8 fits on a 32GB GPU.

## Versions Comparison

| Feature | Original LaWa | Semantic LaWa (New) |
|---------|--------------|---------------------|
| Watermark Type | 48-bit binary | 512-dim semantic vector |
| Information Capacity | 48 bits | ~2048 bits equivalent |
| Security | Moderate | High (cryptographic) |
| Semantic Binding | None | Strong (prompt-based) |
| Metadata Support | Limited | Rich (model, user, timestamp) |
| Embedding Method | Linear layers | U-Net perturbation generator |
| Verification | Bit accuracy | Cosine similarity |

### When to Use Each Version

**Original LaWa**: 
- Simple binary watermark needs
- Well-established training pipeline
- Lower computational requirements

**Semantic LaWa**:
- Need cryptographic security
- Want to embed rich metadata
- Require prompt-based verification
- Need stronger forgery resistance

## 📚 Citation
If you find LaWa useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.
```bibtex
@misc{rezaei2024lawausinglatentspace,
      title={LaWa: Using Latent Space for In-Generation Image Watermarking}, 
      author={Ahmad Rezaei and Mohammad Akbari and Saeed Ranjbar Alvar and Arezou Fatemi and Yong Zhang},
      year={2024},
      eprint={2408.05868},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.05868}, 
}
```
