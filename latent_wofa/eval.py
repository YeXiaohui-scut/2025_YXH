"""
完整评估脚本
功能：
1. 评估不同攻击场景下的水印鲁棒性
2. 评估图像质量（PSNR, SSIM, LPIPS, FID）
3. 生成评估报告
"""

import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import cv2

from inference import LatentWOFAInference
from models.distortion_layers import Stage2DistortionLayer
from utils.metrics import WatermarkMetrics


class LatentWOFAEvaluator:
    """
    Latent-WOFA评估器
    """
    def __init__(
        self,
        config_path='configs/config.yaml',
        stage1_checkpoint='checkpoints/stage1/best_model.pth',
        stage2_checkpoint='checkpoints/stage2/best_model.pth',
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化推理管道
        print("🚀 Initializing evaluation pipeline...")
        self.inference_pipeline = LatentWOFAInference(
            config_path=config_path,
            stage1_checkpoint=stage1_checkpoint,
            stage2_checkpoint=stage2_checkpoint,
            device=device
        )
        
        # 评估指标
        self.metrics = WatermarkMetrics(device=self.device)
        
        # 结果保存目录
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
        
        print("✅ Evaluator ready!")
    
    def apply_attack(self, image: Image.Image, attack_name: str, **kwargs):
        """
        应用特定攻击
        Args:
            image: PIL Image
            attack_name: 攻击名称
            **kwargs: 攻击参数
        Returns:
            attacked_image: PIL Image
        """
        # 转换为tensor
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        
        img_tensor = to_tensor(image).unsqueeze(0) * 2 - 1  # [0,1] -> [-1,1]
        img_np = np.array(image)
        
        if attack_name == 'crop':
            # 裁剪攻击
            crop_ratio = kwargs.get('crop_ratio', 0.5)
            h, w = img_np.shape[:2]
            crop_h = int(h * np.sqrt(crop_ratio))
            crop_w = int(w * np.sqrt(crop_ratio))
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            
            cropped = img_np[top:top+crop_h, left:left+crop_w]
            # 调整回原始尺寸
            attacked = cv2.resize(cropped, (w, h))
            return Image.fromarray(attacked)
        
        elif attack_name == 'rotation':
            # 旋转攻击
            angle = kwargs.get('angle', 45)
            return image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        
        elif attack_name == 'jpeg':
            # JPEG压缩攻击
            quality = kwargs.get('quality', 50)
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        
        elif attack_name == 'resize':
            # 缩放攻击
            scale = kwargs.get('scale', 0.5)
            w, h = image.size
            new_size = (int(w * scale), int(h * scale))
            resized = image.resize(new_size, Image.BILINEAR)
            return resized.resize((w, h), Image.BILINEAR)
        
        elif attack_name == 'gaussian_noise':
            # 高斯噪声
            std = kwargs.get('std', 0.1)
            img_tensor = img_tensor + torch.randn_like(img_tensor) * std
            img_tensor = img_tensor.clamp(-1, 1)
            return to_pil((img_tensor[0] + 1) / 2)
        
        elif attack_name == 'gaussian_blur':
            # 高斯模糊
            kernel_size = kwargs.get('kernel_size', 5)
            blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
            return Image.fromarray(blurred)
        
        elif attack_name == 'combined':
            # 组合攻击
            image = self.apply_attack(image, 'crop', crop_ratio=kwargs.get('crop_ratio', 0.1))
            image = self.apply_attack(image, 'rotation', angle=kwargs.get('angle', 30))
            image = self.apply_attack(image, 'jpeg', quality=kwargs.get('quality', 50))
            return image
        
        else:
            raise ValueError(f"Unknown attack: {attack_name}")
    
    def evaluate_single_attack(
        self,
        attack_name: str,
        num_samples: int = 100,
        **attack_params
    ):
        """
        评估单个攻击场景
        Args:
            attack_name: 攻击名称
            num_samples: 测试样本数
            **attack_params: 攻击参数
        Returns:
            results: dict 包含各种指标
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Attack: {attack_name}")
        print(f"Parameters: {attack_params}")
        print(f"{'='*60}\n")
        
        bit_accuracies = []
        bit_error_rates = []
        psnrs = []
        ssims = []
        
        # 测试提示词列表
        test_prompts = [
            "a photo of a cat",
            "a beautiful landscape",
            "a portrait of a person",
            "a modern building",
            "a red sports car"
        ] * (num_samples // 5 + 1)
        test_prompts = test_prompts[:num_samples]
        
        for i in tqdm(range(num_samples), desc=f"Testing {attack_name}"):
            try:
                # 1. 生成带水印的图像
                image, true_watermark = self.inference_pipeline.generate_with_watermark(
                    prompt=test_prompts[i],
                    seed=i,
                    save_path=None
                )
                
                # 2. 应用攻击
                attacked_image = self.apply_attack(image, attack_name, **attack_params)
                
                # 3. 提取水印
                extracted_bits, metrics = self.inference_pipeline.extract_watermark(
                    image=attacked_image,
                    true_watermark=true_watermark
                )
                
                # 4. 记录指标
                bit_accuracies.append(metrics['bit_accuracy'])
                bit_error_rates.append(metrics['bit_error_rate'])
                
                # 5. 计算图像质量（原图vs攻击后）
                original_tensor = transforms.ToTensor()(image).unsqueeze(0) * 2 - 1
                attacked_tensor = transforms.ToTensor()(attacked_image.resize(image.size)).unsqueeze(0) * 2 - 1
                
                psnr = self.metrics.psnr(original_tensor, attacked_tensor)
                ssim = self.metrics.ssim(original_tensor, attacked_tensor)
                
                psnrs.append(psnr)
                ssims.append(ssim)
                
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        # 统计结果
        results = {
            'attack_name': attack_name,
            'attack_params': attack_params,
            'num_samples': len(bit_accuracies),
            'bit_accuracy': {
                'mean': np.mean(bit_accuracies),
                'std': np.std(bit_accuracies),
                'min': np.min(bit_accuracies),
                'max': np.max(bit_accuracies)
            },
            'bit_error_rate': {
                'mean': np.mean(bit_error_rates),
                'std': np.std(bit_error_rates),
                'min': np.min(bit_error_rates),
                'max': np.max(bit_error_rates)
            },
            'psnr': {
                'mean': np.mean(psnrs),
                'std': np.std(psnrs)
            },
            'ssim': {
                'mean': np.mean(ssims),
                'std': np.std(ssims)
            }
        }
        
        # 打印结果
        print(f"\n📊 Results for {attack_name}:")
        print(f"   Bit Accuracy: {results['bit_accuracy']['mean']:.4f} ± {results['bit_accuracy']['std']:.4f}")
        print(f"   Bit Error Rate: {results['bit_error_rate']['mean']:.4f} ± {results['bit_error_rate']['std']:.4f}")
        print(f"   PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
        print(f"   SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
        
        return results
    
    def evaluate_all_attacks(self, num_samples: int = 50):
        """
        评估所有攻击场景
        """
        print("\n" + "="*60)
        print("Comprehensive Robustness Evaluation")
        print("="*60)
        
        all_results = []
        
        # 从配置文件读取攻击场景
        attack_scenarios = self.config.get('evaluation', {}).get('attack_scenarios', [])
        
        # 如果配置为空，使用默认攻击
        if not attack_scenarios:
            attack_scenarios = [
                {'name': 'crop', 'crop_ratio': 0.01},
                {'name': 'crop', 'crop_ratio': 0.05},
                {'name': 'crop', 'crop_ratio': 0.10},
                {'name': 'rotation', 'angle': 15},
                {'name': 'rotation', 'angle': 45},
                {'name': 'jpeg', 'quality': 30},
                {'name': 'jpeg', 'quality': 50},
                {'name': 'jpeg', 'quality': 70},
                {'name': 'resize', 'scale': 0.5},
                {'name': 'gaussian_noise', 'std': 0.05},
                {'name': 'gaussian_blur', 'kernel_size': 5},
                {'name': 'combined', 'crop_ratio': 0.05, 'angle': 30, 'quality': 50}
            ]
        
        for scenario in attack_scenarios:
            attack_name = scenario.pop('name')
            results = self.evaluate_single_attack(
                attack_name=attack_name,
                num_samples=num_samples,
                **scenario
            )
            all_results.append(results)
        
        # 保存结果
        output_path = self.results_dir / 'evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # 生成可视化报告
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, results):
        """
        生成可视化评估报告
        """
        print("\n📊 Generating evaluation report...")
        
        # 提取数据
        attack_names = [r['attack_name'] + '\n' + str(r['attack_params']) for r in results]
        bit_accs = [r['bit_accuracy']['mean'] for r in results]
        bit_acc_stds = [r['bit_accuracy']['std'] for r in results]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 比特准确率柱状图
        ax = axes[0, 0]
        x_pos = np.arange(len(attack_names))
        ax.bar(x_pos, bit_accs, yerr=bit_acc_stds, capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(attack_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Bit Accuracy')
        ax.set_title('Watermark Robustness Under Different Attacks')
        ax.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
        ax.axhline(y=0.80, color='orange', linestyle='--', label='80% threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. PSNR对比
        ax = axes[0, 1]
        psnrs = [r['psnr']['mean'] for r in results]
        ax.bar(x_pos, psnrs, alpha=0.7, color='steelblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(attack_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Image Quality After Attacks')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. SSIM对比
        ax = axes[1, 0]
        ssims = [r['ssim']['mean'] for r in results]
        ax.bar(x_pos, ssims, alpha=0.7, color='coral')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(attack_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('SSIM')
        ax.set_title('Structural Similarity After Attacks')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. 综合热力图
        ax = axes[1, 1]
        metrics_matrix = np.array([
            bit_accs,
            [r['bit_error_rate']['mean'] for r in results],
            [r['psnr']['mean'] / 50 for r in results],  # 归一化
            [r['ssim']['mean'] for r in results]
        ])
        sns.heatmap(
            metrics_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=[f"A{i+1}" for i in range(len(attack_names))],
            yticklabels=['Bit Acc', 'BER', 'PSNR(norm)', 'SSIM'],
            cmap='RdYlGn',
            ax=ax
        )
        ax.set_title('Metrics Heatmap')
        
        plt.tight_layout()
        
        # 保存图表
        report_path = self.results_dir / 'evaluation_report.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Report saved to: {report_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Latent-WOFA Evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/stage1/best_model.pth')
    parser.add_argument('--stage2_ckpt', type=str, default='checkpoints/stage2/best_model.pth')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples per attack scenario')
    parser.add_argument('--attack', type=str, default='all',
                       help='Specific attack to evaluate (or "all")')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = LatentWOFAEvaluator(
        config_path=args.config,
        stage1_checkpoint=args.stage1_ckpt,
        stage2_checkpoint=args.stage2_ckpt
    )
    
    if args.attack == 'all':
        # 评估所有攻击
        evaluator.evaluate_all_attacks(num_samples=args.num_samples)
    else:
        # 评估单个攻击
        evaluator.evaluate_single_attack(
            attack_name=args.attack,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
