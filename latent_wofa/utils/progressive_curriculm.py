"""
渐进式课程学习控制器
根据训练进度动态调整攻击强度
"""

import numpy as np


class ProgressiveCurriculum:
    """
    渐进式课程学习调度器
    """
    def __init__(self, config, stage='stage1'):
        self.config = config[stage]['progressive']
        self.stage = stage
        
        self.start_epoch = self.config['start_epoch']
        self.medium_epoch = self.config['medium_epoch']
        self.final_epoch = self.config['final_epoch']
        
    def get_progressive_level(self, current_epoch):
        """
        根据当前epoch返回攻击级别
        Args:
            current_epoch: int
        Returns:
            level: str, one of ['initial', 'medium', 'final']
        """
        if current_epoch < self.medium_epoch:
            return 'initial'
        elif current_epoch < self.final_epoch:
            return 'medium'
        else:
            return 'final'
    
    def get_interpolated_config(self, current_epoch):
        """
        获取插值后的配置（平滑过渡）
        Args:
            current_epoch: int
        Returns:
            interpolated_config: dict
        """
        level = self.get_progressive_level(current_epoch)
        
        if level == 'initial':
            # 在initial阶段内线性插值
            progress = (current_epoch - self.start_epoch) / max(1, self.medium_epoch - self.start_epoch)
            return self._interpolate_configs('initial', 'medium', progress)
        
        elif level == 'medium':
            # 在medium阶段内线性插值
            progress = (current_epoch - self.medium_epoch) / max(1, self.final_epoch - self.medium_epoch)
            return self._interpolate_configs('medium', 'final', progress)
        
        else:
            # 已到达final阶段
            return self.config['final'] if 'final' in self.config else {}
    
    def _interpolate_configs(self, level1, level2, progress):
        """
        在两个配置级别之间插值
        Args:
            level1: str, 起始级别
            level2: str, 目标级别
            progress: float, [0, 1] 插值进度
        Returns:
            interpolated: dict
        """
        config1 = self.config.get(level1, {})
        config2 = self.config.get(level2, {})
        
        interpolated = {}
        for key in config1.keys():
            if key in config2:
                val1 = config1[key]
                val2 = config2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # 数值插值
                    interpolated[key] = val1 + (val2 - val1) * progress
                else:
                    # 非数值：直接使用level2的值
                    interpolated[key] = val2 if progress > 0.5 else val1
            else:
                interpolated[key] = config1[key]
        
        return interpolated
    
    def should_update_distortion(self, current_epoch):
        """
        判断是否需要更新失真层配置
        Args:
            current_epoch: int
        Returns:
            bool
        """
        return current_epoch in [self.medium_epoch, self.final_epoch]
    
    def get_description(self, current_epoch):
        """
        获取当前训练阶段的描述
        """
        level = self.get_progressive_level(current_epoch)
        descriptions = {
            'initial': f"🟢 Initial Phase (Epoch {current_epoch}): Mild attacks for warm-up",
            'medium': f"🟡 Medium Phase (Epoch {current_epoch}): Moderate attacks",
            'final': f"🔴 Final Phase (Epoch {current_epoch}): Extreme attacks"
        }
        return descriptions.get(level, "Unknown phase")


# ============ 测试代码 ============

if __name__ == "__main__":
    import yaml
    
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    curriculum = ProgressiveCurriculum(config, stage='stage1')
    
    # 模拟训练过程
    test_epochs = [0, 15, 30, 45, 60, 75, 90]
    
    print("Progressive Curriculum Schedule:")
    print("=" * 60)
    for epoch in test_epochs:
        level = curriculum.get_progressive_level(epoch)
        desc = curriculum.get_description(epoch)
        print(f"\n{desc}")
        print(f"Level: {level}")
        
        if curriculum.should_update_distortion(epoch):
            print("⚠️  DISTORTION LAYER UPDATE REQUIRED!")
