import os
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

class Logger:
    """统一日志记录器，同时支持 TensorBoard 和 WandB"""
    
    def __init__(self, project_name, run_name, config, use_wandb=True, use_tensorboard=True, log_dir='./logs'):
        """
        初始化日志记录器
        
        Args:
            project_name (str): 项目名称
            run_name (str): 运行名称
            config: 配置对象
            use_wandb (bool): 是否使用 wandb
            use_tensorboard (bool): 是否使用 tensorboard
            log_dir (str): tensorboard 日志保存目录
        """
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # 初始化 TensorBoard
        if self.use_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
            print(f"TensorBoard logs will be saved to: {os.path.join(log_dir, run_name)}")
        else:
            self.tensorboard_writer = None
        
        # 初始化 WandB
        if self.use_wandb:
            try:
                wandb.init(project=project_name, name=run_name, config=config)
                self.wandb = wandb
                print(f"WandB initialized for project: {project_name}, run: {run_name}")
            except Exception as e:
                print(f"Failed to initialize WandB: {e}. Continuing without WandB.")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None
        
        self.config = config
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False
    
    def log(self, data_dict, step=None):
        """
        记录日志数据
        
        Args:
            data_dict (dict): 要记录的数据字典
            step (int, optional): 训练步数
        """
        # 记录到 TensorBoard
        if self.use_tensorboard and self.tensorboard_writer:
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, global_step=step if step is not None else 0)
            self.tensorboard_writer.flush()
        
        # 记录到 WandB
        if self.use_wandb and self.wandb:
            self.wandb.log(data_dict, step=step)
    
    def watch(self, model, log="gradients", log_freq=10):
        """
        监控模型的梯度和参数
        
        Args:
            model: PyTorch 模型
            log (str): 记录类型 ("gradients", "parameters", "all")
            log_freq (int): 记录频率
        """
        # WandB 支持 watch
        if self.use_wandb and self.wandb:
            self.wandb.watch(model, log=log, log_freq=log_freq)
        
        # TensorBoard 可以通过 add_graph 记录模型结构
        if self.use_tensorboard and self.tensorboard_writer:
            # 注意：add_graph 只需要调用一次，通常在训练前
            # 这里不自动调用，可以在外部手动调用
            pass
    
    def save_model(self, model, save_dir, save_name, ep=None):
        """
        保存模型
        
        Args:
            model: PyTorch 模型
            save_dir (str): 保存目录
            save_name (str): 保存文件名
            ep (int, optional): epoch 数
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if ep is not None:
            save_path = os.path.join(save_dir, f"{save_name}{ep}.pth")
        else:
            save_path = os.path.join(save_dir, f"{save_name}.pth")
        
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
        
        # 如果使用 WandB，同时上传到 WandB
        if self.use_wandb and self.wandb:
            try:
                self.wandb.save(save_path, base_path="../")
                print(f"Model also saved to WandB: {save_path}")
            except Exception as e:
                print(f"Failed to save model to WandB: {e}")
    
    def close(self):
        """关闭日志记录器"""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.use_wandb and self.wandb:
            self.wandb.finish()
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()
