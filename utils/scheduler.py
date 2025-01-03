import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR


'''
Linear Warmup & Linear/Cosine Decay Scheduler
'''
class LWarmupLCDecayScheduler(LambdaLR):
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, decay_type: str, total_steps: int,
                 warmup_start_lr: float = 0.0, warmup_end_lr: float = 1.0, decay_end_lr: float = 0.0) -> None:
        assert decay_type in ['linear', 'cosine'], 'Decay type should be either \'linear\' or \'cosine\''

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.decay_type = decay_type
        self.decay_end_lr = decay_end_lr
        self.total_steps = total_steps
        
        # Return the coefficient for the learning rate
        def lr_lambda(step: int) -> float:
            # Linear warmup phase
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Decay phase
            else:
                if decay_type == 'linear':
                    return max(0,0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
                elif decay_type == 'cosine':
                    return 0.5 * (1.0 + math.cos(math.pi * float(step - warmup_steps) / float(total_steps - warmup_steps)))
                else:
                    raise ValueError('Decay type should be either \'linear\' or \'cosine\'')

        
        super(LWarmupLCDecayScheduler, self).__init__(optimizer, lr_lambda)


# This is just for test
if __name__ == '__main__':
    optimizer = optim.Adam([torch.zeros(1)], lr=0.1)
    scheduler = LWarmupLCDecayScheduler(optimizer, warmup_steps=10, decay_type='linear', total_steps=100)

    lr_list = []
    for step in range(1, 100):
        scheduler.step()
        lr_list.append(optimizer.param_groups[0]['lr'])
    
    plt.plot(range(1, 100), lr_list)
    plt.show()