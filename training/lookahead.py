from collections import defaultdict
import torch

class AutoLookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha=0.5, auto_k = None):
        self.optimizer = optimizer
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.auto_k = auto_k
        self.steps = 0

        self.reset_lookahead()
        
    def init_lookahead_step(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        
        for group in self.param_groups:
            for fast in group["params"]:
                param_state = self.state[fast]
                # Reset slow_params
                param_state["slow_params"] = torch.zeros_like(fast.data)
                slow = param_state["slow_params"]
                
                # slow <- slow + alpha * (fast - slow)
                slow += (fast.data - slow) * alpha
                fast.data.copy_(slow)
        
    def lookahead_step(self):
        for group in self.param_groups:
            for fast in group["params"]:
                param_state = self.state[fast]
                if "slow_params" not in param_state:
                    param_state["slow_params"] = torch.zeros_like(fast.data)
                    param_state["slow_params"].copy_(fast.data)
                slow = param_state["slow_params"]
                # slow <- slow + alpha * (fast - slow)
                slow += (fast.data - slow) * self.alpha
                fast.data.copy_(slow)
                
    def reset_lookahead(self):
        for group in self.param_groups:
            for fast in group["params"]:
                param_state = self.state[fast]
                param_state["slow_params"] = torch.zeros_like(fast.data)
                param_state["slow_params"].copy_(fast.data)

    def step(self, closure = None):
        loss = self.optimizer.step(closure)

        if self.auto_k is not None:
          if (self.steps + 1) % self.auto_k == 0:
            self.lookahead_step()
          
          self.steps += 1

        return loss
   