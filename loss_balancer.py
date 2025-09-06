"""
Loss balancer for stabilizing training with varying gradient scales.
Based on the actual EnCodec codebase implementation.
"""

import torch
import typing as tp
from collections import defaultdict


class LossBalancer:
    """
    Loss balancer that normalizes gradients from different losses to stabilize training.
    
    Based on the actual EnCodec implementation from the codebase.
    """
    
    def __init__(self, 
                 weights: tp.Dict[str, float],
                 balance_grads: bool = True,
                 total_norm: float = 1.0,
                 ema_decay: float = 0.999,
                 per_batch_item: bool = True,
                 epsilon: float = 1e-12,
                 monitor: bool = False):
        """
        Args:
            weights: Dictionary of loss names to weights
            balance_grads: Whether to balance gradients (True) or just weighted sum (False)
            total_norm: Target total gradient norm
            ema_decay: EMA decay rate for gradient norms
            per_batch_item: Whether to compute norms per batch item
            epsilon: Small epsilon for numerical stability
            monitor: Whether to monitor gradient ratios
        """
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm or 1.0
        self.epsilon = epsilon
        self.monitor = monitor
        self.balance_grads = balance_grads
        self._metrics: tp.Dict[str, tp.Any] = {}
        
        # Simple averager for EMA
        self.averager = self._create_averager(ema_decay)
        
        print(f"LossBalancer initialized:")
        print(f"  Weights: {self.weights}")
        print(f"  Balance grads: {self.balance_grads}")
        print(f"  Total norm: {self.total_norm}")
        print(f"  EMA decay: {ema_decay}")
    
    def _create_averager(self, ema_decay: float):
        """Create a simple EMA averager."""
        class SimpleAverager:
            def __init__(self, decay):
                self.decay = decay
                self.state = {}
            
            def __call__(self, metrics, count=1):
                result = {}
                for k, v in metrics.items():
                    if k in self.state:
                        self.state[k] = self.decay * self.state[k] + (1 - self.decay) * v
                    else:
                        self.state[k] = v
                    result[k] = self.state[k]
                return result
        
        return SimpleAverager(ema_decay)
    
    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor) -> torch.Tensor:
        """
        Compute the backward and return the effective train loss.
        
        Args:
            losses: Dictionary with the same keys as self.weights
            input: The input of the losses, typically the output of the model
            
        Returns:
            Effective loss (sum of balanced losses)
        """
        norms = {}
        grads = {}
        
        for name, loss in losses.items():
            # Compute partial derivative of the loss with respect to the input
            grad, = torch.autograd.grad(loss, [input], retain_graph=True)
            
            if self.per_batch_item:
                # We do not average the gradient over the batch dimension
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims, p=2).mean()
            else:
                norm = grad.norm(p=2)
            
            norms[name] = norm
            grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(grad)
        
        # Average norms across workers (simplified for single GPU)
        avg_norms = self.averager(norms, count)
        
        # We approximate the total norm of the gradient as the sums of the norms
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            # Store the ratio of the total gradient represented by each loss
            for k, v in avg_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        assert total_weights > 0.
        desired_ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad = torch.zeros_like(input)
        effective_loss = torch.tensor(0., device=input.device, dtype=input.dtype)
        
        for name, avg_norm in avg_norms.items():
            if self.balance_grads:
                # g_balanced = g / avg(||g||) * total_norm * desired_ratio
                scale = desired_ratios[name] * self.total_norm / (self.epsilon + avg_norm)
            else:
                # We just do regular weighted sum of the gradients
                scale = self.weights[name]
            
            out_grad.add_(grads[name], alpha=scale)
            effective_loss += scale * losses[name].detach()
        
        # Send the computed partial derivative with respect to the output of the model to the model
        input.backward(out_grad)
        return effective_loss
    
    @property
    def metrics(self) -> tp.Dict[str, float]:
        """Return current gradient norm metrics."""
        return self._metrics


def create_loss_balancer(reconstruction_weight: float = 1.0,
                        adversarial_weight: float = 3.0,
                        feature_matching_weight: float = 3.0,
                        balance_grads: bool = True,
                        total_norm: float = 1.0,
                        ema_decay: float = 0.999) -> LossBalancer:
    """
    Create a loss balancer with paper parameters.
    
    Args:
        reconstruction_weight: Weight for reconstruction loss
        adversarial_weight: Weight for adversarial loss  
        feature_matching_weight: Weight for feature matching loss
        balance_grads: Whether to balance gradients
        total_norm: Target total gradient norm
        ema_decay: EMA decay rate
    """
    weights = {
        'reconstruction': reconstruction_weight,
        'adversarial': adversarial_weight,
        'feature_matching': feature_matching_weight
    }
    
    return LossBalancer(
        weights=weights,
        balance_grads=balance_grads,
        total_norm=total_norm,
        ema_decay=ema_decay
    )


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("LOSS BALANCER EXAMPLE")
    print("=" * 60)
    
    # Create dummy model output
    model_output = torch.randn(2, 32, 1000, requires_grad=True)
    
    # Create dummy losses
    losses = {
        'reconstruction': torch.tensor(1.0, requires_grad=True),
        'adversarial': torch.tensor(0.5, requires_grad=True),
        'feature_matching': torch.tensor(0.3, requires_grad=True)
    }
    
    # Create balancer
    balancer = create_loss_balancer(
        reconstruction_weight=1.0,
        adversarial_weight=3.0,
        feature_matching_weight=3.0
    )
    
    # Test backward pass
    effective_loss = balancer.backward(losses, model_output)
    print(f"Effective loss: {effective_loss.item():.6f}")
    print(f"Metrics: {balancer.metrics}")
    
    print("✅ Loss balancer working correctly!")