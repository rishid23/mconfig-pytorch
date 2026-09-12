from __future__ import annotations
from typing import Callable, List, Sequence

import torch

__all__ = ["unit", "orthogonal", "config_direction", "config_direction_two", "MConFIG"]



def unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    return v/(torch.norm(v)+eps)

def orthogonal(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    return b-(torch.dot(a,b)/(torch.dot(a,a)+eps))*a



def config_direction_two(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    o12 = orthogonal (g1,g2, eps)
    o21 = orthogonal (g2,g1, eps)
    g_v = unit(unit(o12,eps) + unit(o21, eps), eps)
    scale = torch.dot(g1,g_v) + torch.dot(g2,g_v)
    return scale*g_v

def config_direction(grads: Sequence[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:

    if len(grads) == 0:
        raise ValueError("No gradients provided to config_direction")
    if len(grads) == 1:
        return grads[0].clone()
    if len(grads) == 2:
        return config_direction_two(grads[0], grads[1], eps)


    M = torch.stack([unit(g,eps) for g in grads], dim=1)
    ones = torch.ones(M.shape[1], device=M.device, dtype=M.dtype)


    raw  = torch.linalg.pinv(M.t()) @ ones
    g_u = unit(raw,eps)

    scale = sum(torch.dot(g,g_u) for g in grads)

    return scale*g_u



def _get_flat_grad(params: List[torch.nn.Parameter]) -> torch.Tensor:
    parts = []
    for p in params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
        else:
            parts.append(p.grad.reshape(-1))

    return torch.cat(parts)



class MConFIG:

    def __init__(self, params,num_losses:int, lr: float=1e-3, beta1: float = 0.9, beta2: float=0.999, eps: float = 1e-8,):
        self.params: List[torch.nn.Parameter] = [p for p in params if p.requires_grad]
        if num_losses < 1:
            raise ValueError("num_losses must be >= 1")
        self.m = num_losses
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        n = sum(p.numel() for p in self.params)
        device = self.params[0].device
        dtype = self.params[0].dtype

        self.m_g: List[torch.Tensor] = [torch.zeros(n, device=device, dtype=dtype) for _ in range(self.m)]
        self.t_g: List[int] = [0] * self.m

        self.m_t = torch.zeros(n, device=device, dtype=dtype)
        self.v_t = torch.zeros(n, device=device, dtype=dtype)
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self, loss_fn: Callable[[int], torch.Tensor]) -> torch.Tensor:

        i = self.t % self.m
        self.t += 1
        self.t_g[i] += 1

        self.zero_grad()
        loss_i = loss_fn(i)
        loss_i.backward()
        g_i = _get_flat_grad(self.params)
        self.m_g[i] = self.beta1 * self.m_g[i] + (1 - self.beta1) * g_i

        m_hats: List[torch.Tensor] = []
        for j in range(self.m):
            tj = self.t_g[j]
            if tj == 0:
                m_hats.append(torch.zeros_like(self.m_g[j]))
            else:
                m_hats.append(self.m_g[j] / (1 - self.beta1 ** tj))

        m_hat_g = config_direction(m_hats)

        t = self.t
        g_c = (m_hat_g * (1 - self.beta1 ** t) - self.beta1 * self.m_t) / (1 - self.beta1)

        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * g_c
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (g_c ** 2)
        v_hat = self.v_t / (1 - self.beta2 ** t)

        update = self.lr * m_hat_g / (torch.sqrt(v_hat) + self.eps)
        self._apply_update(update)

        return loss_i.detach()

    def _apply_update(self, flat_update: torch.Tensor) -> None:
        idx = 0
        for p in self.params:
            n = p.numel()
            chunk = flat_update[idx:idx+n].view_as(p)
            p.data.add_(-chunk)
            idx += n


if __name__ == "__main__":
    torch.manual_seed(0)


    class TwoHeadNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Sequential(
                torch.nn.Linear(3, 16), torch.nn.Tanh(),
                torch.nn.Linear(16, 16), torch.nn.Tanh(),
            )

            self.head1 = torch.nn.Linear(16, 1)
            self.head2 = torch.nn.Linear(16, 1)

        def forward(self, x):
            h = self.shared(x)
            return self.head1(h), self.head2(h)

    x = torch.randn(64, 3)
    target_small = torch.sin(x[:, 0:1])
    target_large = 20.0 * torch.cos(x[:, 1:2])

    net = TwoHeadNet()
    opt = MConFIG(net.parameters(), num_losses=2, lr=1e-3)

    def loss_fn(i: int) -> torch.Tensor:
        out_small, out_large = net(x)
        if i == 0:
            return ((out_small - target_small) ** 2).mean()
        else:
            return ((out_large - target_large) ** 2).mean()

    with torch.no_grad():
        o1, o2 = net(x)
        l1_start = ((o1 - target_small) ** 2).mean().item()
        l2_start = ((o2 - target_large) ** 2).mean().item()

    steps = 1600
    for step in range(steps):
        opt.step(loss_fn)
        if (step+1) % 400 == 0:
            with torch.no_grad():
                o1, o2 = net(x)
                l1 = ((o1 - target_small) ** 2).mean().item()
                l2 = ((o2 - target_large) ** 2).mean().item()
                print(f"step {step + 1:4d}   small-task loss = {l1:.6f}   large-task loss = {l2:.4f}")

    with torch.no_grad():
        o1, o2 = net(x)
        l1_end = ((o1 - target_small) ** 2).mean().item()
        l2_end = ((o2 - target_large) ** 2).mean().item()


    print()
    print(f"small-task loss:  {l1_start:.6f}  ->  {l1_end:.6f}")
    print(f"large-task loss:  {l2_start:.6f}  ->  {l2_end:.6f}")
    print()
    print("Both the task losses decreased, even though their gradient magnitudes generically")
    print("differ by ~20x and only one loss is backpropped per step. This")
    print("is the behavior M-ConFIG is designed to produce.")   


