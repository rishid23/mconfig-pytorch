# mconfig-pytorch

I have worked on this side project for the past couple days... It is based on the M-ConFIG method (Algorithm 1) from a paper called ConFIG, by Qiang Liu, Mengyu Chu, and Nils Thuerey, published at ICLR 2025. They came up with this to help train Physics-Informed Neural Networks (or PINNs) when gradients from different losses clash.

It is specifically attempted and designed to optimize several losses at the same time, which happens a lot in PINNs. When the gradients from these losses point in different or conflicting directions, it can make training unnecessarily unstable or less effective, since whichever loss has the bigger gradient tends to dominate the update. M-ConFIG solves this by combining the gradients in a way that avoids the conflicts alltogether, and it does it while only backpropagating one loss per step instead of all of them at once, while using momentum to keep track of the others. This helps the model improve every loss together, more smoothly and more efficiently, without needing a full backward pass per loss every step.

Here is Figure 1 from the paper, showing the overall intuition behind the conflict problem and how ConFIG resolves it compared to plain Adam:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/53e91e51-b273-4387-a289-4c0b6dc8551a" />

---

### Example

```python
import torch
from mconfig import MConFIG

class TwoHeadNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(3, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 16), torch.nn.Tanh(),
        )
        self.head1 = torch.nn.Linear(16, 1)
        self.head2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head1(h), self.head2(h)

x = torch.randn(64, 3)
target_small = torch.sin(x[:, 0:1])
target_large = 20.0 * torch.cos(x[:, 1:2])

net = TwoHeadNet()
opt = MConFIG(net.parameters(), num_losses=2, lr=1e-3)

def loss_fn(i):
    out_small, out_large = net(x)
    return ((out_small - target_small) ** 2).mean() if i == 0 \
        else ((out_large - target_large) ** 2).mean()

for step in range(1600):
    opt.step(loss_fn)
```

---

### Requirements

* python 3.9 or higher
* pyTorch ≥ 1.13
* no other dependencies are really necessary

---

### Results

On a toy two-task problem where one loss has gradients roughly 20x bigger than the other, the optimizer reduced both losses together:

```
step  400   small-task loss = 0.000177   large-task loss = 99.4770
step  800   small-task loss = 0.000098   large-task loss = 60.3211
step 1200   small-task loss = 0.000048   large-task loss = 29.7918
step 1600   small-task loss = 0.000036   large-task loss = 20.1890

small-task loss:  0.360519 to 0.000036
large-task loss:  240.514221 to 20.189047
```

Below is Figure 10 from the original paper, showing test MSE against wall clock time for Adam, ConFIG, and M-ConFIG on one of their real PINN benchmarks:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/1d604441-38b2-43af-b94e-9fff9da643c9" />


you can see that M-ConFIG keeps up with plain ConFIG despite doing a lot less work per step, since it only backprops one loss at a time instead of all of them.

---

### Code Layout

* `mconfig.py`: full implementation of the optimizer plus a toy test at the bottom
* no extra files, should all be self-contained

---

### Paper, credit to all visualizations

**ConFIG: Towards Conflict-Free Training of Physics Informed Neural Networks**
ICLR 2025 (https://arxiv.org/abs/2408.11104)

---

### How it works

* gradients from each loss get flattened into one vector, then combined using the ConFIG operator, then unflattened and applied to the model
* only one loss is backpropagated per step, chosen round robin, and its own momentum buffer is updated while the others carry over from earlier steps
* `eps` avoids division by zero when normalizing

The figure below (Figure 2 from the paper) illustrates the geometric idea behind how ConFIG combines two conflicting gradients:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/9fff1656-418f-4032-9083-022ed33220d1" />


Here is the actual algorithm box from the paper (Algorithm 1), which is what `step()` follows line by line:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/0b0680b6-b4c1-40e5-aa3b-065bf97939e4" />


---

### Learning Experience

When I first started this project, I mainly focused on getting the optimizer to run on some toy data. I would not say I was super confident with all the math behind it at first, but working through the implementation pushed me to understand how PyTorch handles gradients, especially when you are dealing with multiple losses and momentum buffers for each one separately. It builds upon my previous attempt at implementing the Dual Cone Gradient Descent as well. Flattening gradients, reassigning them, and reconstructing an equivalent gradient from combined momentum was tricky at first but became a lot clearer once I traced through it step by step.

Since I am interested in physics, building this project gave me a chance to connect what I have learned in class to real-world problems where physics and machine learning meet. Even if I did not fully grasp every single equation on the first pass, the hands-on coding helped me understand the core ideas and why the algorithm matters in a bigger setting.

---

### Footnotes

* maybe integrate with torch.optim if needed
* flattening and unflattening gradients was a bit tricky but helped me understand how optimization works better
* since I did not have a good GPU to run on, I tested on simple synthetic data to check that the optimizer works..
