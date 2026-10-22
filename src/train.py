"""Reproducible CNN optimizer benchmark on the MNIST dataset.

The defaults intentionally approximate the supplied semester notebook: a 20%
training subset, a compact two-convolution CNN, batch size 64, and five epochs.
Unlike the original run, this script sets random seeds and records metrics in CSV.
"""
from __future__ import annotations
import argparse, csv, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LSGD(torch.optim.Optimizer):
    """Top-k sparsification + symmetric int8-simulated gradients + error feedback.

    This is the supplied notebook's software-side optimizer idea, not a verified
    microcontroller training implementation. `sparsity=0.9` drops about 90% of
    gradient elements before quantization; parameters remain float32.
    """
    def __init__(self, params, lr: float = 0.01, sparsity: float = 0.9, bits: int = 8):
        if not 0 <= sparsity < 1: raise ValueError('sparsity must be in [0, 1)')
        if bits < 2 or bits > 8: raise ValueError('bits must be in [2, 8]')
        super().__init__(params, dict(lr=lr, sparsity=sparsity, bits=bits))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            keep_fraction = 1.0 - group['sparsity']
            qmax = 2 ** (group['bits'] - 1) - 1
            for parameter in group['params']:
                if parameter.grad is None: continue
                grad = parameter.grad.detach()
                state = self.state[parameter]
                residual = state.setdefault('error_feedback', torch.zeros_like(grad))
                corrected = grad + residual
                flat = corrected.reshape(-1)
                k = max(1, int(round(keep_fraction * flat.numel())))
                indices = torch.topk(flat.abs(), min(k, flat.numel()), sorted=False).indices
                sparse = torch.zeros_like(flat)
                sparse[indices] = flat[indices]
                peak = sparse.abs().max()
                if peak.item() > 0:
                    scale = peak / qmax
                    quantized = torch.round(sparse / scale).clamp(-qmax, qmax) * scale
                else:
                    quantized = sparse
                quantized = quantized.view_as(grad)
                residual.copy_(corrected - quantized)
                parameter.add_(quantized, alpha=-group['lr'])
        return loss

OPTIMIZERS = ('SGD','SGD+Momentum','SGD+Nesterov','Adam','RMSprop','Adagrad','Adadelta','AdamW','L-SGD')
def make_optimizer(name, params):
    if name == 'SGD': return torch.optim.SGD(params, lr=.01)
    if name == 'SGD+Momentum': return torch.optim.SGD(params, lr=.01, momentum=.9)
    if name == 'SGD+Nesterov': return torch.optim.SGD(params, lr=.01, momentum=.9, nesterov=True)
    if name == 'Adam': return torch.optim.Adam(params, lr=.001)
    if name == 'RMSprop': return torch.optim.RMSprop(params, lr=.001)
    if name == 'Adagrad': return torch.optim.Adagrad(params, lr=.01)
    if name == 'Adadelta': return torch.optim.Adadelta(params, lr=1.0)
    if name == 'AdamW': return torch.optim.AdamW(params, lr=.001)
    if name == 'L-SGD': return LSGD(params, lr=.01, sparsity=.9, bits=8)
    raise ValueError(f'Unknown optimizer: {name}')

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run(args) -> None:
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    subset_n = int(len(train_data) * args.subset_fraction)
    split_generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_data), generator=split_generator)[:subset_n].tolist()
    subset = Subset(train_data, indices)
    results_dir = Path(args.output_dir); results_dir.mkdir(parents=True, exist_ok=True)
    result_rows, loss_rows = [], []
    selected = args.optimizers or list(OPTIMIZERS)
    for name in selected:
        seed_everything(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
        model = SimpleCNN().to(device)
        optimizer = make_optimizer(name, model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        epoch_losses=[]
        start=time.perf_counter()
        for epoch in range(args.epochs):
            model.train(); total=0.0
            for images, labels in train_loader:
                images, labels=images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(images),labels); loss.backward(); optimizer.step()
                total += loss.item()
            mean_loss=total/len(train_loader); epoch_losses.append(mean_loss)
            loss_rows.append({'optimizer':name,'epoch':epoch+1,'train_loss':f'{mean_loss:.6f}'})
            print(f'{name} | epoch {epoch+1}/{args.epochs} | loss {mean_loss:.4f}')
        elapsed=time.perf_counter()-start
        model.eval(); truth=[]; predicted=[]
        with torch.inference_mode():
            for images,labels in test_loader:
                output=model(images.to(device)).argmax(1).cpu().tolist()
                predicted.extend(output); truth.extend(labels.tolist())
        result_rows.append({'optimizer':name,'dataset':'MNIST','train_samples':subset_n,'test_samples':len(test_data),'epochs':args.epochs,'seed':args.seed,'accuracy_pct':round(accuracy_score(truth,predicted)*100,4),'macro_f1':round(f1_score(truth,predicted,average='macro'),6),'training_time_s':round(elapsed,3),'device':str(device)})
        print(f'{name} | accuracy={result_rows[-1]["accuracy_pct"]:.2f}% | macro_f1={result_rows[-1]["macro_f1"]:.4f} | time={elapsed:.2f}s')
    for filename, rows in [('reproduced_optimizer_results.csv',result_rows),('reproduced_epoch_losses.csv',loss_rows)]:
        if rows:
            with (results_dir/filename).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    print(f'Results saved to {results_dir.resolve()}')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default=str(ROOT/'data'))
    parser.add_argument('--output-dir',default=str(ROOT/'results'))
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--subset-fraction',type=float,default=.2)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--optimizers',nargs='+',choices=OPTIMIZERS,default=None)
    run(parser.parse_args())
