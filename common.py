"""Shared config, data pipeline, and checkpoint utilities used by train/test/calibration."""
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from vit import build_vit_cifar
from resnet import build_resnet50_cifar

# ----------------------------------------------------------------------------- config
SEED = 42
NUM_CLASSES = 10
CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kaggle-safe project dirs (falls back to a local folder when not on Kaggle)
if os.path.isdir('/kaggle/working'):
    PROJECT_DIR = '/kaggle/working/cifar10_ood_project'
else:
    PROJECT_DIR = os.path.join(os.getcwd(), 'cifar10_ood_project')
CKPT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
for _d in (PROJECT_DIR, CKPT_DIR, DATA_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

# Model registry: short key -> checkpoint name, builder, display title
MODELS = {
    'vit': dict(ckpt_name='vit_scratch_cifar10_v2', build=build_vit_cifar,
                title='ViT (scratch, 4x4 patch)'),
    'resnet': dict(ckpt_name='resnet50_cifar10_v2', build=build_resnet50_cifar,
                   title='ResNet-50'),
}


def resolve_models(choice):
    """'vit' -> ['vit'], 'resnet' -> ['resnet'], 'both' -> ['vit', 'resnet']."""
    return list(MODELS) if choice == 'both' else [choice]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
def resolve_cifar_root(user_root, expected_subdir):
    """Return the dir to pass as root= to torchvision. Accepts either the parent of the
    standard layout or the extracted batches folder itself."""
    user_root = os.path.abspath(user_root)
    if os.path.isdir(os.path.join(user_root, expected_subdir)):
        return user_root
    if os.path.basename(user_root) == expected_subdir and os.path.isdir(user_root):
        return os.path.dirname(user_root)
    return user_root  # not found yet -- torchvision can download here


def add_data_args(parser):
    """CLI flags shared by every script."""
    parser.add_argument('--cifar10-root', default=DATA_DIR,
                        help="Folder containing 'cifar-10-batches-py' (or that folder itself).")
    parser.add_argument('--cifar100-root', default=DATA_DIR,
                        help="Folder containing 'cifar-100-python' (or that folder itself).")
    parser.add_argument('--no-download', action='store_true',
                        help='Do not download CIFAR; use files already on disk (offline).')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=2)
    return parser


def build_transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return train_tf, test_tf


def build_loaders(cifar10_root=DATA_DIR, cifar100_root=DATA_DIR, download=True,
                  batch_size=128, num_workers=2, seed=SEED):
    """Returns dict of DataLoaders: train / val / test / ood.

    train: 45k CIFAR-10 train images (augmented)    val: 5k held-out train images (clean)
    test : 10k CIFAR-10 test images                 ood: CIFAR-100 test set (surrogate OOD,
                                                         never used to train classifiers)
    The train/val split is fixed by `seed`, so every script sees the same split."""
    c10 = resolve_cifar_root(cifar10_root, 'cifar-10-batches-py')
    c100 = resolve_cifar_root(cifar100_root, 'cifar-100-python')
    train_tf, test_tf = build_transforms()

    try:
        full_train_aug = torchvision.datasets.CIFAR10(c10, train=True, download=download, transform=train_tf)
        full_train_clean = torchvision.datasets.CIFAR10(c10, train=True, download=download, transform=test_tf)
        test_set = torchvision.datasets.CIFAR10(c10, train=False, download=download, transform=test_tf)
    except RuntimeError as e:
        raise RuntimeError(
            f"Couldn't load CIFAR-10 from '{c10}' (download={download}). Make sure the path "
            "contains 'cifar-10-batches-py' in standard torchvision pickle format.") from e
    try:
        ood_set = torchvision.datasets.CIFAR100(c100, train=False, download=download, transform=test_tf)
    except RuntimeError as e:
        raise RuntimeError(
            f"Couldn't load CIFAR-100 from '{c100}' (download={download}). Make sure the path "
            "contains 'cifar-100-python'.") from e

    n_train = len(full_train_aug)
    val_size = int(0.1 * n_train)
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_clean, val_idx)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    loaders = dict(
        train=DataLoader(train_set, shuffle=True, drop_last=True, **kw),
        val=DataLoader(val_set, shuffle=False, **kw),
        test=DataLoader(test_set, shuffle=False, **kw),
        ood=DataLoader(ood_set, shuffle=False, **kw),
    )
    print(f"train {len(train_set)} | val {len(val_set)} | test {len(test_set)} | ood {len(ood_set)}")
    return loaders


def loaders_from_args(args):
    return build_loaders(args.cifar10_root, args.cifar100_root, not args.no_download,
                         args.batch_size, args.num_workers)


# ----------------------------------------------------------------------------- checkpoints
def _empty_history():
    return {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


def save_checkpoint(name, epoch, model, optimizer, scheduler, best_acc, history):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict() if scheduler is not None else None,
        'best_acc': best_acc,
        'history': history,
    }, os.path.join(CKPT_DIR, f"{name}.pt"))
    # permanent best-weights copy
    if history and history['val_acc'][-1] == best_acc:
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}_best.pt"))


def load_checkpoint(name, model, optimizer, scheduler):
    """Resume from the last saved epoch if a checkpoint exists. Returns (epoch, best_acc, history)."""
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        return 0, 0.0, _empty_history()
    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own file
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    if scheduler is not None and ckpt['sched_state'] is not None:
        scheduler.load_state_dict(ckpt['sched_state'])
    history = ckpt['history']
    history.setdefault('train_acc', [])
    print(f"Resumed '{name}' from epoch {ckpt['epoch']} (best_acc={ckpt['best_acc']:.4f})")
    return ckpt['epoch'], ckpt['best_acc'], history


def load_trained_model(key):
    """Build the model for `key` and load its best saved weights, ready for eval."""
    cfg = MODELS[key]
    best_path = os.path.join(CKPT_DIR, f"{cfg['ckpt_name']}_best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"No trained weights at {best_path}. Run: python train.py --model {key}")
    model = cfg['build']().to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    return model
