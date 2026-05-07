from .cifar import load_cifar10, load_cifar100, load_cifar100_vit
from .slimpajama import load_slimpajama
from .squad import load_squad
from .tiny_imagenet import load_tinyimagenet
from .wikitext import load_wikitext2

__all__ = [
    "load_cifar10",
    "load_cifar100",
    "load_cifar100_vit",
    "load_slimpajama",
    "load_squad",
    "load_tinyimagenet",
    "load_wikitext2",
]
