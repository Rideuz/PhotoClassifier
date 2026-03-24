from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    data_root: Path = Path("data/category_and_attribbute_prediction")
    anno_subdir: str = "Anno_coarse"
    image_subdir: str = "img_highres"

    random_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    epochs: int = 10
    batch_size: int = 32
    # 0 = загрузка картинок в главном процессе (часто узкое место и низкая загрузка GPU).
    # На Windows с Python 3.8+ обычно работает 2–8; при зависаниях поставьте 0.
    num_workers: int = 4
    # Val/test: без отдельных процессов. На Windows spawn снова грузит torch/CUDA в каждом воркере
    # и может дать WinError 1455 («файл подкачки слишком мал»).
    num_workers_eval: int = 0
    # Mixed precision на CUDA (быстрее и меньше VRAM на современных GPU)
    amp: bool = True
    lr: float = 3e-4
    weight_decay: float = 0.01
    image_size: int = 224

    # Обычный запуск train_resnet без флагов: не ~289k картинок, а стратифицированная выжимка
    # (см. --full-dataset для полного датасета — долго, для финальных экспериментов ВКР).
    default_max_samples: int = 20_000

    backbone: str = "resnet50"
    pretrained: bool = True

    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"type": 1.0, "color": 1.0, "style": 1.0}
    )

    output_dir: Path = Path("runs/resnet50_baseline")
    rebuild_splits: bool = False
    splits_file: str = "splits.npz"

    top_k_type: int = 3
    inference_warmup_batches: int = 3

    def validate_ratios(self) -> None:
        s = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Сумма долей сплитов должна быть 1.0, сейчас {s}")
