import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from baranet import TrainingConfig
from baranet.engine import Trainer, load_checkpoint


class TinySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 1, 1)

    def forward(self, x, sp_map=None):
        return {"ba_logits": self.conv(x[:, -1]), "boundary_logits": None}


class ToyData(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(index)
        return {
            "inputs": torch.randn(3, 8, 16, 16, generator=gen),
            "ba_target": (torch.rand(1, 16, 16, generator=gen) > 0.5).float(),
            "fire_id": f"fire_{index % 2}",
        }


def test_trainer_runs_and_saves_checkpoints(tmp_path):
    cfg = TrainingConfig(
        epochs=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_epochs=1,
        mixed_precision=False,
        early_stopping_patience=10,
    )
    loader = DataLoader(ToyData(), batch_size=2)
    trainer = Trainer(TinySeg(), loader, loader, cfg, tmp_path, device="cpu", seed=42)
    result = trainer.fit()

    assert len(result["history"]) == 2
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "checkpoints" / "latest.pt").exists()
    assert (tmp_path / "logs" / "train.csv").exists()
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "environment.txt").exists()

    state = load_checkpoint(tmp_path / "checkpoints" / "best.pt", TinySeg())
    assert state["metadata"]["threshold"] == 0.5
    assert state["metadata"]["seed"] == 42
