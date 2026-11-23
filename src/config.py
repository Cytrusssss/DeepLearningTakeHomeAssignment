from pathlib import Path

SEED = 42
DATASET = "FashionMNIST"
BATCH_SIZE = 128
NUM_WORKERS = 2

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
FIG_DIR = PROJECT_ROOT / "figures"
RESULT_DIR = PROJECT_ROOT / "results"
SPLIT_FILE = RESULT_DIR / f"split_indices_{DATASET}_seed{SEED}.json"

IN_DIM = 28 * 28
IMG_SHAPE = (1, 28, 28)
Z_DIM_AE = 64
Z_DIM_GAN = 100
T_DIFF = 1000
