"""Central hyperparameter / path config. Edit this rather than hardcoding values in scripts."""

import os
import torch

# --- Paths ------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(MODEL_DIR, "outputs")

# --- Data ---------------------------------------------------------------
IN_CHANNELS = 4          # e.g. R, G, B, NIR
IMG_SIZE = 256
USE_MOCK_DATASET = False  # flip to False once CroplandDataset is wired in

# --- Model ----------------------------------------------------------------
BASE_CHANNELS = 64
EMBED_DIM = 256
TRANSFORMER_DEPTH = 4
NUM_HEADS = 8
DROPOUT = 0.1

# --- Loss weights ----------------------------------------------------------
W_BCE = 1.0
W_DICE = 1.0
W_BOUNDARY = 1.0
BOUNDARY_DILATE_PX = 1

# --- Training ---------------------------------------------------------------
BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"      # "cosine" or "step"
LR_STEP_SIZE = 15            # only used if LR_SCHEDULER == "step"
LR_GAMMA = 0.5                # only used if LR_SCHEDULER == "step"
WARMUP_EPOCHS = 2
GRAD_CLIP_NORM = 1.0

# --- Eval / inference ---------------------------------------------------------
PRED_THRESHOLD = 0.5
BOUNDARY_TOLERANCE_PX = 2

# --- Misc ---------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if (torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1") else "cpu"
