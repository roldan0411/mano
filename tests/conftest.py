import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
