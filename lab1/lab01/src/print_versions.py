import sys
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
import seaborn
import torch
import torchvision
import torchinfo
import onnx
import onnxruntime
import mlflow
import memory_profiler
import psutil
import fastapi
import uvicorn
import pytest
import httpx
import locust
import requests
import pyarrow
import joblib
import tqdm

versions = [
    f"Python: {sys.version}",
    f"numpy: {np.__version__}",
    f"pandas: {pd.__version__}",
    f"scikit-learn: {sklearn.__version__}",
    f"scipy: {scipy.__version__}",
    f"matplotlib: {matplotlib.__version__}",
    f"seaborn: {seaborn.__version__}",
    f"torch: {torch.__version__}",
    f"torchvision: {torchvision.__version__}",
    f"torchinfo: {torchinfo.__version__}",
    f"onnx: {onnx.__version__}",
    f"onnxruntime: {onnxruntime.__version__}",
    f"mlflow: {mlflow.__version__}",
    f"memory_profiler: {memory_profiler.__version__}",
    f"psutil: {psutil.__version__}",
    f"fastapi: {fastapi.__version__}",
    f"uvicorn: {uvicorn.__version__}",
    f"pytest: {pytest.__version__}",
    f"httpx: {httpx.__version__}",
    f"locust: {locust.__version__}",
    f"requests: {requests.__version__}",
    f"pyarrow: {pyarrow.__version__}",
    f"joblib: {joblib.__version__}",
    f"tqdm: {tqdm.__version__}"
]

for v in versions:
    print(v)

with open("lab01/results/versions.txt", "w") as f:
    f.write("\n".join(versions) + "\n")
