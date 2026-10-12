"""octrecon: GPU-accelerated reconstruction of tiled Thorlabs OCT scans (port of myOCT)."""
import os as _os

# avoid oversubscription between numba / pocketfft threads and OpenBLAS (measured 26s -> 3s stitch)
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# number CUDA GPUs like nvidia-smi (PCI bus order); must be set before CUDA is initialised
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
