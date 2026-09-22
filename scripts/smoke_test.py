"""Report the CUDA runtime and verify the two editable packages import."""

import importlib

import torch


def main() -> None:
    lawam = importlib.import_module("starVLA")
    project = importlib.import_module("lawam_adaptation")

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"gpu_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"gpu_0={torch.cuda.get_device_name(0)}")
        tensor = torch.ones(1, device="cuda")
        print(f"gpu_tensor={tensor.item()}")
    print(f"lawam_module={lawam.__name__}")
    print(f"project_version={project.__version__}")


if __name__ == "__main__":
    main()
