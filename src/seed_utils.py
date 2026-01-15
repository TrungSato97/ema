# # seed_utils.py
# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# import os
# import random
# import numpy as np
# import torch

# def set_global_seed(seed: int, deterministic: bool = True):
#     """Set seeds for python, numpy and torch. Call BEFORE creating dataloaders/models/transforms."""
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     if deterministic:
#         # Try new API, fallback to cudnn flags
#         try:
#             torch.use_deterministic_algorithms(True)
#         except Exception:
#             pass
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     else:
#         try:
#             torch.use_deterministic_algorithms(False)
#         except Exception:
#             pass
#         torch.backends.cudnn.deterministic = False
#         torch.backends.cudnn.benchmark = True

# seed_utils.py
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int, deterministic: bool = True):
    """Set seeds for python, numpy and torch. Call BEFORE creating dataloaders/models/transforms."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # NOTE: CUBLAS_WORKSPACE_CONFIG should ideally be set in ENTRYPOINT before any torch import.
        # Keeping it here as a fallback (works best if CUDA context not initialized yet).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Try deterministic algorithms (prefer warn_only if available)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # older torch without warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
