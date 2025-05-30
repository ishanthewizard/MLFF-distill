"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

if __name__ == "__main__":
    import sys
    sys.path.append("./")
    from src_v2.cliv2 import main
    import os
    # os.environ["WANDB_MODE"] = "disabled"
    # import torch.backends.cudnn as cudnn

    # print(cudnn)                   # e.g. <module 'torch.backends.cudnn' (built-in)>
    # print(type(cudnn))             # <class 'module'>
    # print(dir(cudnn))              # list of all attributes
    
    main()