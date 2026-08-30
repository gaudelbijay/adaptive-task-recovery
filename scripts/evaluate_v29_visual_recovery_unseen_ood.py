#!/usr/bin/env python3
"""Run the immutable V28 unseen transforms with V29 accounting."""

import evaluate_visual_recovery_unseen_ood as unseen
from evaluate_v29_visual_recovery import install_protocol_adapter


if __name__ == "__main__":
    unseen.install_extensions()
    install_protocol_adapter()
    unseen.base.main()
