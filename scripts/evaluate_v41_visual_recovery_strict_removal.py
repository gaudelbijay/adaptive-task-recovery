#!/usr/bin/env python3
"""Run strict removal with V41's non-PPO accounting adapter."""

from evaluate_v41_visual_recovery import install_protocol_adapter
import evaluate_visual_recovery_strict_removal as strict


if __name__ == "__main__":
    install_protocol_adapter()
    strict.main()
