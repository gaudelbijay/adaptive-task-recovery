"""V52 plus confidence-gated experts for the opened seed-127M renderers."""

from copy import deepcopy

import torch
import torch.nn as nn

from v52_subpixel_specialist_agent import SubpixelSpecialistAgent


class OpenedRendererExpertAgent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(); self.base = SubpixelSpecialistAgent(*args, **kwargs)
        self.experts = nn.ModuleList([deepcopy(self.base.v51.v47.v41.base.encoder) for _ in range(4)])
        self.router = nn.Sequential(nn.Conv2d(3,24,5,stride=2,padding=2),nn.SiLU(),nn.Conv2d(24,48,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(48,96,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(96,5))
        self.actor=self.base.actor; self.goal_progress_predictor=self.base.goal_progress_predictor; self.route_confidence=0.9

    def initialize_from_v52(self,state):
        self.base.load_state_dict(state,strict=True); baseline=self.base.v51.v47.v41.base.encoder.state_dict()
        for expert in self.experts: expert.load_state_dict(baseline,strict=True)
        for parameter in self.base.parameters(): parameter.requires_grad_(False)

    def router_logits(self,rgb): return self.router(rgb.permute(0,3,1,2).float().div(255.0))
    def expert_latent(self,index,rgb): return self.experts[index](rgb.permute(0,3,1,2).float().div(255.0))

    def encode(self,rgb,augment=False):
        if augment: raise ValueError("V53 deployment does not use stochastic augmentation")
        result=self.base.encode(rgb); probabilities=torch.softmax(self.router_logits(rgb),dim=1); confidence,choice=probabilities.max(dim=1)
        for label in range(1,5):
            route=(choice==label)&(confidence>=self.route_confidence)
            if bool(route.any()): result=torch.where(route[:,None],self.expert_latent(label-1,rgb),result)
        return result
