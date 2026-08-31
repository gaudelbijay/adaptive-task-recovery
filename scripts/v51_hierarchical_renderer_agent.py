"""V47 top-level routing with V50 lighting experts used only inside lighting."""

import torch

from v50_renderer_expert_agent import DedicatedRendererExpertAgent


class HierarchicalRendererExpertAgent(DedicatedRendererExpertAgent):
    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V51 deployment does not use stochastic augmentation")
        top_choice = self.v47.router_logits(rgb).argmax(dim=1)
        result = self.v47.v41.encode(rgb)
        if bool((top_choice == 1).any()):
            result = torch.where(
                (top_choice == 1)[:, None], self.v47.camera_latent(rgb), result
            )
        lighting = top_choice == 2
        if bool(lighting.any()):
            subtype = self.router_logits(rgb)[:, 2:4].argmax(dim=1)
            bright = lighting & (subtype == 0)
            green = lighting & (subtype == 1)
            if bool(bright.any()):
                result = torch.where(bright[:, None], self.bright_latent(rgb), result)
            if bool(green.any()):
                result = torch.where(green[:, None], self.green_latent(rgb), result)
        return result
