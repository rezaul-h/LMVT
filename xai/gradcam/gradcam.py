import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: PyTorch model
        target_layer: string, name of the convolutional layer to hook
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                break
        if not self.hook_handles:
            raise ValueError(f"Layer {self.target_layer} not found in model")

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: torch.Tensor of shape (1, C, H, W)
        target_class: int, class index to compute CAM for; if None, uses argmax
        """
        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward(retain_graph=True)
        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        # Normalize to [0,1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()
