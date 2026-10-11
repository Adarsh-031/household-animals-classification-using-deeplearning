import torch
import torch.nn.functional as F


def binary_gradcam(model, image_tensor, target_layer, target_class=None):
    """
    Grad-CAM for a binary model returning one logit per image.

    image_tensor: normalized tensor shaped [1, 3, H, W]
    target_class: 0=cat, 1=dog, None=predicted class
    """
    if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
        raise ValueError("Provide exactly one image shaped [1, 3, H, W].")

    if target_class not in (None, 0, 1):
        raise ValueError("target_class must be None, 0, or 1.")

    captured = {}

    def capture(module, inputs, output):
        captured["features"] = output

    handle = target_layer.register_forward_hook(capture)
    was_training = model.training
    model.eval()

    try:
        with torch.enable_grad():
            # Enables gradient tracking even when model features are frozen.
            inputs = image_tensor.detach().clone().requires_grad_(True)
            logits = model(inputs)

            if logits.shape != (1,):
                raise ValueError("Expected a single binary logit.")

            predicted_class = int(logits.detach().item() >= 0)
            explained_class = (
                predicted_class if target_class is None else target_class
            )

            # Positive logit supports dog; negative logit supports cat.
            score = logits[0] if explained_class == 1 else -logits[0]

            features = captured["features"]
            gradients = torch.autograd.grad(
                outputs=score,
                inputs=features,
            )[0]

            # Spatially average gradients to weight each feature channel.
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            if not torch.isfinite(cam).all().item():
                raise RuntimeError("Non-finite heatmap detected.")

            maximum = cam.max().item()
            has_positive_signal = maximum > 0

            if has_positive_signal:
                cam = cam / maximum
            else:
                cam = torch.zeros_like(cam)

            return {
                "heatmap": cam[0, 0].detach().cpu().numpy(),
                "dog_probability": torch.sigmoid(logits[0]).detach().item(),
                "predicted_class": predicted_class,
                "target_class": explained_class,
                "has_positive_signal": has_positive_signal,
            }

    finally:
        handle.remove()
        captured.clear()
        model.train(was_training)