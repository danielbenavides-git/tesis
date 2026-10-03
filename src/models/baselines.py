import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def raw_features(images):
    """Flatten each scalogram into one vector of pixels."""
    return images.reshape(len(images), -1)


def load_mobilenet(device):
    """MobileNetV2 with ImageNet weights, classifier removed, all weights frozen."""
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    return model.eval().to(device)


@torch.no_grad()
def mobilenet_features(images, device, batch_size=64, size=224, model=None):
    """1280 features per scalogram: repeat to 3 channels, resize, ImageNet normalization, global average pooling."""
    model = model if model is not None else load_mobilenet(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    for start in range(0, len(images), batch_size):
        x = torch.as_tensor(images[start:start + batch_size]).to(device)
        x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        out.append(model((x - mean) / std).cpu().numpy())
    return np.concatenate(out)


def pca_reduce(features, train_idx, n_components=16):
    """Standardize and project onto the first n_components principal axes, fitted on the training days only."""
    scaler = StandardScaler().fit(features[train_idx])
    pca = PCA(n_components=n_components, random_state=42).fit(scaler.transform(features[train_idx]))
    return pca.transform(scaler.transform(features)), pca.explained_variance_ratio_
