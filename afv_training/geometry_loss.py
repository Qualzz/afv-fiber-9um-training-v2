"""Align the local H/V probability-field structure with observed spline tangents."""

import torch
import torch.nn.functional as F

PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
EPSILON = 1e-6
COHERENCE = 0.95
MARGIN = 11


def gaussian(x, sigma):
    """Separable grouped convolution; bounded to the128-cubed training patch."""
    radius = int(4 * sigma + 0.5)
    q = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (q / sigma).square())
    kernel /= kernel.sum()
    channels = x.shape[1]
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = len(kernel)
        weight = kernel.reshape(1, 1, *shape).expand(channels, 1, *shape)
        padding = [0] * 6
        padding[2 * (2 - axis)] = radius
        padding[2 * (2 - axis) + 1] = radius
        x = F.conv3d(F.pad(x, padding, mode="reflect"), weight, groups=channels)
    return x


def tensors(probability):
    smooth = gaussian(F.avg_pool3d(probability.float(), 2), 0.9)
    gradients = torch.gradient(smooth, dim=(2, 3, 4))  # z,y,x
    components = torch.stack([gradients[i] * gradients[j] for i, j in PAIRS], 2)
    n, f, c, d, h, w = components.shape
    return gaussian(components.reshape(n, f * c, d, h, w), 1.5).reshape(
        n, f, c, d, h, w
    )


@torch.no_grad()
def targets(batch, margin=MARGIN):
    """Axial second moments tolerate tangent sign; reject mixed coarse tangents."""
    direction = F.normalize(batch["direction"].float(), dim=1, eps=1e-6).flip(1)
    second = torch.stack([direction[:, i] * direction[:, j] for i, j in PAIRS], 1)
    all_targets = []
    all_valid = []
    for family in (1, 2):
        known = (
            (batch["centre"] > 0)
            & (batch["direction_valid"] > 0)
            & (batch["semantic"] == family)
        )
        count = F.avg_pool3d(known[:, None].float(), 2)
        moment = F.avg_pool3d(second * known[:, None], 2) / count.clamp_min(1e-12)
        # tr(S^2) close to1 iff axial annotations agree. No eigenvector gradients.
        coherence = moment[:, :3].square().sum(1) + 2 * moment[:, 3:].square().sum(1)
        valid = (count[:, 0] > 0) & (coherence >= COHERENCE)
        if margin:
            for axis in range(1, 4):
                sl = [slice(None)] * 4
                sl[axis] = slice(0, margin)
                valid[tuple(sl)] = False
                sl[axis] = slice(-margin, None)
                valid[tuple(sl)] = False
        all_targets.append(moment)
        all_valid.append(valid)
    return torch.stack(all_targets, 1), torch.stack(all_valid, 1)


def energy_loss(tensor, moment, valid, weights, epsilon=EPSILON):
    terms = []
    used = []
    for i in range(len(tensor)):
        if valid[i].any():
            components = tensor[i].movedim(1, -1)[valid[i]]
            target = moment[i].movedim(1, -1)[valid[i]]

            def matrix(a):
                return torch.stack(
                    (
                        a[:, 0],
                        a[:, 3],
                        a[:, 4],
                        a[:, 3],
                        a[:, 1],
                        a[:, 5],
                        a[:, 4],
                        a[:, 5],
                        a[:, 2],
                    ),
                    1,
                ).reshape(-1, 3, 3)

            J = matrix(components)
            with torch.no_grad():
                tangent = torch.linalg.eigh(matrix(target))[1][:, :, -1]
                basis = F.one_hot(tangent.abs().argmin(1), 3).to(tangent.dtype)
                u = F.normalize(torch.cross(tangent, basis, dim=1), dim=1)
                v = torch.cross(tangent, u, dim=1)

            def energy(a, b):
                return torch.einsum("ni,nij,nj->n", a, J, b)

            parallel = energy(tangent, tangent).clamp_min(0)
            aa = energy(u, u)
            bb = energy(v, v)
            ab = energy(u, v)
            transverse_min = (
                0.5
                * (aa + bb - torch.sqrt((aa - bb).square() + 4 * ab.square() + 1e-16))
            ).clamp_min(0)
            error = (parallel + epsilon / 3) / (parallel + 2 * transverse_min + epsilon)
            terms.append(error.mean())
            used.append(weights[i])
    if not terms:
        return tensor.sum() * 0
    w = torch.as_tensor(used, device=tensor.device, dtype=tensor.dtype)
    return (torch.stack(terms) * w).sum() / w.sum()


def loss(logits, batch, weights, prepared=None):
    moment, valid = targets(batch) if prepared is None else prepared
    probability = logits.float().softmax(1)[:, 1:3]
    return energy_loss(tensors(probability), moment, valid, weights)
