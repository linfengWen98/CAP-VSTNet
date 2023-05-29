import torch
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse


def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def compute_laplacian(img, mask=None, eps=10 ** (-7), win_rad=1):
    """Computes Matting Laplacian for a given image.
    Args:
        img: 3-dim Image or str
        mask: mask of pixels for which Laplacian will be computed.
            If not set Laplacian will be computed for all pixels.
        eps: regularization parameter controlling alpha smoothness
            from Eq. 12 of the original paper. Defaults to 1e-7.
        win_rad: radius of window used to build Matting Laplacian (i.e.
            radius of omega_k in Eq. 12).
    Returns: sparse matrix holding Matting Laplacian.
    """

    if type(img) == str:
        img = cv2.imread(img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(img)

    # img must be in [0, 1]
    img = 1.0 * img / 255.0

    h, w, _ = img.shape

    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    if mask is not None:
        mask = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((win_diam, win_diam), np.uint8)
        ).astype(np.bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)

    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    # vals = np.eye(win_size) - (1.0 / win_size) * (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))
    vals = (1.0 / win_size) * (1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w))

    sum_L = L.sum(axis=1).T.tolist()[0]
    L = scipy.sparse.diags([sum_L], [0], shape=(h * w, h * w)) - L

    L = L.tocoo().astype(np.float32)
    return L


def laplacian_loss_grad(image, M):
    laplacian_m = M
    img = image

    channel, height, width = img.size()
    loss = 0
    grads = list()
    for i in range(channel):
        grad = torch.mm(laplacian_m, img[i, :, :].reshape(-1, 1))/(height*width)
        loss += torch.mm(img[i, :, :].reshape(1, -1), grad)
        grads.append(grad.reshape((height, width)))
    gradient = torch.stack(grads, dim=0)
    return loss, 2.0*gradient
