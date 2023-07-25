import torch
import numpy as np
from PIL import Image


class cWCT(torch.nn.Module):
    def __init__(self, eps=2e-5, use_double=False):
        super().__init__()
        self.eps = eps
        self.use_double = use_double

    def transfer(self, cont_feat, styl_feat, cmask=None, smask=None):
        if cmask is None or smask is None:
            return self._transfer(cont_feat, styl_feat)
        else:
            return self._transfer_seg(cont_feat, styl_feat, cmask, smask)

    def _transfer(self, cont_feat, styl_feat):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()
            styl_feat = styl_feat.double()

        # whitening and coloring transforms
        whiten_fea = self.whitening(cont_feat)
        color_fea = self.coloring(whiten_fea, styl_feat)

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

    def _transfer_seg(self, cont_feat, styl_feat, cmask, smask):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :param cmask: numpy [B, _, _]
        :param smask: numpy [B, _, _]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        _, _, sH, sW = styl_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()
            styl_feat = styl_feat.double()

        for i in range(B):
            label_set, label_indicator = self.compute_label_info(cmask[i], smask[i])
            resized_content_segment = self.resize(cmask[i], cH, cW)
            resized_style_segment = self.resize(smask[i], sH, sW)

            single_content_feat = cont_feat[i]     # [N, cH*cW]
            single_style_feat = styl_feat[i]   # [N, sH*sW]
            target_feature = single_content_feat.clone()   # [N, cH*cW]

            for label in label_set:
                if not label_indicator[label]:
                    continue

                content_index = self.get_index(resized_content_segment, label).to(single_content_feat.device)
                style_index = self.get_index(resized_style_segment, label).to(single_style_feat.device)
                if content_index is None or style_index is None:
                    continue

                masked_content_feat = torch.index_select(single_content_feat, 1, content_index)
                masked_style_feat = torch.index_select(single_style_feat, 1, style_index)
                whiten_fea = self.whitening(masked_content_feat)
                _target_feature = self.coloring(whiten_fea, masked_style_feat)

                new_target_feature = torch.transpose(target_feature, 1, 0)
                new_target_feature.index_copy_(0, content_index,
                                               torch.transpose(_target_feature, 1, 0))
                target_feature = torch.transpose(new_target_feature, 1, 0)

            cont_feat[i] = target_feature
        color_fea = cont_feat

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

    def cholesky_dec(self, conv, invert=False):
        cholesky = torch.linalg.cholesky if torch.__version__ >= '1.8.0' else torch.cholesky
        try:
            L = cholesky(conv)
        except RuntimeError:
            # print("Warning: Cholesky Decomposition fails")
            iden = torch.eye(conv.shape[-1]).to(conv.device)
            eps = self.eps
            while True:
                try:
                    conv = conv + iden * eps
                    L = cholesky(conv)
                    break
                except RuntimeError:
                    eps = eps+self.eps

        if invert:
            L = torch.inverse(L)

        return L.to(conv.dtype)

    def whitening(self, x):
        mean = torch.mean(x, -1)
        mean = mean.unsqueeze(-1).expand_as(x)
        x = x - mean

        conv = (x @ x.transpose(-1, -2)).div(x.shape[-1] - 1)
        inv_L = self.cholesky_dec(conv, invert=True)

        whiten_x = inv_L @ x

        return whiten_x

    def coloring(self, whiten_xc, xs):
        xs_mean = torch.mean(xs, -1)
        xs = xs - xs_mean.unsqueeze(-1).expand_as(xs)

        conv = (xs @ xs.transpose(-1, -2)).div(xs.shape[-1] - 1)
        Ls = self.cholesky_dec(conv, invert=False)

        coloring_cs = Ls @ whiten_xc
        coloring_cs = coloring_cs + xs_mean.unsqueeze(-1).expand_as(coloring_cs)

        return coloring_cs

    def compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size is False or styl_seg.size is False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
        return self.label_set, self.label_indicator

    def resize(self, img, H, W):
        size = (W, H)
        if len(img.shape) == 2:
            return np.array(Image.fromarray(img).resize(size, Image.NEAREST))
        else:
            return np.array(Image.fromarray(img, mode='RGB').resize(size, Image.NEAREST))

    def get_index(self, feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0])

    def interpolation(self, cont_feat, styl_feat_list, alpha_s_list, alpha_c=0.0):
        """
        :param cont_feat: Tensor [B, N, cH, cW]
        :param styl_feat_list: List [Tensor [B, N, _, _], Tensor [B, N, _, _], ...]
        :param alpha_s_list: List [float, float, ...]
        :param alpha_c: float
        :return color_fea: Tensor [B, N, cH, cW]
        """
        assert len(styl_feat_list) == len(alpha_s_list)

        B, N, cH, cW = cont_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()

        c_mean = torch.mean(cont_feat, -1)
        cont_feat = cont_feat - c_mean.unsqueeze(-1).expand_as(cont_feat)

        cont_conv = (cont_feat @ cont_feat.transpose(-1, -2)).div(cont_feat.shape[-1] - 1)  # interpolate Conv works well
        inv_Lc = self.cholesky_dec(cont_conv, invert=True)  # interpolate L seems to be slightly better

        whiten_c = inv_Lc @ cont_feat

        # First interpolate between style_A, style_B, style_C, ...
        mix_Ls = torch.zeros_like(inv_Lc)   # [B, N, N]
        mix_s_mean = torch.zeros_like(c_mean)   # [B, N]
        for styl_feat, alpha_s in zip(styl_feat_list, alpha_s_list):
            assert styl_feat.shape[0] == B and styl_feat.shape[1] == N
            styl_feat = styl_feat.reshape(B, N, -1)

            if self.use_double:
                styl_feat = styl_feat.double()

            s_mean = torch.mean(styl_feat, -1)
            styl_feat = styl_feat - s_mean.unsqueeze(-1).expand_as(styl_feat)

            styl_conv = (styl_feat @ styl_feat.transpose(-1, -2)).div(styl_feat.shape[-1] - 1)  # interpolate Conv works well
            Ls = self.cholesky_dec(styl_conv, invert=False)  # interpolate L seems to be slightly better

            mix_Ls += Ls * alpha_s
            mix_s_mean += s_mean * alpha_s

        # Second interpolate between content and style_mix
        if alpha_c != 0.0:
            Lc = self.cholesky_dec(cont_conv, invert=False)
            mix_Ls = mix_Ls * (1-alpha_c) + Lc * alpha_c
            mix_s_mean = mix_s_mean * (1-alpha_c) + c_mean * alpha_c

        color_fea = mix_Ls @ whiten_c
        color_fea = color_fea + mix_s_mean.unsqueeze(-1).expand_as(color_fea)

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)


if __name__ == '__main__':
    # transfer
    c = torch.rand((2, 16, 512, 256))
    s = torch.rand((2, 16, 64, 128))

    cwct = cWCT(use_double=True)
    cs = cwct.transfer(c, s)
    print(cs.shape)


    # interpolation
    c = torch.rand((1, 16, 512, 256))
    s_list = [torch.rand((1, 16, 64, 128)) for _ in range(4)]
    alpha_s_list = [0.25 for _ in range(4)]     # interpolate between style_A, style_B, style_C, ...
    alpha_c = 0.5   # interpolate between content and style_mix if alpha_c!=0.0

    cwct = cWCT(use_double=True)
    cs = cwct.interpolation(c, s_list, alpha_s_list, alpha_c)
    print(cs.shape)
