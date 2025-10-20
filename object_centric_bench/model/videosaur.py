from einops import rearrange
import torch as pt
import torch.nn as nn

from . import DINOSAUR


class VideoSAUR(DINOSAUR):
    """
    Zadaianchuk et al. Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities. NeurIPS 2023.

    Different from DINOSAUR in the decoder, i.e., SlotMixerDecoder, and the loss, i.e., extra FeatureSimilarityLoss.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        decode,
        transit,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            decode,
        )
        self.transit = transit
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
        )

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c)
        """
        b, t, c, h, w = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        query = self.initializ(b if condit is None else condit[:, 0, :, :])  # (b,n,c)
        slotz = []
        attent = []
        for i in range(t):
            slotz_i, attent_i = self.aggregat(encode[:, i, :, :], query)
            query = self.transit(slotz_i)
            slotz.append(slotz_i)  # [(b,n,c),..]
            attent.append(attent_i)  # [(b,n,h*w),..]
        slotz = pt.stack(slotz, 1)  # (b,t,n,c)
        attent = pt.stack(attent, 1)  # (b,t,n,h*w)
        attent = rearrange(attent, "b t n (h w) -> b t n h w", h=h)

        clue = [h, w]
        recon, attent2 = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attent2 = rearrange(attent2, "(b t) n (h w) -> b t n h w", b=b, h=h)

        return feature, slotz, attent, attent2, recon

    # segment acc: attent > attent2


class SlotMixerDecoder(nn.Module):
    """http://arxiv.org/abs/2206.06922"""

    def __init__(self, embed_dim, posit_embed, allocat, attent, render):
        super().__init__()
        self.posit_embed = posit_embed  # 1d
        self.norm_m = nn.LayerNorm(embed_dim, eps=1e-5)

        assert isinstance(allocat, nn.TransformerDecoder)
        for tfdb in allocat.layers:
            assert isinstance(tfdb, nn.TransformerDecoderLayer)
            tfdb.self_attn = nn.Identity()
            tfdb.self_attn.batch_first = True  # for compatiblity
            tfdb.dropout1 = nn.Identity()
            tfdb._sa_block = lambda *a, **k: a[0]
        self.allocat = allocat  # Tfd

        self.norm_q = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm_k = nn.LayerNorm(embed_dim, eps=1e-5)

        assert isinstance(attent, nn.MultiheadAttention)
        if attent._qkv_same_embed_dim:
            chunks = attent.in_proj_weight.chunk(3, 0)
            attent.q_proj_weight = nn.Parameter(chunks[0])
            attent.k_proj_weight = nn.Parameter(chunks[1])
            attent.in_proj_weight = None
            attent._qkv_same_embed_dim = False
        del attent.v_proj_weight, attent.out_proj.weight
        attent.register_buffer(
            "v_proj_weight", pt.eye(attent.embed_dim, dtype=pt.float), persistent=False
        )
        attent.out_proj.register_buffer(
            "weight",
            pt.eye(attent.out_proj.in_features, dtype=pt.float),
            persistent=False,
        )
        self.attent = attent  # mha

        self.render = render  # MLP

    def forward(self, input, slotz):
        """
        input: destructed target, height and width
        slotz: slots, shape=(b,n,c)
        """
        h, w = input
        b, n, c = slotz.shape
        x = pt.zeros([b, h * w, c], dtype=slotz.dtype, device=slotz.device)

        query, pe = self.posit_embed(x, True)  # bmc
        memory = self.norm_m(slotz)  # bnc
        query = self.allocat(query, memory=memory)  # bmc

        q = self.norm_q(query)  # bmc
        k = self.norm_k(slotz)  # bnc
        v = slotz  # bnc
        slotmix, attent = self.attent(q, k, v)  # bmc bmn
        slotmix = slotmix + pe  # bmc
        recon = self.render(slotmix)  # bmc

        attent = attent.permute(0, 2, 1)  # bnm  # to match outside rearrange
        return recon, attent
