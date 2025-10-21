"""
Reasoning-Enhanced Object-Centric Learning for Videos
https://github.com/intell-sci-comput/STATM

SlotPi: Physics-informed Object-centric Reasoning Models
https://github.com/intell-sci-comput/SlotPi

Object-Centric Video Prediction via Decoupling of Object Dynamics and Interactions
https://github.com/hanoonaR/object-centric-ovd
"""

from einops import rearrange
import torch as pt
import torch.nn as nn


class RandSFQ(nn.Module):
    """
    RandSF-Q
    ---
    Learning Video Slot Attention Query from Random Slot-Feature Pairs.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        transit,  # type: RSFQTransit
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.transit = transit
        self.decode = decode
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c)
        """
        b, t, c0, h0, w0 = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        slotz = None
        attent = []

        for i in range(t):
            if i == 0:
                query_i = self.initializ(
                    b if condit is None else condit[:, 0, :, :]
                )  # (b,n,c)
            else:
                # slotz: [0,i); encode: [0,i]
                query_i = self.transit(slotz, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            slotz_i, attent_i = self.aggregat(
                encode[:, i, :, :], query_i, num_iter=niter
            )

            slotz = (  # (b,i+1,n,c)
                slotz_i[:, None, :, :]
                if slotz is None
                else pt.concat([slotz, slotz_i[:, None, :, :]], 1)
            )
            attent.append(attent_i)  # t*(b,n,h*w)
            # TODO XXX add latent act_token

        attent = pt.stack(attent, 1)  # (b,t,n,h*w)
        attent = rearrange(attent, "b t n (h w) -> b t n h w", h=h)

        clue = rearrange(feature, "b t c h w -> (b t) (h w) c")
        recon, attent2 = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attent2 = rearrange(attent2, "(b t) n (h w) -> b t n h w", b=b, h=h)

        return feature, slotz, attent, attent2, recon


class RSFQTransit(nn.Module):

    def __init__(
        self, dt, ci, c, nhead=4, expanz=4, pdo=0.5, norm_first=False, bias=False
    ):
        super().__init__()
        self.dt = dt
        self.te = nn.Embedding(dt, c)
        self.proji = nn.Linear(ci, c)
        self.transit = nn.TransformerDecoderLayer(
            d_model=c,
            nhead=nhead,
            dim_feedforward=c * expanz,
            dropout=pdo,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
        )

    def forward(self, slotzs0, encodes0):
        """
        slotzs: all past step slots, shape=(b,t=i,n,c)
        encodes: all past and current step frame features, shape=(b,i+1,h*w,c)
        """
        # slotzs = slotzs[:, -self.dt :, :, :]  # window size <=dt
        # encodes = encodes[:, -(self.dt + 1) :, :, :]  # window size <=dt+1
        # NOTE This assumes training window size == dt, otherwise there will be error in self.te(ts) !!!
        #   Thus ``slotzs.size(1) always< self.dt`` and ``encodes.size(1) always<= self.dt`` !!!
        # Below is the corrected implementation. TODO Not sure if equivalent during training.
        slotzs = slotzs0[:, -self.dt + 1 :, :, :]  # window size <=dt
        encodes = encodes0[:, -self.dt :, :, :]  # window size <=dt+1

        b, i, n, c = slotzs.shape
        assert i + 1 == encodes.size(1)
        device = slotzs.device
        bidx = pt.arange(b, dtype=pt.int64, device=device)

        if self.training and i > 1:
            ts = pt.randint(0, i, [b], dtype=pt.int64, device=device)
            slotz = slotzs[bidx, ts, :, :]
            dts = i - ts
            # ensuring ts<te: always bad
            te = pt.randint(1, i + 1, [b], dtype=pt.int64, device=device)
            encode = encodes[bidx, te, :, :]
            dte = i - te
        else:
            slotz = slotzs[:, -1, :, :]
            dts = pt.ones(b, dtype=pt.int64, device=device)
            encode = encodes[:, -1, :, :]
            dte = pt.zeros(b, dtype=pt.int64, device=device)

        tes = self.te(dts)[:, None, :]
        slotz = slotz.detach() + tes
        tee = self.te(dte)[:, None, :]
        encode = self.proji(encode.detach()) + tee

        query = self.transit(slotz, encode)
        return query
