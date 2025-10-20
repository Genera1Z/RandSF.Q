from einops import rearrange, repeat
import torch as pt
import torch.nn.functional as ptnf

from .metric import Metric


class FeatureTimeSimilarity:  # more like a Transform rather than a Metric

    def __init__(self, time_shift=1, thresh=None, tau=1.0, softmax=True):
        self.time_shift = time_shift
        self.thresh = thresh
        self.tau = tau
        self.softmax = softmax

    def __call__(self, feature):
        """
        - feature: shape=(b,t,n,c)
        """
        assert feature.ndim == 4
        b, t, n, c = feature.shape

        feature = ptnf.normalize(feature, 2, -1)
        source = feature[:, : -self.time_shift, :, :]
        target = feature[:, self.time_shift :, :, :]

        similarity = __class__.cross_similarity(
            source.flatten(0, 1),
            target.flatten(0, 1),
            self.thresh,
            self.tau,
            self.softmax,
        ).unflatten(0, [b, t - self.time_shift])

        return similarity

    @staticmethod
    @pt.no_grad()
    def cross_similarity(source, target, thresh=None, tau=1.0, softmax=True):
        """
        - source: shape=(b,m,c)
        - target: shape=(b,n,c)
        """
        product = pt.einsum("bmc,bnc->bmn", source, target)
        b, m, n = product.shape
        if thresh is not None:
            product[product < thresh] = -pt.inf
        product /= tau
        if softmax:
            flag = product.isinf().all(-1, keepdim=True).expand(-1, -1, n)  # -inf
            product[flag] = 0.0
            product = product.softmax(-1)
        return product


class SlotContrastLoss(Metric):
    """Temporally Consistent Object-Centric Learning by Contrasting Slots"""

    def __init__(self, tau=0.1, mean=()):
        super().__init__(mean)
        self.tau = tau

    def forward(self, input, shift=1):
        """
        - input: slots, shape=(b,t,s,c)
        """
        b, t, s, c = input.shape
        dtype = input.dtype
        device = input.device

        slots = ptnf.normalize(input, p=2.0, dim=-1)
        slots = rearrange(slots, "b t s c -> t (s b) c")

        s1 = slots[:-shift, :, :]
        s2 = slots[shift:, :, :]
        ss = pt.einsum("tmc,tnc->tmn", s1, s2) / self.tau  # (t,s*b,s*b)
        eye = pt.eye(s * b, dtype=dtype, device=device)
        eye = repeat(eye, "m n -> t m n", t=t - shift)
        # loss = ptnf.cross_entropy(ss, eye, reduction="none")  # (b,..)
        loss = ptnf.cross_entropy(ss, eye)[None]  # (b=1,)
        return self.finaliz(loss)  # (b,..) (b,)
