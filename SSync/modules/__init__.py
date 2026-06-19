from SSync.modules import timm
from SSync.modules.decoders import build as build_decoder
from SSync.modules.encoders import build as build_encoder
from SSync.modules.groupers import build as build_grouper
from SSync.modules.initializers import build as build_initializer
from SSync.modules.networks import build as build_network
from SSync.modules.utils import Resizer, SoftToHardMask
from SSync.modules.utils import build as build_utils
from SSync.modules.utils import build_module, build_torch_function, build_torch_module
from SSync.modules.video import LatentProcessor, MapOverTime, ScanOverTime
from SSync.modules.video import build as build_video

__all__ = [
    "build_decoder",
    "build_encoder",
    "build_grouper",
    "build_initializer",
    "build_network",
    "build_utils",
    "build_module",
    "build_torch_module",
    "build_torch_function",
    "timm",
    "MapOverTime",
    "ScanOverTime",
    "LatentProcessor",
    "Resizer",
    "SoftToHardMask",
]


BUILD_FNS_BY_MODULE_GROUP = {
    "decoders": build_decoder,
    "encoders": build_encoder,
    "groupers": build_grouper,
    "initializers": build_initializer,
    "networks": build_network,
    "utils": build_utils,
    "video": build_video,
    "torch": build_torch_function,
    "torch.nn": build_torch_module,
    "nn": build_torch_module,
}
