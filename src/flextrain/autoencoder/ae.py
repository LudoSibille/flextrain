from typing import Any

from torch import nn


class AutoEncoderType(nn.Module):
    def forward(self, x: Any) -> Any: ...

    def encode(self, x: Any) -> Any: ...

    def decode(self, x: Any) -> Any: ...


class AutoEncoder(AutoEncoderType):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.encoder(*args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self.encoder(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.decoder(*args, **kwargs)
