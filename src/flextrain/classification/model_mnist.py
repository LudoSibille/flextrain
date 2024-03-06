from torch import nn


def create_model(nb_classes: int=10, dropout: float = 0.5, num_features: int =32, nb_linear: int = 256) -> nn.Module:
    nb_convs = 2
    image_size = 28
    nb_dense = (num_features * 2) * (int(image_size / (2 ** nb_convs)) ** 2)

    model = nn.Sequential(
        nn.Conv2d(1, num_features, 3, padding=1),
        nn.Conv2d(num_features, num_features, 3, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        nn.InstanceNorm2d(num_features),

        nn.Conv2d(num_features, num_features * 2, 3, padding=1),
        nn.Conv2d(num_features * 2, num_features * 2, 3, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        nn.InstanceNorm2d(num_features * 2),

        nn.Flatten(),

        nn.Linear(nb_dense, nb_linear),
        nn.Dropout(dropout),
        nn.LeakyReLU(),
        nn.InstanceNorm1d(nb_linear),
        
        nn.Linear(nb_linear, nb_linear // 2),
        nn.Dropout(dropout),
        nn.LeakyReLU(),
        nn.InstanceNorm1d(nb_linear // 2),
        nn.Linear(nb_linear // 2, nb_classes),
    )
    return model