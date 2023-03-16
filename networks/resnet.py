from torch import nn


def first_non_one_kernel(kernels):  # Inspired by https://arxiv.org/pdf/1812.01187.pdf
    for i, kernel in enumerate(kernels):
        if kernel > 1:
            return i
    return 0  # Return 0 by default (if all kernel sizes are 1)


class GlobalAvgPool2d(nn.Module):

    def __init__(self, keepdim=False):
        super().__init__()

        self.keepdim = keepdim

    def forward(self, x):  # Take average of last two dimensions (H, W)
        x = x.mean(dim=(-2, -1), keepdim=self.keepdim)
        return x


class Residual(nn.Module):

    def __init__(self, in_channels, filters, kernels, downsample=False):
        super().__init__()

        assert len(filters) == len(kernels), \
            f"Filters length {len(filters)} must match kernels length {len(kernels)}"
        for kernel in kernels:  # Kernel sizes must be all odd (otherwise padding is annoying)
            assert kernel % 2 == 1, f"Kernel size {kernel} is not odd"

        layers = []
        layers_skip = [nn.Identity()]  # Placeholder for no downsample

        downsample_i = first_non_one_kernel(kernels)
        for i in range(len(filters)):
            stride = 2 if downsample and i == downsample_i else 1  # downsample at first Conv2d
            layers += [
                nn.Conv2d(in_channels=in_channels if i == 0 else filters[i - 1],
                          out_channels=filters[i], kernel_size=kernels[i], stride=stride,
                          padding=kernels[i] // 2),
                nn.ReLU(),  # V2 architecture, so full pre-activation
                nn.BatchNorm2d(num_features=filters[i]),
            ]

        if downsample:
            layers_skip.append(nn.AvgPool2d(kernel_size=2, stride=2))
        if downsample or in_channels != filters[-1]:
            layers_skip += [nn.Conv2d(in_channels=in_channels, out_channels=filters[-1],
                                      kernel_size=1, stride=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(num_features=filters[-1])]

        self.layers = nn.Sequential(*layers)  # The nn.Sequential API is amazing
        self.layers_skip = nn.Sequential(*layers_skip)  # I'll fight anyone who says otherwise

    def forward(self, x):
        h = self.layers(x)
        x = self.layers_skip(x)
        x = x + h
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, filters, kernels, repeat, downsample=False):
        super().__init__()

        layers = []
        for r in range(repeat):  # Stack repeat number of Residual modules
            layers.append(Residual(in_channels=in_channels if r == 0 else filters[-1],
                                   filters=filters, kernels=kernels,
                                   downsample=downsample and r == 0))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_shape, num_classes, filters, kernels, repeats,
                 in_kernel=5, in_stride=2, in_max_pool_kernel=1, in_max_pool_stride=1):
        super().__init__()

        # Initial convolution (+ max pooling, potentially)
        assert in_kernel % 2 == 1, f"Input kernel size {in_kernel} must be odd"
        layers = [nn.Conv2d(in_channels=in_shape[0], out_channels=filters[0][0],
                            kernel_size=in_kernel, stride=in_stride, padding=in_kernel // 2),
                  nn.ReLU(),
                  nn.BatchNorm2d(num_features=filters[0][0])]
        if in_max_pool_kernel > 1 or in_max_pool_stride > 1:
            layers.append(nn.MaxPool2d(kernel_size=in_max_pool_kernel, stride=in_max_pool_stride,
                                       padding=(in_max_pool_kernel - 1) // 2))
        # ResNet blocks
        for i in range(len(filters)):
            layers.append(ResidualBlock(in_channels=filters[0][0] if i == 0 else filters[i - 1][-1],
                                        filters=filters[i], kernels=kernels[i], repeat=repeats[i],
                                        downsample=i > 0))
        # Global average pooling and final fully-connected layer
        layers += [GlobalAvgPool2d(),
                   nn.Linear(in_features=filters[-1][-1], out_features=num_classes)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
