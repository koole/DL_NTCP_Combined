"""
Same as DCNN, but where LeakyReLU activation has been applied whenever possible.
"""

import math
import torch
from .layers import conv3d_padding_same, Output, pooling_conv


class conv_block(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_size,
        strides,
        pad_value,
        lrelu_alpha,
        use_activation,
        use_bias=False,
    ):
        super(conv_block, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (
            len(kernel_size) == 3
        ):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)

        self.pad = conv3d_padding_same(
            depth=kernel_depth,
            height=kernel_height,
            width=kernel_width,
            pad_value=pad_value,
        )
        self.conv1 = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            bias=use_bias,
        )
        self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = torch.nn.Conv3d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=1,
            bias=use_bias,
        )
        self.norm2 = torch.nn.InstanceNorm3d(filters)
        self.use_activation = use_activation
        self.activation2 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)

        x = self.norm1(x)
        x = self.activation1(x)
        x = self.pad(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_activation:
            x = self.activation2(x)
        return x


class DCNN_LReLU(torch.nn.Module):
    """
    Deep CNN
    """

    def __init__(
        self,
        n_input_channels,
        depth,
        height,
        width,
        n_features,
        num_classes,
        filters,
        kernel_sizes,
        strides,
        pad_value,
        n_down_blocks,
        lrelu_alpha,
        dropout_p,
        pooling_conv_filters,
        perform_pooling,
        linear_units,
        clinical_variables_position,
        clinical_variables_linear_units,
        clinical_variables_dropout_p,
        use_bias=False,
    ):
        super(DCNN_LReLU, self).__init__()
        self.n_features = n_features
        self.pooling_conv_filters = pooling_conv_filters
        self.perform_pooling = perform_pooling
        self.clinical_variables_position = clinical_variables_position
        self.clinical_variables_linear_units = clinical_variables_linear_units

        # Determine Linear input channel size
        end_depth = math.ceil(depth / (2 ** n_down_blocks[0]))
        end_height = math.ceil(height / (2 ** n_down_blocks[1]))
        end_width = math.ceil(width / (2 ** n_down_blocks[2]))

        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        self.conv_blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):
            use_activation = True
            self.conv_blocks.add_module(
                "conv_block%s" % i,
                conv_block(
                    in_channels=in_channels[i],
                    filters=filters[i],
                    kernel_size=kernel_sizes[i],
                    strides=strides[i],
                    pad_value=pad_value,
                    lrelu_alpha=lrelu_alpha,
                    use_activation=use_activation,
                    use_bias=use_bias,
                ),
            )

        # Initialize pooling conv
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [end_depth, end_height, end_width]
            self.pool = pooling_conv(
                in_channels=filters[-1],
                filters=pooling_conv_filters,
                kernel_size=pooling_conv_kernel_size,
                strides=1,
                lrelu_alpha=lrelu_alpha,
                use_bias=use_bias,
            )
            end_depth, end_height, end_width = 1, 1, 1
            filters[-1] = self.pooling_conv_filters
        elif self.perform_pooling:
            # self.pool = torch.nn.AvgPool3d(kernel_size=(1, end_height, end_width))
            # end_depth, end_height, end_width = depth, 1, 1
            # self.pool = torch.nn.AvgPool3d(kernel_size=(end_depth, end_height, end_width))
            self.pool = torch.nn.MaxPool3d(
                kernel_size=(end_depth, end_height, end_width)
            )
            end_depth, end_height, end_width = 1, 1, 1

        # Initialize flatten layer
        self.flatten = torch.nn.Flatten()

        # Initialize MLP layers for clinical variables
        if (self.n_features > 0) and (clinical_variables_linear_units is not None):
            self.clinical_variables_layers = torch.nn.ModuleList()
            clinical_variables_linear_units = [
                n_features
            ] + clinical_variables_linear_units
            for i in range(len(clinical_variables_linear_units) - 1):
                self.clinical_variables_layers.add_module(
                    "dropout%s" % i, torch.nn.Dropout(clinical_variables_dropout_p[i])
                )
                self.clinical_variables_layers.add_module(
                    "linear%s" % i,
                    torch.nn.Linear(
                        in_features=clinical_variables_linear_units[i],
                        out_features=clinical_variables_linear_units[i + 1],
                        bias=use_bias,
                    ),
                )
                self.clinical_variables_layers.add_module(
                    "lrelu%s" % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
                )
        else:
            clinical_variables_linear_units = [n_features]

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        linear_units = [end_depth * end_height * end_width * filters[-1]] + linear_units
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module(
                "dropout%s" % i, torch.nn.Dropout(dropout_p[i])
            )
            self.linear_layers.add_module(
                "linear%s" % i,
                torch.nn.Linear(
                    in_features=linear_units[i],
                    out_features=linear_units[i + 1],
                    bias=use_bias,
                ),
            )
            self.linear_layers.add_module(
                "lrelu%s" % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
            )
        self.n_sublayers_per_linear_layer = len(self.linear_layers) / (
            len(linear_units) - 1
        )

        # Initialize output layer
        if (self.n_features > 0) and (clinical_variables_linear_units is not None):
            # Add additional input units to the output layer if MLP output is concatenated at the last linear layer
            if self.clinical_variables_position + 1 == len(linear_units) - 1:
                self.out_layer = Output(
                    in_features=linear_units[-1] + clinical_variables_linear_units[-1],
                    out_features=num_classes,
                    bias=use_bias,
                )
            else:
                self.out_layer = Output(
                    in_features=linear_units[-1],
                    out_features=num_classes,
                    bias=use_bias,
                )
        else:
            self.out_layer = Output(
                in_features=linear_units[-1] + self.n_features,
                out_features=num_classes,
                bias=use_bias,
            )
        # self.out_layer.__class__.__name__ = 'Output'

    def forward(self, x, features):
        # Blocks
        for block in self.conv_blocks:
            x = block(x)

        # Pooling layers
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)

        x = self.flatten(x)

        # MLP clinical variables
        if (self.n_features > 0) and (self.clinical_variables_linear_units is not None):
            for layer in self.clinical_variables_layers:
                features = layer(features)

        # Linear layers
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if (self.n_features > 0) and (
                (i + 1) / self.n_sublayers_per_linear_layer
                == self.clinical_variables_position + 1
            ):
                # Add features
                x = torch.cat([x, features], dim=1)

        # Output
        x = self.out_layer(x)

        return x
