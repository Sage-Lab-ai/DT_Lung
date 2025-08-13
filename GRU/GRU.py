import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, num_dynamic_feature, output_sequence_length, gru_hidden_size, num_static_feature=0,
                 GRU_params=None, mlp_hidden_size=None, mlp_activation="ReLU"):
        super(GRU, self).__init__()
        self.output_sequence_length = output_sequence_length
        self.num_static_feature = num_static_feature
        if GRU_params is None:
            GRU_params = {}
        self.gru = nn.GRU(input_size=num_dynamic_feature, hidden_size=gru_hidden_size, batch_first=True, **GRU_params)
        total_input_size = gru_hidden_size * (2 if GRU_params.get("bidirectional", False) else 1) + num_static_feature
        self.fcs = nn.Sequential()
        mlp_activation = getattr(nn, mlp_activation)
        if mlp_hidden_size is not None and len(mlp_hidden_size) > 0:
            for i, h in enumerate(mlp_hidden_size):
                if i == 0:
                    self.fcs.add_module(f"fc{i + 1}", nn.Linear(total_input_size, h))
                else:
                    self.fcs.add_module(f"act{i}", mlp_activation())
                    self.fcs.add_module(f"fc{i + 1}", nn.Linear(mlp_hidden_size[i - 1], h))
            self.fcs.add_module(f"act{len(mlp_hidden_size)}", mlp_activation())
            self.fcs.add_module(f"fc{len(mlp_hidden_size) + 1}", nn.Linear(mlp_hidden_size[-1], output_sequence_length))
        else:
            self.fcs.add_module(f"fc1", nn.Linear(total_input_size, output_sequence_length))

    def forward(self, x_dynamic, x_static=None):
        if x_static is None and self.num_static_feature > 0:
            raise ValueError("Static features are required for this model")
        if x_static is not None and x_static.size(1) != self.num_static_feature:
            raise ValueError(f"Expected {self.num_static_feature} static features, but got {x_static.size(1)}")

        # Passing dynamic features through the GRU
        gru_output, _ = self.gru(x_dynamic)
        # Considering only the last output of the GRU
        gru_last_output = gru_output[:, -1, :]

        # Concatenating the last GRU output with static features
        if x_static is not None:
            combined_features = torch.cat((gru_last_output, x_static), dim=1)
        else:
            combined_features = gru_last_output

        # Passing combined features through the fully connected layer
        y_pred = self.fcs(combined_features)
        # Reshaping the output to match target shape
        batch_size = x_dynamic.size(0)
        y_pred = y_pred.reshape(batch_size, self.output_sequence_length, -1)
        return y_pred