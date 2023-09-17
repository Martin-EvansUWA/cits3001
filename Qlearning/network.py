from torch import nn

class MarioNetwork(nn.Module):

    # "qnetwork"
    def __init__(self, input_dim=(4,84,84), action_dim=84):
        super().__init__()
        c, height, width = input_dim

        if height != 84:
            raise ValueError(f"Expecting input height: 84, got: {height}")
        if width != 84:
            raise ValueError(f"Expecting input width: 84, got: {width}")
        print(c)
        print(input_dim)

        self.q_network = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )


    def forward(self, input):
       return self.q_network(input)