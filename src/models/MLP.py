class MLPImage(nn.Module):
    def __init__(self, in_channels = 3, num_classes=10):
        super().__init__()
        self.Flatten = nn.Flatten()
        self.linear_relu_block = nn.Sequential(
            nn.Linear(in_features=in_channels * 32 * 32 , out_features=1024,
                     bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )
    def forward(self, x):
        output = self.Flatten(x)
        output = self.linear_relu_block(output)
        return output

