
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=True)
        self.conv.weight.data.fill_(0.3)
        self.conv.bias.data.fill_(0.2)

    def forward(self, x):
        x = self.conv(x)
        # return x.view(int(x.size(0)), -1)
        return x.view(-1, int(x.numel() // x.size(0)))

model = Model().eval()

x = torch.full((1, 1, 3, 3), 1.0)
y = model(x)
print(y)

torch.onnx.export(
    model, (x, ), "lesson1.onnx", verbose=True
)




