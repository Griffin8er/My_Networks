import torch
from torch import nn

device = 'mps:0' if torch.backends.mps.is_available() else 'default'

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step, device=device).unsqueeze(dim=1)
y = weight*X + bias

train_split = int(0.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

'''plt.scatter(X_train, y_train, 2)
plt.scatter(X_test, y_test, 2)
plt.show()'''


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


model = Model()
model.to(device)
torch.manual_seed(42)

loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=.03)

y_preds = []
epochs = 1000

for epoch in range(epochs):
    model.train()
    y_pred = model.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.inference_mode():
        y_fake = ((model.linear_layer.weight * X_test + model.linear_layer.bias).flatten()).tolist()
        y_preds.append(y_fake)
    model.eval()