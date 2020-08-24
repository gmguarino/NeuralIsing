import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from matplotlib.colors import colorConverter
from model import Network
from dataset import Ising
from utils import evaluate, format_mse

MODEL_PATH = "./cfg/net.pt"
DATA_PATH = "../Ising/data/"
CUDA = torch.cuda.is_available()
BATCH_SIZE = 1

test_set = Ising(DATA_PATH, train=False)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True 
)

net = Network()
net.load_state_dict(torch.load(MODEL_PATH))

if CUDA:
    net.to(torch.device('cuda'))
net.eval()

losses, targets, predictions = evaluate(net, test_loader, CUDA=CUDA)


r = spearmanr(targets, predictions).correlation

loss_color = colorConverter.to_rgba('mediumseagreen', alpha=.5)

markerOptions = dict(markerfacecolor='none', markeredgewidth=1.5)

fig, ax = plt.subplots(figsize=(10,6))

ax.title.set_text(f"Spearman Correlation Coefficient r={r:.4f}")

ax.plot(targets, predictions, linestyle='None', marker='s', color='blue', **markerOptions)
ax.set_ylabel('Predicted Temperature T/T$_C$')
ax.set_xlabel("Ground truth T/T$_C$")
ax.yaxis.label.set_color('blue')
print(min(targets))
targets, losses, width = format_mse(targets, losses)
print(len(targets))
ax2 = ax.twinx()

percent_losses = np.array(losses) / np.array(targets) * 100

ax2.bar(targets, percent_losses, width=width, color='red', align='center', alpha=0.2)
ax2.set_ylabel('Mean squared error (%)')
ax2.yaxis.label.set_color('red')

plt.show()
fig.savefig("./images/correlation.png")

test_iter = iter(test_loader)
for i in range(5):
    img, targets = next(test_iter)
    if CUDA:
        img = img.cuda()
        targets["T"] = targets["T"].cuda()
    with torch.no_grad():
        output = net(img)
    img = img.cpu().squeeze(0).squeeze(0)
    plt.figure()
    plt.imshow(img.numpy(), cmap='Greys',  interpolation='nearest')
    plt.title(f"Predicted temperature : {output.item():.2f}; simulated temperature : {targets['T'].item():.2f}")
    plt.savefig("./images/" + "{:.2f}".format(output.item()).replace(".", "_") + ".png")