import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

def loss_fn(outputs, targets):
    return F.mse_loss(outputs, targets)

def train(data_loader, model, optimizer, device):
    model.train()
    result_loss = 0

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = d["X"].to(device, dtype=torch.float)
        target = d["Y"].to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs, embed = model(inputs)
        loss = loss_fn(outputs, target)

        loss.backward()
        optimizer.step()

        result_loss += loss
    return result_loss



def eval(data_loader, model, device):
    model.eval()
    result_loss = 0

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = d["X"].to(device, dtype=torch.float)
            target = d["Y"].to(device, dtype=torch.float)

            output, embed = model(inputs)

            loss = loss_fn(output, target)

            result_loss += loss

    print('Validation Loss:- ', result_loss/len(data_loader))


def generate_low_dimensional_embeddings(data_loader, model, device):
    model.eval()
    embed_list = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = d["X"].to(device, dtype=torch.float)
            
            output, embed = model(inputs)

            embed_list.append(embed.cpu().detach().numpy().tolist())

        return embed_list





