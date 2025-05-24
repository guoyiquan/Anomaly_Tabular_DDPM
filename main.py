import torch
import os
import copy
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataloader.dataloader import get_dataloader, transform_loader2array
from methods.diffusion import GaussianDiffusion
from methods.model import TabDDPM, MLP
from methods.loss import AnomalyLoss
from methods.utils import find_sim
from metrics import aucPerformance, F1Performance, observation_indicators


def train(
    dir,
    data_name,
    batch_size=2048,
    TIMESTEPS=100,
    lr=5e-4,
    _lambda=100,
    epochs=300,
    schedule_name="cosine",
    scaler="minmax",
):
    # load_data
    path = os.path.join(dir, data_name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, test_loader = get_dataloader(
        path, batch_size, scaler=scaler, type="1"
    )

    train_data, train_label, test_data, test_label = transform_loader2array(
        train_loader, test_loader
    )

    # def models
    generator = TabDDPM(data_dim=train_data.shape[1]).to(device)
    ddpm = GaussianDiffusion(
        device, generator, schedule_name=schedule_name, timesteps=TIMESTEPS
    )

    discriminator = MLP(
        d_in=train_data.shape[1],
        d_layers=[256, 128, 128, 16],
        dropouts=0.1,
        d_out=2,
        bias=False,
    ).to(device)

    DynamicConstraintLoss = AnomalyLoss(
        device=device,
        dim=train_data.shape[1],
    )
    # optim
    optimizer_G = optim.Adam(generator.parameters(), lr=2 * lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs)

    for epoch in range(epochs):
        discriminator.train()

        for x, _ in train_loader:

            # train discriminator

            cur_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, TIMESTEPS, (x.shape[0],), device=device).long()

            with torch.no_grad():
                x_aug = x.detach()
                fake_x = ddpm.p_sample_loop((cur_batch_size, x.shape[1])).detach()
                x_aug, fake_x = find_sim(x, fake_x)

            real_outputs = discriminator(x_aug)
            fake_outputs = discriminator(fake_x)

            real_labels = torch.zeros(x_aug.shape[0], dtype=torch.long, device=device)
            fake_labels = torch.ones(fake_x.size(0), dtype=torch.long, device=device)

            discriminator_loss = nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, 10.0]).to(device)
            )(
                torch.cat([real_outputs, fake_outputs]),
                torch.cat([real_labels, fake_labels]),
            )

            optimizer_D.zero_grad()
            discriminator_loss.backward()
            optimizer_D.step()

            # train generator

            generator.train()
            discriminator.eval()

            fake_x = ddpm.p_sample_loop((cur_batch_size, x.shape[1]))
            loss_elbo = ddpm.p_losses(x, t)
            loss_adv = nn.CrossEntropyLoss()(
                discriminator(fake_x),
                torch.zeros(fake_x.shape[0], dtype=torch.long).to(device),
            )
            loss_ano = DynamicConstraintLoss(fake_x, x)

            loss = (loss_elbo + loss_adv) + _lambda * loss_ano

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
        
        scheduler_G.step()
        scheduler_D.step()

    torch.save(generator, "generator.pth")
    torch.save(discriminator, "discriminator.pth")

    return generator, discriminator


def test(
    dir,
    data_name,
    discriminator,
    batch_size=2048,
    scaler="minmax",
):
    path = os.path.join(dir, data_name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, test_loader = get_dataloader(
        path, batch_size, scaler=scaler, type="1"
    )

    train_data, train_label, test_data, test_label = transform_loader2array(
        train_loader, test_loader
    )
    discriminator.eval()
    y_test_pred = discriminator(torch.tensor(test_data).to(device))
    y_test_pred = torch.nn.Softmax(dim=1)(y_test_pred)
    y_test_pred = y_test_pred.detach().cpu().numpy()
    rauc, ap = aucPerformance(y_test_pred[:, 1], test_label)
    f1,thre = F1Performance(y_test_pred[:, 1], test_label)
    print(ap, rauc)

    return ap, rauc, f1


if __name__ == "__main__":

    # config
    dir = "./data"
    data_name = "4_breastw.npz"
    batch_size = 2048
    TIMESTEPS=100
    lr=5e-4
    _lambda=100
    epochs=300
    schedule_name="cosine"
    scaler="minmax"

    # train
    generator, discriminator = train(
        dir,
        data_name,
        batch_size=batch_size,
        TIMESTEPS=TIMESTEPS,
        lr=lr,
        _lambda=_lambda,
        epochs=epochs,
        schedule_name=schedule_name,
        scaler=scaler,
    )
    # test
    ap, rauc, f1 = test(
        dir,
        data_name,
        discriminator,
        batch_size=batch_size,
        scaler=scaler,
    )
    print(ap, rauc, f1)