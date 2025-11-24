import numpy as np
import random
import time
import utils
import matplotlib.pyplot as plt
import torch
from pytorch3d.loss import chamfer_distance


def train_epoch(train_loader, net, optimizer, max_rollout_horizon, device):
    epoch_loss = 0
    
    for i, (x_t, a, delta_tpH, _) in enumerate(train_loader):
        optimizer.zero_grad()
        
        x_t = x_t.to(device).permute(0, 2, 1)     # [B, 3, N1]
        a = a.to(device)
        delta_tpH = delta_tpH.to(device)          # [B, N1, 3]
        rollout_horizon = random.randint(1, max_rollout_horizon)
        
        delta_tp1 = torch.zeros_like(x_t.permute(0, 2, 1))
        
        for j in torch.arange(rollout_horizon):
            delta_tp1 = delta_tp1 + net(x_t, a[:,j,:])
            x_t = x_t + delta_tp1.permute(0, 2, 1)

        loss, _ = chamfer_distance(delta_tpH[:,rollout_horizon-1,:,:], delta_tp1) # compare prediction to ground truth
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/(i+1), rollout_horizon


def test_batch(x_t, a, delta_tpH, net, rollout_horizon, device): # test with a batch of inputs
    with torch.no_grad():
        x_t = x_t.to(device).permute(0, 2, 1)     # [B, 3, N1]
        a = a.to(device)
        delta_tpH = delta_tpH.to(device)
        delta_tp1 = torch.zeros_like(x_t.permute(0, 2, 1))

        for j in torch.arange(rollout_horizon):
            delta_tp1 = delta_tp1 + net(x_t, a[:,j,:])
            x_t = x_t + delta_tp1.permute(0, 2, 1)
        
        loss, _ = chamfer_distance(delta_tpH[:,rollout_horizon-1,:,:], delta_tp1)
    return loss.item(), x_t.permute(0, 2, 1).cpu()


def test_epoch(test_loader, net, rollout_horizon, device):
    with torch.no_grad():
        epoch_loss = 0
        for i, (x_t, a, delta_tpH, _) in enumerate(test_loader):
            loss, output = test_batch(x_t, a, delta_tpH, net, rollout_horizon, device)
            epoch_loss += loss
    return epoch_loss/(i+1)


def train_model(train_loader, test_loader, net, epochs, max_rollout_horizon, optimizer, device, save_results, output_folder):
    train_loss_list = []  
    test_loss_list = []
    test_output_list = []
    
    for i in range(epochs) :
    
        startTime = time.time()
        
        train_loss, rollout_horizon = train_epoch(train_loader, net, optimizer, max_rollout_horizon, device)
        train_loss_list.append(train_loss)
        
        test_loss = test_epoch(test_loader, net, rollout_horizon, device)
        test_loss_list.append(test_loss)
        
        epoch_time = time.time() - startTime
        
        writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
        
        plt.plot(train_loss_list, label="Train")
        plt.plot(test_loss_list, label="Test")
        plt.legend()
    
        if(save_results):
    
            # write the text output to file
            with open(output_folder + "prints.txt","a") as file: 
                file.write(writeString)
    
            # update the loss graph
            plt.savefig(output_folder + "loss.png")
            plt.close()
    
            # save input/output as image file
            if(i%50==0):
                test_samples, test_actions, test_deltas, test_samples_next = next(iter(test_loader))
                loss , test_output = test_batch(test_samples, test_actions, test_deltas, net, max_rollout_horizon, device)
                utils.plotPCbatch(test_samples, test_samples_next[:,-1,:,:], test_output, show=False, save=True, name = (output_folder + "epoch_" + str(i)))









