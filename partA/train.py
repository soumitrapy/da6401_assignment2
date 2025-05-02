import torch
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train_one_epoch(model, dl, optimizer, loss_fn, epoch=1, device='cpu', use_wandb=False):
    model.to(device)
    model.train()
    running_loss = 0.0
    correct, total  = 0,0
    pbar = tqdm(dl,desc=f"Epoch {epoch+1}")
    n_steps_per_epoch = math.ceil(len(dl.dataset) / dl.batch_size)
    for step, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
        avg_loss = running_loss/(step+1)
        acc = 100.*correct/total

        pbar.set_postfix(train_loss=avg_loss, train_accuracy=acc)
        if use_wandb:
            metrics = {"train/train_loss": avg_loss, "train/train_accuracy": acc}
            wandb.log(metrics, step=step+1+n_steps_per_epoch*epoch)
    

    
    return avg_loss, acc

def val_one_epoch(model, dl, loss_fn, device='cpu', visualize = False):
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct, total  = 0,0
    pbar = tqdm(dl,desc=f"Validation: ")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            avg_loss = running_loss/(i+1)
            acc = 100.*correct/total

            pbar.set_postfix(val_loss=avg_loss, val_accuracy=acc)
    
    return avg_loss, acc



# def train(model, traindl, optimizer, loss_fn, config, scheduler=None, valdl=None, device='cpu'):
#     model.to(device)
#     best_loss = float('inf')
#     os.makedirs('checkpoints', exist_ok=True)
#     for epoch in range(config['epochs']):
#         model.train()
#         train_loss, train_acc = train_one_epoch(model, traindl, optimizer, loss_fn, config, epoch=epoch, device=device)
#         if valdl and (epoch+1)%config['val_interval']==0:
#             val_loss, val_acc = val_one_epoch(model, valdl, loss_fn, config, epoch=epoch, device=device)
#             if val_loss<best_loss:
#                 best_loss = val_loss
#                 model_name = type(model).__name__+'_'+device.type+str(datetime.now())[:15]+'.pth'
#                 model_path = os.path.join('checkpoints', model_name)
#                 torch.save(model.state_dict(), model_path)
                
#         if scheduler:
#             scheduler.step()


def train(model, optimizer, loss_fn, dataloaders, config, model_config, scheduler = None, device = 'cpu', use_wandb=False):
    model.to(device)
    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(config['epochs']):
        model.train()
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, loss_fn, epoch=epoch, device=device, use_wandb=use_wandb)
        if (epoch+1)%config['val_interval']==0:
            val_loss, val_acc = val_one_epoch(model, dataloaders['val'], loss_fn, device=device)
            if use_wandb:
                val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": val_acc}
                wandb.log(val_metrics,step=wandb.run.step)
            if val_loss<best_loss:
                best_loss = val_loss
                model_name = type(model).__name__+'_'+device.type+'epoch'+str(epoch)+str(datetime.now())[8:18]+'.pt'
                model_path = os.path.join('checkpoints', model_name)
                torch.save(model.state_dict(), model_path)
                if use_wandb:
                    artifact = wandb.Artifact("trained_models", type="model", metadata=model_config)
                    artifact.add_file(model_path)
                    artifact.save()
                    wandb.log_artifact(artifact)

        if scheduler:
            scheduler.step()
    return 

def evaluate(model, dl, loss_fn, device = 'cpu', use_wandb=False):
    test_loss, test_acc = 0, 0
    # test_loss, test_acc = val_one_epoch(model, dl, loss_fn, device=device)
    highest_losses, hardest_examples, true_labels, predictions = get_examples(model, dl.dataset, loss_fn, n = 3, hard = 3, device = device)
    if use_wandb:
        wandb.summary.update({"loss": test_loss, "accuracy": test_acc})
        wandb.log({"examples":
            [wandb.Image(hard_example, caption=str(int(pred)) + "," +  str(int(label)))
             for hard_example, pred, label in zip(hardest_examples, predictions, true_labels)]})
    return true_labels, predictions

def get_examples(model, testing_set, loss_fn, n = 10, hard = 10, device = 'cpu'):
    model.eval()
    loader = DataLoader(testing_set, 1, shuffle=True)

    # get the losses and predictions for each item in the dataset
    losses = None
    predictions = None
    i = 0
    with torch.no_grad():
        for data, target in loader:
            i += 1
            if i>10:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)
            break

    argsort_loss = torch.argsort(losses, dim=0)
    print(argsort_loss.shape)
    indices = argsort_loss[-hard] #


    # highest_k_losses = losses[argsort_loss[-k:]]
    # hardest_k_examples = testing_set[argsort_loss[-k:]][0]
    # true_labels = testing_set[argsort_loss[-k:]][1]
    # predicted_labels = predictions[argsort_loss[-k:]]

    # return highest_k_losses, hardest_k_examples, true_labels, predicted_labels
    return losses[indices], testing_set[indices][0], testing_set[indices][1], predictions[indices]

def visualize_predictions(model, test_loader, NUM_BATCHES_TO_LOG=10, device='cpu'):
        # W&B: Create a Table to store predictions for each test step
    columns=["id", "image", "guess", "truth"]
    for digit in range(10):
      columns.append("score_" + str(digit))
    test_table = wandb.Table(columns=columns)

    # test the model
    model.eval()
    model.to(device)
    log_counter = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if log_counter < NUM_BATCHES_TO_LOG:
              log_test_predictions(images, labels, outputs, predicted, test_table, log_counter)
              log_counter += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        # W&B: Log accuracy across training epochs, to visualize in the UI
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

    # W&B: Log predictions table to wandb
    wandb.log({"test_predictions" : test_table})


# convenience funtion to log predictions for a batch of test images
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  # obtain confidence scores for all classes
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  # adding ids based on the order of the images
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # add required info to data table:
    # id, image pixels, model's guess, true label, scores for all classes
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1

# def imshow(inp, title=None):
#     """Display image for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# def visualize_model(model,dl, class_names, num_images=6, device='cpu'):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dl):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[preds[j]]}')
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)