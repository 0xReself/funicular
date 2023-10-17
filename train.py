import model
import dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

EPOCHS = 50
BATCH_SIZE = 74
LR = 1

train_dataset = dataset.ImageDataset("data/left", "data/right", transform=dataset.transform_img)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = dataset.ImageDataset("data/left", "data/right", transform=dataset.transform_img)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

TRAIN_DATA = len(train_dataset)

model = model.Swipo()

optimizer = SGD(model.parameters(), lr=LR)
loss_class = CrossEntropyLoss()

def run_epoch(idx, dataloader, testing=False):
    running_loss = 0.
    total = len(dataloader)
    equal_total = 0

    for i, data in enumerate(dataloader):
        total += BATCH_SIZE
        batch, labels = data

        if testing == False:
            optimizer.zero_grad()

        output = model(batch)

        output = output.reshape(-1)
        labels = labels.reshape(-1)

        rounded_output = torch.round(output)
        _, equal = torch.unique(rounded_output[rounded_output == labels], return_counts=True)
        equal = equal[0]
        equal_total += equal

        loss = loss_class(output, labels)
        running_loss += loss.item()
        
        if testing == False:
            loss.backward()
            optimizer.step()

        print("epoch: {idx}, batch: {batch} / {number_of_batches}, loss: {loss}, accuracy: {accuracy}".format(
            idx=idx, batch=i + 1, number_of_batches=int(TRAIN_DATA/BATCH_SIZE), loss=loss.item(), accuracy =  (equal * 100 / BATCH_SIZE)))
    
    return (equal_total * 100 / total).item()

model.train(True)

for epoch in range(EPOCHS):
    model.train(True)
    accuracy = run_epoch(epoch, train_dataloader)
    print(accuracy)


model.train(False)
accuracy = run_epoch(epoch, test_dataloader, testing=True)
print("accuracy test: {acc}".format(acc = accuracy))



