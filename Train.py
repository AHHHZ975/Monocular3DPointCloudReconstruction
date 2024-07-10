# import the necessary packages
from NN import CAE_AHZ, PSGN, TCNN, CAE_new, Pixel2Point, Pixel2Point_InitialPC, PreTrainedTransformer, PureTransformer, ViT_CNN
from torch.autograd import Variable
from torch.utils.data import random_split           # Constructs a random training/testing split from an input set of data
from torch.utils.data import DataLoader             # PyTorch’s awesome data loading utility that allows us to effortlessly build data pipelines to train our CNN
from torch.optim import Adam, AdamW, Adadelta                        # The optimizer we’ll use to train our neural network
from torch.optim import SGD                         # The optimizer we’ll use to train our neural network
from torch import nn, utils                         # PyTorch’s neural network implementations


import numpy as np
import torch
import time
import Dataset
import Config as cfg
from neuralnet_pytorch.metrics import chamfer_loss
import Utils
from SAM import SAM

def trainInitialization():
    numberOfTrainData = 470    # Airplane: 3230  # Cellphone: 750  # Bottle: 470   # Rifle: 2140  # Car: 3160   # Bench: 1640  # Bike: 300
    numberOfTestData =  25     # Airplane: 810   # Cellphone: 80   # Bottle: 25    # Rifle: 230  # Car: 350    # Bench: 170    # Bike: 30 
                               
                               # Speaker: 1370    # Sofa: 2860
                               # Speaker: 220     # Sofa: 310
    
    global device
    global trainDataLoader, valDataLoader, testDataLoader
    global trainSteps, valSteps, testSteps

    # Setup device (CPU or GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("GPU") 
    else:
        device = torch.device("cpu")
        print("CPU")

    
    # Load the ShapeNet dataset
    print("[INFO] loading the ShapeNet dataset...")
    trainData = Dataset.ShapeNetDataset(numberOfTrainData, train=True)
    testData = Dataset.ShapeNetDataset(numberOfTestData, train=False)


    # Calculate the train/validation split
    print("[INFO] generating the train/validation split...")
    numTrainSamples = int(len(trainData) * cfg.TRAIN_SPLIT)
    numValSamples = int(len(trainData) * cfg.VAL_SPLIT)
    numTestSamples = len(testData)

    # Splitting training data
    valData = []
    splittedTrainData = []
    for i in range(numValSamples):
        valData.append(trainData[numTrainSamples + i])
    for i in range(numTrainSamples):
        splittedTrainData.append(trainData[i])


    # Initialize training, validation, and test data loaders
    trainDataLoader = DataLoader(splittedTrainData, shuffle=True, batch_size=cfg.BATCH_SIZE)
    valDataLoader = DataLoader(valData, batch_size=cfg.BATCH_SIZE)
    testDataLoader = DataLoader(testData, batch_size=cfg.BATCH_SIZE)

    # Calculate steps per epoch for training, validation, and test sets
    trainSteps = len(trainDataLoader.dataset) // cfg.BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // cfg.BATCH_SIZE
    testSteps = len(testDataLoader.dataset) // cfg.BATCH_SIZE

    # Show data numbers and steps
    print("Number of train data: ", len(trainDataLoader.dataset))
    print("Number of validation data: ", len(valDataLoader.dataset))
    print("Number of Test data: ", len(testDataLoader.dataset))
    print("Train steps: ", trainSteps)
    print("Validation steps: ", valSteps)
    print("Test steps: ", testSteps)


def trainCAE():
    # Initialize the model
    print("[INFO] initializing the CAE model...")
    model = CAE_AHZ().to(device=device)
    # model = PSGN().to(device=device)
    # model = Pixel2Point().to(device=device)
    
    # Initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=cfg.INIT_LR, weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)

    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):        
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred = model(x)
            
            # Calculate the training loss
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            loss = chamfer_loss(pred, y)

            # Zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss


        ################################################## Testing ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in testDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalTestLoss += loss


        ############################################## Learning Rate Scheduler ####################################################################
        lr_scheduler.step()

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        avgTestLoss = totalTestLoss / testSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")
        print(f"Test loss: {avgTestLoss:.6f}\n")



    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)


def trainCAE_AutoEnd():

    lastLoss = 0.0
    currentLoss = 0.0
    e = 0
    ee = 0
    cnt = 1


    # Initialize the model
    print("[INFO] initializing the CAE model...")
    model = CAE_new().to(device=device)

    # Initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=cfg.INIT_LR, weight_decay=1e-5)

    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    # for e in range(cfg.EPOCHS):        
    while True:
        ################################################## Training ####################################################################
                
        lastLoss = currentLoss

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred = model(x)
            
            # Calculate the training loss
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            loss = chamfer_loss(pred, y)
            
            # Zero out the gradients, perform the backpropagation step, and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {cnt}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")

        currentLoss = avgValLoss
        diffLoss = abs(currentLoss - lastLoss)
        print(f"Difference loss: {diffLoss:.6f}")
        print(f"e: {e:.6f}")
        # print(f"ee: {ee:.6f}\n")
        if (diffLoss <= 0.05):
            e += 1
        # if (currentLoss > lastLoss):
        #     ee += 1

        if e >= 10: # and ee >= 25:
            break
        
        cnt += 1


    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)


def trainViT(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, num_points_conv):
    
    # Initialize the model
    print("[INFO] initializing the TCNN model...")
    model = TCNN(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, num_points_conv).to(device=device)

    # Initialize our optimizer and loss function
    optimizer = Adam(model.parameters(), lr=cfg.INIT_LR, weight_decay=1e-5)

    # Algorithms for changing learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.5) # For ViT+CNN -> Bottle
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70], gamma=0.5)  # For ViT+CNN -> Cellphone 
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.5)  # For ViT+CNN -> Airplane and Rifle
    # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=5, max_iters=cfg.EPOCHS)
    
    
    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred = model(x)
            
            # Calculate the training loss            
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            loss = chamfer_loss(pred, y)
            
            # Zero out the gradients, perform the backpropagation step, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the loss to the total training loss so far
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss


        ################################################## Testing ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in testDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalTestLoss += loss


        ############################################## Learning Rate Scheduler ####################################################################
        lr_scheduler.step()

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        avgTestLoss = totalTestLoss / testSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")
        print(f"Test loss: {avgTestLoss:.6f}\n")


    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/TCNN.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)


def trainPreTrainViT(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, num_points_conv):
    
    # Initialize the model
    print("[INFO] initializing the TCNN model...")
    model = PreTrainedTransformer(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, num_points_conv).to(device=device)

    # Initialize our optimizer and loss function
    optimizer = Adam(model.parameters(), lr=cfg.INIT_LR, weight_decay=1e-5)

    # Algorithms for changing learning rate
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.5) # For ViT+CNN -> Bottle
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70], gamma=0.5)  # For ViT+CNN -> Cellphone 
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.5)  # For ViT+CNN -> Airplane and Rifle
    # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=5, max_iters=cfg.EPOCHS)
    
    
    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass
            pred = model(x)
            
            # Calculate the training loss            
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
            loss = chamfer_loss(pred, y)
            
            # Zero out the gradients, perform the backpropagation step, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the loss to the total training loss so far
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss


        ################################################## Testing ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in testDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalTestLoss += loss


        ############################################## Learning Rate Scheduler ####################################################################
        lr_scheduler.step()

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        avgTestLoss = totalTestLoss / testSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")
        print(f"Test loss: {avgTestLoss:.6f}\n")


    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)


def trainViT_SAM(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, num_points_conv):
    
    # Initialize the model
    print("[INFO] initializing the TCNN model...")
    model = PureTransformer(embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, num_points_conv).to(device=device)

    # Initialize our optimizer and loss function
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer, lr=cfg.INIT_LR, weight_decay=1e-5)

    # Algorithms for changing learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=cfg.EPOCHS)

    # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=5, max_iters=cfg.EPOCHS)
    
    
    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            # Perform a forward pass            
            
            # Calculate the training loss                        
            # Zero out the gradients, perform the backpropagation step, and update the weights

            def closure():
                pred = model(x)
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))               
                loss = chamfer_loss(pred, y)
                loss.backward()
                return loss

            
            pred = model(x)
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))            
            loss = chamfer_loss(pred, y)
            
            loss.backward()


            optimizer.step(closure)            
            optimizer.zero_grad()
      


            # Add the loss to the total training loss so far
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss


        ################################################## Testing ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in testDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalTestLoss += loss


        ############################################## Learning Rate Scheduler ####################################################################
        lr_scheduler.step()

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        avgTestLoss = totalTestLoss / testSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")
        print(f"Test loss: {avgTestLoss:.6f}\n")


    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)


def trainCAE_SAM():
    # Initialize the model
    print("[INFO] initializing the CAE model...")
    model = CAE_AHZ().to(device=device)

    # Initialize our optimizer and loss function
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer, lr=cfg.INIT_LR, weight_decay=1e-5)


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=cfg.EPOCHS)
    
    # Initialize a dictionary to store training and validation history
    H = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    # Measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    

    # Loop over our epochs
    for e in range(cfg.EPOCHS):        
        ################################################## Training ####################################################################

        # Set the model in training mode
        model.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (x, y) in trainDataLoader:
            # Reshaping the image
            x = Variable(x)

            # Send the input to the device            
            (x, y) = (x.to(device), y.to(device))
            
            def closure():
                pred = model(x)
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))               
                loss = chamfer_loss(pred, y)
                loss.backward()
                return loss

            
            pred = model(x)
            pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
            y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))            
            loss = chamfer_loss(pred, y)
            
            loss.backward()
            optimizer.step(closure)            
            optimizer.zero_grad()

            # Add the loss to the total training loss so far and calculate the number of correct predictions
            totalTrainLoss += loss


        ################################################## Validation ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in valDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalValLoss += loss


        ################################################## Testing ####################################################################

        # Switch off autograd for evaluation
        # Use the torch.no_grad() context to turn off gradient tracking and computation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            # Loop over the validation set
            for (x, y) in testDataLoader:
                # Reshaping the image
                x = Variable(x)

                # Send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # Make the predictions
                pred = model(x)

                # Calculate the validation loss
                pred = torch.reshape(pred, (-1, cfg.SAMPLE_SIZE, 3))
                y = torch.reshape(y, (-1, cfg.SAMPLE_SIZE, 3))
                loss = chamfer_loss(pred, y)

                totalTestLoss += loss


        ############################################## Learning Rate Scheduler ####################################################################
        lr_scheduler.step()

        ################################################## Statistics ####################################################################

        # Calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        avgTestLoss = totalTestLoss / testSteps

        # Update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{cfg.EPOCHS}")
        print(f"Train loss: {avgTrainLoss:.6f}")
        print(f"Val loss: {avgValLoss:.6f}")
        print(f"Test loss: {avgTestLoss:.6f}\n")



    # finish measuring how long training took
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime:.2f}s")


    # Serialize and save the model to disk
    modelPath = cfg.ROOT_DIR + '/Convolution_AE.pt'
    torch.save(model, modelPath)


    ################################################### Graphically displaying statistics  ##############################################################
    # Plot the training loss and accuracy
    Utils.save_loss_plot(H)



class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor




if __name__ == '__main__':
    
    # Train the model
    torch.cuda.empty_cache()
    trainInitialization()
    trainCAE_SAM()
    trainViT(embed_dim=768,
             hidden_dim=512,
             num_heads=8,
             num_layers=6,
             patch_size=64,
             num_channels=3,
             num_patches=16,
             num_classes=cfg.SAMPLE_SIZE*3, 
             num_points_conv=900)

    torch.cuda.empty_cache()
