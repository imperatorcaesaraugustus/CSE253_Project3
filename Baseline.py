#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:


from baseline_cnn import *
from baseline_cnn import BasicCNN


def Resample(train_loader):
    threshold = [0.0 for i in range(14)]
    stat_true, stat_false = [0 for i in range(14)], [0 for i in range(14)]
    for batch_cnt, (images, labels) in enumerate(train_loader, 0):
        if batch_cnt % 100 == 99: print(batch_cnt)
        if images.size()[0] != 16:
            break
        for i in range(16):
            for j in range(14):
                if labels[i][j] == 1:
                    stat_true[j] += 1
                else:
                    stat_false[j] += 1
    for i in range(14):
        threshold[i] = stat_true[i] / (stat_true[i] + stat_false[i])
        print(threshold[i])


# Setup: initialize the hyperparameters/variables
num_epochs = 2           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.0002  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize(size=(512, 512)), transforms.ToTensor()])


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 0, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform,
    p_val=p_val, p_test=p_test, shuffle=True, show_sample=False, extras=extras)

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = BasicCNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

#TODO: Define the loss criterion and instantiate the gradient descent optimizer
criterion = nn.BCELoss()

#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(model.parameters())


# In[26]:


# Track the loss across training
total_loss = []
avg_minibatch_loss = []
val_criterion = nn.BCELoss()
total_val_loss = []

# Begin training procedure
for epoch in range(num_epochs):

    N = 500  # output every N mini-batches
    N_minibatch_loss = 0.0    

    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)
        #labels = labels.long()

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()
        
        # Perform the forward pass through the network and compute the loss
        outputs = model(images)   # images = batch in BasicCNN.forawrd(self, batch)
        if minibatch_count == 10: print(outputs)
        loss = criterion(outputs, labels)
        
        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        # Add this iteration's loss to the total_loss
        #total_loss.append(loss.item())
        N_minibatch_loss += loss
        
        #TODO: Implement cross-validation
        #if minibatch_count >= 600: break
        
        if minibatch_count % N == N - 1:    
            
            # Print the loss averaged over the last N mini-batches 
            N_minibatch_loss /= N
            print(labels, outputs)
            print('Epoch %d, average minibatch %d loss: %.3f' %
                (epoch + 1, minibatch_count + 1, N_minibatch_loss))
            
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            total_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0
            
            with torch.no_grad():
                minibatch_val_loss = 0.0
                print("minibatch: ", minibatch_count + 1)
                i = 0
                best_val_loss = 1e6 
                ratio_val_used = 10
                flag = True
                for i, (val_images, val_labels) in enumerate(val_loader, 0):
                    val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
                    if i % ratio_val_used != 0: continue
                    if val_images.size()[0] != batch_size: break
                    val_output = model(val_images)
                    val_loss = val_criterion(val_output, val_labels)
                    minibatch_val_loss += val_loss
                minibatch_val_loss /= (i/ratio_val_used)
                if best_val_loss < minibatch_val_loss:
                    print("Early stop triggered.")
                    print("Best validation loss:", best_val_loss)
                    flag = False
                    break
                if flag == False: break
                best_val_loss = minibatch_val_loss
                total_val_loss.append(minibatch_val_loss)
                print("Validation loss:", best_val_loss)
                del minibatch_val_loss, val_images, val_labels, val_loss, val_output
        del loss, outputs
        
    print("Finished", epoch + 1, "epochs of training")
print("Training complete after", epoch + 1, "epochs")


# In[ ]:


def tester(model, test_loader, batch_size):
    with torch.no_grad():
        minibatch_val_loss = 0.0
        #print("minibatch: ", minibatch_count + 1)
        conf_mat = [[0 for i in range(15)] for j in range(15)]
        conf_vec = [0 for i in range(14)]
        naive_thres = [0.5]*14
        thres_judge = [0.1021,0.0252,0.1191,0.177,0.0518,0.0568,0.0119,0.0469,0.0413,0.0207,0.022,0.015,0.0305,0.002]
        thres = naive_thres
        cnt = 0
        for cnt, (test_images, test_labels) in enumerate(test_loader, 0):
            test_images, test_labels = test_images.to(computing_device), test_labels.to(computing_device)
            if test_images.size()[0] != batch_size: break
            test_output = model(test_images)
            for i in range(batch_size):
                for j in range(14):
                    if test_output[i][j] < thres[j] and test_labels[i][j] < 0.5:   # True negatives
                        conf_mat[14][14] += 1
                        conf_vec[j] += 1
                    if test_output[i][j] < thres[j] and test_labels[i][j] >= 0.5:  # False positives
                        conf_mat[14][j] += 1
                        for k in range(14):
                            if test_output[i][k] > thres[k]: conf_mat[k][j] += 1
                    if test_output[i][j] >= thres[j] and test_labels[i][j] >= 0.5: # True positives
                        conf_mat[j][j] += 1 
                    if test_output[i][j] >= thres[j] and test_labels[i][j] < 0.5:  # False negatives
                        conf_mat[j][14] += 1
                        for k in range(14):
                            if test_labels[i][k] > 0.5: conf_mat[j][k] += 1
        Accuracy, Recall, Precision, BCR = [0.0]*14, [0.0]*14, [0.0]*14, [0.0]*14 
        for i in range(14):
            Accuracy[i] = (conf_mat[i][i] + conf_vec[i])/(conf_mat[i][i]+conf_mat[i][14]+conf_mat[14][i]+conf_vec[i])
            Recall[i] = conf_mat[i][i]/(conf_mat[14][i] + conf_mat[i][i])
            if(conf_mat[i][i] + conf_mat[14][i] == 0):
                Recall[i] = 0
            else:
                Recall[i] = conf_mat[i][i]/(conf_mat[i][i] + conf_mat[14][i])
            if(conf_mat[i][i] + conf_mat[i][14] == 0):
                Precision[i] = 0
            else:
                Precision[i] = conf_mat[i][i]/(conf_mat[i][i] + conf_mat[i][14])
            BCR[i] = (Recall[i] + Precision[i])/2.0
        print("Accuracy:", end = ' ') 
        for i in range(14): 
            print("%.4f" % (Accuracy[i]), end = ',')
            aggre_accu += Accuracy[i]
        print("\nRecall:", end = ' ') 
        for i in range(14): 
            print("%.4f" % (Recall[i]), end = ',')
            aggre_recall += Recall[i]
        print("\nPrecision:", end = ' ') 
        for i in range(14): 
            print("%.4f" % (Precision[i]), end = ',')
            aggre_preci += Precision[i]
        print("\nBCR:", end = ' ') 
        for i in range(14): print("%.4f" % (BCR[i]), end = ',')
        print("Aggregated:", aggre_accu/14.0, aggre_recall/14.0, aggre_preci/14.0, (aggre_recall + aggre_preci)/28.0)
        print(conf_mat, conf_vec)
        for i in range(15):
            for j in range(15):
                conf_mat[i][j] /= (batch_size*0.14*cnt)
        print(conf_mat)
            
        plt.title('Baseline Training and Validation Loss')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        x_label = [i + 1 for i in range(len(total_val_loss))]
        total_loss2 = [total_loss[i*1000] for i in x_label]
        plt.plot(x_label, total_loss, label = "Training")
        plt.plot(x_label, total_val_loss, label = "Validation")
        plt.legend()
        plt.show()

model.eval()
tester(model, test_loader, batch_size) 

