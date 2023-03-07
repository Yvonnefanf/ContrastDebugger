import torch
import torch.nn as nn
import torch.optim as optim


class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MappingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
# Define the loss function
def mapping_loss(x,y,cka_similarity,nn_penalty):
        loss = cka_similarity(x,y) + nn_penalty(x,y)
        return loss
    
# Define the optimizer
# net = MappingNetwork(input_dim, output_dim)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# # Train the network
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         x,y = data
#         optimizer.zero_grad()

#         # Compute the loss
#         loss = mapping_loss(x,y, cka_similarity, nn_penalty)

#         # Backpropagate the error and update the parameters
#         loss.backward()
#         optimizer.step()

#         # Print statistics
#         running_loss += loss.item()
#         if i % 100 == 99:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0