from darts import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 데이터셋 설정
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2, pin_memory = True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2, pin_memory = True)

op_types = ['1x3_3x1_conv',
            '3x3_avgpool',
            '3x3_maxpool',
            '3x3_conv',
            '3x3_sepconv',
            '5x5_sepconv',
            '3x3_dilconv',
            '5x5_dilconv',
            'identity',
            'zero']


N = 7
input_size = 3
output_size = 10
BackBone = [1, 1, 1]  # fixed macro structure
init_channels = 8

darts = DARTS(
    op_types=op_types,
    N=N,
    input_size=input_size,
    output_size=output_size,
    BackBone=BackBone,
    init_channels=init_channels,
    train_loader=trainloader,
    test_loader=testloader
)

w_learning_rate = 0.025
alpha_learning_rate = 0.001
eta = 0.01
epsilon = 0.01
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, criterion, data_loader):
  model.eval()
  total_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      total_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

  accuracy = 100. * correct / total
  avg_loss = total_loss / len(data_loader)
  return avg_loss, accuracy

print("Starting architecture search...")
model = darts.search_continuous_cells(
    w_learning_rate=w_learning_rate,
    alpha_learning_rate=alpha_learning_rate,
    eta=eta,
    epsilon=epsilon,
    epochs=epochs
)

criterion = nn.CrossEntropyLoss().to(device)
test_loss, test_acc = evaluate(model, criterion, testloader)
print(f"\nFinal Test Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

discrete_model = darts.get_discrete_model(model)
print(darts.discrete_ops)