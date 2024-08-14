import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def send_email(subject, body, to_email, attachment_path):
    from_email = "zhangboyuan202211@163.com"
    from_password = "IAGOLZTNCWCIKKMS"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Add attachment
    filename = os.path.basename(attachment_path)
    attachment = open(attachment_path, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {filename}")

    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.163.com', 25)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
    finally:
        attachment.close()

class CustomProgressBar:
    def __init__(self, total):
        self.total = total
        self.current = 0

    def update(self, step=1):
        if self.current >= self.total:
            self.current = 0
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()
        self.current += step
        self.display()

    def display(self):
        progress = self.current / self.total * 100
        sys.stdout.write(f'\rProgress: [{self.current}/{self.total}] {progress:.2f}% ')
        sys.stdout.flush()

# Create a new folder to store output data
# folder_name = 'Dirichlet_6_Adam_lr5_FTL_test_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
folder_name = 'FTL_c2t2_alpha0.3_5'
folder_path = os.path.join(".", folder_name)
print(folder_name)
os.makedirs(folder_path, exist_ok=True)
start_time = time.time()

# Write running information to a file
file_content = f"{folder_name}"
file_path = os.path.join(folder_path, "codeRunningInfo.txt")

with open(file_path, "w") as file:
    file.write(file_content)

data_folder_path = "DirichletDistribution0.3"

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(20*32*32, 320)
        self.fc2 = nn.Linear(320, 8)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 20*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练CNN模型的函数
def train_model(model, train_data, criterion, optimizer):
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    return average_loss

def get_sample_counts(data_folder_path, i_range):
    sample_counts = []
    for i in i_range:
        csv_file = os.path.join(data_folder_path, f'train{i+1}.csv')
        df = pd.read_csv(csv_file)
        sample_counts.append(len(df))
    return sample_counts

def average_weights(state_dicts, sample_counts):
    average_dict = {}
    total_samples = sum(sample_counts)
    
    for key in state_dicts[0].keys():
        key_params = []
        for state_dict, count in zip(state_dicts, sample_counts):
            # print(state_dict[key])
            # print(count)
            key_params.append(state_dict[key] * count)
            
        weighted_sum = sum(key_params)
        # 计算加权平均值，并存入average_dict
        average_dict[key] = weighted_sum / total_samples

    # 返回计算得到的加权平均权重字典
    return average_dict

def evaluate_model(model, test_data, criterion):
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0  # 初始化total_loss
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

def federated_train(num_rounds=100):
    global_model = CNN().to(device) 
    global_model_2 = CNN().to(device)
    sub_models = [CNN().to(device) for _ in range(2)]  
    remaining_models = [CNN().to(device) for _ in range(4)]  
    criterion = nn.CrossEntropyLoss()
    accuracy_list = []
    training_losses = []
    testing_losses = []
    selected_clients = [0,1]
    sample_counts = get_sample_counts(data_folder_path,selected_clients)

    global_state_dict = global_model.state_dict()
    global_model_2.load_state_dict(global_state_dict)
    print("Step 1: Pre-training")
    # progress_bar = CustomProgressBar(100)
    # client_state_dicts = []
    for round in range(num_rounds):
        global_state_dict = global_model.state_dict()
        client_losses = []
        # Train each sub-model on its respective client for one epoch
        for i, client_index in enumerate(selected_clients):
            sub_models[i].load_state_dict(global_state_dict)
            optimizer = optim.SGD(sub_models[i].parameters(), lr=0.0001)  # 创建新的优化器
            train_data = ImageDataset(csv_file=os.path.join( data_folder_path, f'train{client_index+1}.csv'), transform=transform)
            train_loss = train_model(sub_models[i], train_data, criterion, optimizer)
            client_losses.append(train_loss)
            
        # 计算所有客户端的平均训练损失
        average_train_loss = sum(client_losses) / len(client_losses)
        training_losses.append(average_train_loss)

        # 在所有客户端模型上平均权重
        global_state_dict = average_weights([model.state_dict() for model in sub_models],sample_counts=sample_counts)
        global_model.load_state_dict(global_state_dict)
            
        test_data = ImageDataset(csv_file=os.path.join(data_folder_path, 'test.csv'), transform=transform)
        test_loss, accuracy = evaluate_model(global_model, test_data, criterion)  # Evaluate using the first model
        # testing_losses.append(test_loss)
        # accuracy_list.append(accuracy)
        print(f"\nRound {round+1}")
        print(f"Test Accuracy: {accuracy}%")
        # print(f"Average Training Loss: {average_train_loss}")
        # print(f"Test Loss: {test_loss}")
        # 更新进度条
        # progress_bar.update()

    # Step 2: Federated learning with clients 3, 4, 5 and 6
    
    print("Step2: Transfering and training")
    criterion = nn.CrossEntropyLoss()
    accuracy_list = []  # 在这里初始化accuracy_list
    training_losses = []
    testing_losses = []
    selected_clients = [2,3,4,5]
    sample_counts = get_sample_counts(data_folder_path,selected_clients)
    
    # 传递参数
    global_model_2.conv1 = global_model.conv1
    global_model_2.conv2 = global_model.conv2
    
    # 冻结参数
    for model in remaining_models:
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False

    # progress_bar = CustomProgressBar(4)
    for round in range(num_rounds):
        global_state_dict = global_model_2.state_dict()
        client_losses = []

        for i, client_index in enumerate(selected_clients):  # Clients 1345
            remaining_models[i].load_state_dict(global_state_dict)
            optimizer = optim.SGD(remaining_models[i].parameters(), lr=0.0001)  # 创建新的优化器
            # progress_bar.update()
            # 正确地为每个客户端加载其数据集
            train_data = ImageDataset(csv_file=os.path.join( data_folder_path, f'train{client_index+1}.csv'), transform=transform)
            train_loss = train_model(remaining_models[i], train_data, criterion, optimizer)
            client_losses.append(train_loss)
            
    
        average_train_loss = sum(client_losses) / len(client_losses)
        training_losses.append(average_train_loss)
    
        # 在所有客户端模型上平均权重
        global_state_dict = average_weights([model.state_dict() for model in remaining_models],sample_counts=sample_counts)
        global_model_2.load_state_dict(global_state_dict)

        test_data = ImageDataset(csv_file=os.path.join(data_folder_path, 'test.csv'), transform=transform)
        test_loss, accuracy = evaluate_model(global_model_2, test_data, criterion)  # Evaluate using the first model
        testing_losses.append(test_loss)
        accuracy_list.append(accuracy)
        print(f"\nRound {round+1}")
        print(f"Test Accuracy: {accuracy}%")
        print(f"Average Training Loss: {average_train_loss}")
        print(f"Test Loss: {test_loss}")

    return accuracy_list, training_losses, testing_losses

accuracy_list, training_losses, testing_losses = federated_train(num_rounds=100)
accuracy_df = pd.DataFrame(accuracy_list, columns=["Accuracy"])
accuracy_df.to_csv(os.path.join(folder_path, 'accuracy.csv'), index=False)

training_df = pd.DataFrame(training_losses, columns=["Average Training Loss"])
training_df.to_csv(os.path.join(folder_path, 'training_loss.csv'), index=False)

test_df = pd.DataFrame(testing_losses, columns=["Testing Loss"])
test_df.to_csv(os.path.join(folder_path, 'testing_loss.csv'), index=False)

plt.figure(figsize=(10, 6))
plt.plot(range(100), accuracy_list)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
plt.title('Dirichlet_6_FTL_SGD_lr4')
plt.savefig(os.path.join(folder_path, 'accuracy_plot.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(range(100), training_losses, label='Training Loss')
plt.plot(range(100), testing_losses, label='Test Loss')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.title('Training and Test Losses over Federated Learning Rounds')
plt.legend()
plt.savefig(os.path.join(folder_path, 'loss_plot.png'))
plt.close()

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

subject = f"程序已经运行完成 {folder_name}"
body = f"用时：{elapsed_time_str}"
to_email = "zhangbruce239@gmail.com"
attachment_path = os.path.join(folder_path, "accuracy_plot.png")
send_email(subject, body, to_email, attachment_path)