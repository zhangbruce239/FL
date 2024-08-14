import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

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

    # 添加附件
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
        

# 创建用于存储输出数据的新文件夹
folder_name = 'FL_alpha0.1_4'
print(folder_name)
folder_path = os.path.join(".", folder_name)
os.makedirs(folder_path, exist_ok=True)
start_time = time.time()


# 指定包含测试集和训练集的文件夹路径
data_folder_path = "DirichletDistribution0.1"

# 加载和处理数据集
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

# Define the neural network in each client
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training process
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

# Calculating number of samples in a client
def get_sample_counts(data_folder_path, i_range):
    sample_counts = []
    for i in i_range:
        csv_file = os.path.join(data_folder_path, f'train{i+1}.csv')
        df = pd.read_csv(csv_file)
        sample_counts.append(len(df))
    return sample_counts

# FedAvg Alogrithm
def average_weights(state_dicts, sample_counts):
    average_dict = {}
    total_samples = sum(sample_counts)
    for key in state_dicts[0].keys():
        key_params = []
        for state_dict, count in zip(state_dicts, sample_counts):
            key_params.append(state_dict[key] * count)
        weighted_sum = sum(key_params)

        # Calculating the average value, and store in average_dict
        average_dict[key] = weighted_sum / total_samples

    # return the aggregated dictionary
    return average_dict

# Evaluate the global model
def evaluate_model(model, test_data, criterion):
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0  
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
    print(total)
    return test_loss, accuracy

# Main function of FL
def federated_train(num_rounds=100): 
    global_model = CNN().to(device)  # 初始化全局模型
    client_models = [CNN().to(device) for _ in range(6)]  # 初始化客户端模型
    criterion = nn.CrossEntropyLoss()
    accuracy_list = []  # 在这里初始化accuracy_list
    training_losses = []
    testing_losses = []

    for round in range(num_rounds):
        global_state_dict = global_model.state_dict()
        
        # 为每个客户端模型加载特定的数据集并进行训练
        for client_index, model in enumerate(client_models):
            model.load_state_dict(global_state_dict)
            optimizer = optim.SGD(model.parameters(), lr = 0.0001)  # 创建新的优化器
            train_data = ImageDataset(csv_file=os.path.join( data_folder_path, f'train{client_index+1}.csv'), transform=transform)
            train_model(model, train_data, criterion, optimizer)

        
        # 在所有客户端模型上平均权重
        global_state_dict = average_weights([model.state_dict() for model in client_models], sample_counts = sample_counts)
        global_model.load_state_dict(global_state_dict)
    
        # 使用更新后的全局模型评估测试集性能
        test_data = ImageDataset(csv_file=os.path.join(data_folder_path, 'test.csv'), transform=transform)
        test_loss, accuracy = evaluate_model(global_model, test_data, criterion)  # 确保传入正确的参数
        testing_losses.append(test_loss)
        accuracy_list.append(accuracy)
        tqdm.write(f"\nRound {round+1}\nTest Accuracy: {accuracy}%")

    return accuracy_list, training_losses, testing_losses

# 弄清楚数量
sample_counts = get_sample_counts(data_folder_path,[0,1,2,3,4,5])


# 训练并评估联邦学习模型
accuracy_list, training_losses, testing_losses = federated_train(num_rounds=100)
accuracy_df = pd.DataFrame(accuracy_list, columns=["Accuracy"])
accuracy_df.to_csv(os.path.join(folder_path, 'accuracy.csv'), index=False)

training_df = pd.DataFrame(training_losses, columns=["Average Training Loss"])
training_df.to_csv(os.path.join(folder_path, 'training_loss.csv'), index=False)

test_df = pd.DataFrame(testing_losses, columns=["Testing Loss"])
test_df.to_csv(os.path.join(folder_path, 'testing_loss.csv'), index=False)

# 绘制准确率折线图
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(range(100), accuracy_list)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')
# plt.title('Dirichlet_6_Adam_lr4')
plt.savefig(os.path.join(folder_path, 'accuracy_plot.png'))
plt.close()  # 关闭当前的图形，避免绘图冲突

# 绘制训练损失和测试损失折线图
# plt.figure(figsize=(10, 6))  # 设置图像大小
# plt.plot(range(100), training_losses, label='Training Loss')
# plt.plot(range(100), testing_losses, label='Test Loss')
# plt.xlabel('Rounds')
# plt.ylabel('Loss')
# plt.title('Training and Test Losses over Federated Learning Rounds')
# plt.legend()
# plt.savefig(os.path.join(folder_path, 'loss_plot.png'))
# plt.close()  # 关闭当前的图形，以便于后续可以再次绘图
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
subject = f"程序已经运行完成 + {folder_name}"
body = f"用时：{elapsed_time_str}"
to_email = "zhangbruce239@gmail.com"
attachment_path = os.path.join(folder_path, "accuracy_plot.png")
send_email(subject, body, to_email, attachment_path)
