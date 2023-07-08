import torch
import torchvision
from PIL import Image
from train import transform
from train import CatAndDogConvNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = CatAndDogConvNet()
checkpoint = torch.load('Checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

path = '../DB/cat_dog/train/cat/cat.100.jpg'
# path = '../DB/cat_dog/train/dog/dog.104.jpg'
image = Image.open(path).convert('RGB')
image = transform(image).float()
image = image.unsqueeze_(0)
image = image.to(device)

with torch.no_grad():
    model.eval()  
    output =model(image)
    index = output.data.cpu().numpy().argmax()
    classes = ['cat','dog']
    class_name = classes[index]
    print('class:',class_name)