import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
To predict the species and disease of a plant:
- Place 256x256 image of leaf in images/images/ (found in the same dir as script).
- Run 'python3 predict_disease.py'
'''

# Defining list of possible model outcomes
DISEASES = ['Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy']

# Defining list of possible model outcomes (prettier format)
DISEASES_TUPLES = [('Apple', 'Scab'),
                   ('Apple', 'Black Rot'),
                   ('Apple', 'Cedar Apple Rust'),
                   ('Apple', 'Healthy'),
                   ('Blueberry', 'Healthy'),
                   ('Cherry', 'Powdery Mildew'),
                   ('Cherry', 'Healthy'),
                   ('Corn', 'Cercospora Leaf Spot - Gray Leaf Spot'),
                   ('Corn', 'Common Rust'),
                   ('Corn', 'Northern Leaf Blight'),
                   ('Corn', 'Healthy'),
                   ('Grape', 'Black Rot'),
                   ('Grape', 'Esca (Black Measles)'),
                   ('Grape', 'Leaf Blight (Isariopsis Leaf Spot)'),
                   ('Grape', 'Healthy'),
                   ('Orange', 'Haunglongbing (Citrus Greening'),
                   ('Peach', 'Bacterial Spot'),
                   ('Peach', 'Healthy'),
                   ('Bell Pepper', 'Bacterial Spot'),
                   ('Bell Pepper', 'Healthy'),
                   ('Potato', 'Early Blight'),
                   ('Potato', 'Late Blight'),
                   ('Potato', 'Healthy'),
                   ('Raspberry', 'Healthy'),
                   ('Soybean', 'Healthy'),
                   ('Squash', 'Powdery Mildew'),
                   ('Strawberry', 'Leaf Scorch'),
                   ('Strawberry', 'Healthy'),
                   ('Tomato', 'Bacterial Spot'),
                   ('Tomato', 'Early Blight'),
                   ('Tomato', 'Late Blight'),
                   ('Tomato', 'Leaf Mold'),
                   ('Tomato', 'Septoria Leaf Spot'),
                   ('Tomato', 'Spider Mites - Two Spotted Spider Mite'),
                   ('Tomato', 'Target Spot'),
                   ('Tomato', 'Yellow Leaf Curl Virus'),
                   ('Tomato', 'Mosaic Virus'),
                   ('Tomato', 'Healthy')]

# Model definition - removed unnecessary parts for simpler script

class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    pass

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))

    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def predict_image(img, model):
    '''
    Predicts images and returns a string that looks like: 'Apple___Apple_scab'
    '''
    my_img, _ = img;
    xb = my_img.unsqueeze(0).to(torch.device('cpu'), non_blocking=True)
    yb = model(xb)
    _, indices  = torch.max(yb, dim=1)
    return DISEASES[indices[0].item()]

def predict_image_pretty(img, model):
    '''
    Predicts images and returns a tuple that looks like: ('Apple', 'Scab')
    '''
    my_img, _ = img;
    xb = my_img.unsqueeze(0).to(torch.device('cpu'), non_blocking=True)
    yb = model(xb)
    _, indices  = torch.max(yb, dim=1)
    return DISEASES_TUPLES[indices[0].item()]

if __name__ == "__main__":
    # Load model
    model_path = './model.pth'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    # Load images
    images = ImageFolder('./images/', transform=transforms.ToTensor())
    # Predict image(s)
    print(predict_image_pretty(images[0], model))



