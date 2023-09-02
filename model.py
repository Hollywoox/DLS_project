import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import cv2


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
          x = layer(x)

          if str(layer_num) in self.chosen_features:
            features.append(x)

        return features
    

def calculate_loss(g_features, o_features, s_features, alpha, betha):
    style_loss = original_loss = 0
    for gen_feature, orig_feature, style_feature in zip(
        g_features, o_features, s_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        G = gen_feature.view(channel, height*width).mm(
          gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )

        style_loss += torch.mean((G - A)**2)

    total_loss = alpha * original_loss + betha * style_loss
    return total_loss
    

def load_image(image_name, loader, device):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)



def main(source, style, count = 0, epochs = 200, image_size = 256):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(device)

    loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[], std=[])
        ]
    )

    original_img = load_image(source, loader, device)
    style_img = load_image(style, loader, device)

    generated = original_img.clone().requires_grad_(True)
    model = VGG().to(device).eval()

    total_steps = epochs
    lr = 0.001
    alpha = 1
    betha = 0.01
    optimizer = optim.Adam([generated], lr=lr)

    for step in range(total_steps):
        generated_features =  model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        total_loss = calculate_loss(generated_features, original_img_features, 
                                    style_features, alpha, betha)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step %  200 == 0:
            print(total_loss)
            save_image(generated, f"generated{count}.png")

    original = cv2.imread(source)

    generated = cv2.imread(f"generated{count}.png")
    generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
    cv2.imwrite(f"result{count}t.png", generated)

    return f"result{count}t.png"