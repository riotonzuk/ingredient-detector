from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from utils import and_syntax

INPUT_SIZE = 224
TRANSFORM = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor()
     ])

POSITIVE_THRESH = 0.5
NOISE_SCALE = 5
NUM_PICS = 3

def predict_ingredients(image_path, model):
    with torch.no_grad():
        # if ingredient predictions are given, enter image reconstruction mode
        lst = []
        # open the image
        image = TRANSFORM(Image.open(image_path).convert("RGB")).unsqueeze(0)
        # grab the predicted labels from the model
        ingredients_pred = model.encoder(image)
        return ingredients_pred


def binarize(ingredients_pred):
    return (F.sigmoid(ingredients_pred) >= POSITIVE_THRESH).int().squeeze(0)

def soft(ingredients_pred):
    return F.sigmoid(ingredients_pred).squeeze(0)


def reconstruct_images(ingredients_pred, model):
    with torch.no_grad():
        repeated_preds = ingredients_pred.repeat(NUM_PICS, 1)
        noised_preds = repeated_preds + NOISE_SCALE * torch.rand(repeated_preds.shape)
        linear_preds = model.linear(noised_preds)
        reshaped_preds = linear_preds.reshape(-1, 1, 32, 32)
        reconstructed_images = (model.decoder(reshaped_preds) * 255)
        return reconstructed_images

def reconstruct_images_gan(ingredients_pred_binary, model):
    with torch.no_grad():
        repeated_preds = ingredients_pred_binary.repeat(NUM_PICS, 1).float()
        reconstructed_images = model((NOISE_SCALE * torch.rand(repeated_preds.shape), repeated_preds)) * 255
        return reconstructed_images
    
    
def get_labels(pred):
    file = open('IngredientList.txt', 'r')
    label_names = [line.rstrip() for line in file.readlines()]

    lst = []

    # print the predicted ingredient names
    for i in range(len(pred)):
        if pred[i] == 1:
            lst.append(label_names[i])

    return lst

