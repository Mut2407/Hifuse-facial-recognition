import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from main_model import HiFuse_Small as create_model
from facenet_pytorch import MTCNN

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    num_classes = 7
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img_path = "distest.jpg"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."

    img = Image.open(img_path).convert("RGB")
    mtcnn = MTCNN(keep_all=False, device=device)
    face = mtcnn(img)
    if face is None:
        print("⚠️ No face detected, using full image.")
        face = data_transform(img).unsqueeze(0).to(device)
    else:
        face = torch.nn.functional.interpolate(face.unsqueeze(0), size=(224, 224), mode='bilinear').to(device)

    json_path = './class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "model_weight/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(face)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print(f"Prediction: {class_indict[str(predict_cla)]}, prob: {predict[predict_cla]:.3f}")
    plt.imshow(img)
    plt.title(f"{class_indict[str(predict_cla)]} ({predict[predict_cla]:.3f})")
    plt.show()

if __name__ == '__main__':
    main()
