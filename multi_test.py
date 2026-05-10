import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from main_model import HiFuse_Small as create_model
from facenet_pytorch import MTCNN
import glob


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    input_folder = "test10/surprise"
    output_folder = "results_test10/surprise"
    os.makedirs(output_folder, exist_ok=True)  

    num_classes = 7
    img_size = 224
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_paths:
        print(f"⚠️ Không tìm thấy ảnh nào trong thư mục: '{input_folder}'")
        return
    print(f"🔍 Tìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")

    mtcnn = MTCNN(keep_all=False, device=device)

    json_path = "./class_indices.json"
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "model_weight/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()


    for img_path in image_paths:
        try:
            print(f"\nProcessing: {os.path.basename(img_path)}")
            img = Image.open(img_path).convert("RGB")

            face = mtcnn(img)
            if face is None:
                print(f"⚠️ Không tìm thấy khuôn mặt, dùng toàn bộ ảnh.")
                face = data_transform(img).unsqueeze(0).to(device)
            else:
                face = torch.nn.functional.interpolate(
                    face.unsqueeze(0), size=(224, 224), mode="bilinear"
                ).to(device)

            with torch.no_grad():
                output = torch.squeeze(model(face))
                predict = torch.softmax(output, dim=0).cpu()
                predict_cla = torch.argmax(predict).numpy()

            prediction_text = class_indict[str(predict_cla)]
            probability = predict[predict_cla].item()

            # --- TÍNH LOSS ---
            true_label_name = os.path.basename(input_folder)  
            true_label = int(
                [k for k, v in class_indict.items() if v == true_label_name][0]
            )
            target_tensor = torch.tensor([true_label], device=device)

            loss = criterion(output.unsqueeze(0), target_tensor)

            print(
                f"    -> Result: {prediction_text}, Accuracy: {probability:.3f}, Loss: {loss.item():.4f}"
            )

            # --- LƯU ẢNH ---
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"{prediction_text} ({probability:.3f})", fontsize=16)
            plt.axis("on")
            save_path = os.path.join(output_folder, os.path.basename(img_path))
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {img_path}: {e}")

    print(
        f"\n✅ Xử lý hoàn tất! Tất cả kết quả đã được lưu vào thư mục '{output_folder}'."
    )

if __name__ == "__main__":
    main()
