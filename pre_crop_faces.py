from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import os

def crop_faces(input_root="./data/fer2013", output_root="./data/fer2013_aligned", image_size=224):
    mtcnn = MTCNN(image_size=image_size, margin=20, keep_all=False)
    subsets = ["train", "val", "test"]

    for subset in subsets:
        input_dir = os.path.join(input_root, subset)
        output_dir = os.path.join(output_root, subset)
        os.makedirs(output_dir, exist_ok=True)

        for emotion in os.listdir(input_dir):
            in_class = os.path.join(input_dir, emotion)
            out_class = os.path.join(output_dir, emotion)
            os.makedirs(out_class, exist_ok=True)

            img_list = [f for f in os.listdir(in_class) if f.lower().endswith(('.jpg', '.png'))]
            for img_name in tqdm(img_list, desc=f"🖼 {subset}/{emotion}"):
                try:
                    img_path = os.path.join(in_class, img_name)
                    out_path = os.path.join(out_class, img_name)
                    img = Image.open(img_path).convert("RGB")
                    mtcnn(img, save_path=out_path)
                except Exception as e:
                    print(f"❌ {img_name}: {e}")

if __name__ == "__main__":
    crop_faces()
