import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """
    Class để đọc dataset ảnh từ các thư mục con.
    Mỗi thư mục con là một class (nhãn).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.label_list = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Quét qua các thư mục con để lấy đường dẫn ảnh và nhãn
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.label_list.append(self.class_to_idx[cls_name])

    def __len__(self):
        # Trả về tổng số lượng ảnh
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Lấy một mẫu dữ liệu (ảnh và nhãn) tại vị trí idx
        img_path = self.image_paths[idx]
        label = self.label_list[idx]
        
        # Mở ảnh bằng PIL
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Lỗi khi đọc ảnh: {img_path} - {e}")
            # Trả về ảnh rỗng nếu lỗi
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Áp dụng các phép biến đổi (transform)
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    # Dùng để kiểm tra nhanh xem class chạy đúng không
    # (Bạn không cần chạy file này, chỉ cần chạy train.py)
    print("Đang kiểm tra MyDataSet...")
    
    # Giả sử bạn có transform
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # Test thử với thư mục train (ví dụ)
    # Thay './data/fer2013_aligned/train' bằng đường dẫn thực tế nếu khác
    try:
        train_dataset = MyDataSet(root_dir='./data/fer2013_aligned/train', transform=transform)
        print(f"Tổng số lớp: {len(train_dataset.classes)}")
        print(f"Các lớp tìm thấy: {train_dataset.class_to_idx}")
        print(f"Tổng số ảnh: {len(train_dataset)}")
        
        # Lấy thử 1 ảnh
        img, label = train_dataset[0]
        print(f"Kích thước ảnh đầu tiên: {img.shape}")
        print(f"Nhãn ảnh đầu tiên: {label}")
        print("Kiểm tra MyDataSet thành công!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy thư mục. Hãy đảm bảo đường dẫn root_dir đúng.")
    except Exception as e:
        print(f"Lỗi khi khởi tạo MyDataSet: {e}")