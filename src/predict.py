import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import os
import json

from src.config import DATA_DIR

# Import các file khác của chúng ta
try:
    import config
    from model import MultimodalClassifier  # Import class mô hình
    from dataset import get_image_transforms  # Lấy lại hàm transform ảnh
except ImportError:
    print("Lỗi: Không thể import config, model, hoặc dataset.")
    print("Hãy đảm bảo bạn đang chạy file này từ thư mục gốc `ml_service` và đã export PYTHONPATH.")
    exit()


class Predictor:
    """
    Class này bao bọc toàn bộ logic dự đoán:
    1. Tải mô hình và các thành phần 1 LẦN DUY NHẤT.
    2. Cung cấp hàm `predict()` để dùng nhiều lần.
    """

    def __init__(self, model_path):
        print("--- Đang khởi tạo Predictor ---")

        # 1. Thiết lập Device (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")

        # 2. Tải bản đồ nhãn (Labels Map)
        try:
            with open(config.LABELS_MAP_PATH, 'r', encoding='utf-8') as f:
                self.labels_map = json.load(f)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file {config.LABELS_MAP_PATH}")
            exit()

        # Tạo bản đồ ngược (ID -> Tên nhãn)
        self.idx_to_label = {idx: label for label, idx in self.labels_map.items()}
        self.num_classes = len(self.labels_map)
        print(f"Đã tải {self.num_classes} nhãn.")

        # 3. Tải Tokenizer (cho text)
        print(f"Đang tải Tokenizer: {config.TEXT_MODEL_NAME}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        # 4. Tải Image Transform (cho ảnh)
        self.image_transform = get_image_transforms()

        # 5. Tải Mô hình
        print(f"Đang tải mô hình từ: {model_path}")
        # Khởi tạo mô hình (với backbone BỊ ĐÓNG BĂNG,
        # vì mô hình Giai đoạn 1 được train theo cách này)
        self.model = MultimodalClassifier(num_classes=self.num_classes, freeze_backbones=True)

        # Tải trọng số đã lưu
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file mô hình {model_path}")
            exit()

        # Chuyển mô hình lên GPU và đặt ở chế độ eval()
        self.model.to(self.device)
        self.model.eval()

        print("--- Predictor đã sẵn sàng! ---")

    def _process_text(self, text_content):
        """Mã hóa text đầu vào."""
        text_inputs = self.text_tokenizer(
            text_content,
            max_length=config.MAX_TOKEN_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return text_inputs['input_ids'], text_inputs['attention_mask']

    def _process_image(self, image_path):
        """Tải và xử lý ảnh đầu vào."""
        try:
            # Dùng logic đọc file 'rb' (binary) để tránh lỗi format
            with open(image_path, 'rb') as f:
                image_pil = Image.open(f)
                image = image_pil.convert('RGB')
        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy ảnh {image_path}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')
        except Exception as e:
            print(f"LỖI khi đọc ảnh {image_path}: {e}. Sử dụng ảnh rỗng.")
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='black')

        # Áp dụng transform và thêm 1 chiều "batch" (unsqueeze)
        return self.image_transform(image).unsqueeze(0)

    def predict(self, text_content, image_path, threshold=0.5):
        """
        Dự đoán nhãn cho một cặp (text, image).

        Args:
            text_content (str): Nội dung văn bản của bài đăng.
            image_path (str): Đường dẫn đến file ảnh.
            threshold (float): Ngưỡng (từ 0 đến 1) để quyết định 1 nhãn là 'True'.

        Returns:
            dict: Một dict chứa các nhãn được dự đoán và xác suất của chúng.
        """

        # 1. Xử lý đầu vào
        input_ids, attention_mask = self._process_text(text_content)
        image_tensor = self._process_image(image_path)

        # 2. Chuyển lên GPU
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        image_tensor = image_tensor.to(self.device)

        # 3. Chạy dự đoán (không cần gradient)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, image_tensor)

        # 4. Tính xác suất (Áp dụng Sigmoid)
        # logits có shape [1, 28] -> probabilities có shape [1, 28]
        probabilities = torch.sigmoid(logits)

        # Chuyển về CPU để xử lý
        probabilities = probabilities.cpu().numpy()[0]  # Lấy mảng 1D

        # 5. Quyết định nhãn
        results = {
            "predicted_labels": [],
            "all_probabilities": {}
        }

        for i in range(self.num_classes):
            label_name = self.idx_to_label[i]
            prob = probabilities[i]

            results["all_probabilities"][label_name] = float(prob)

            if prob >= threshold:
                results["predicted_labels"].append(label_name)

        return results


# --- DÙNG ĐỂ CHẠY KIỂM TRA TRỰC TIẾP ---
if __name__ == "__main__":

    # --- CẤU HÌNH ĐỂ THỬ NGHIỆM ---
    TEST_IMAGE_PATH = os.path.join(DATA_DIR, 'test/img.png')

    TEST_TEXT_CONTENT = "Darven đánh bại Rồng Nguyên Tố Lửa và thu thập được nhiều vật phẩm quý giá."

    # Ngưỡng dự đoán
    PREDICTION_THRESHOLD = 0.5
    # --------------------------------

    # Đường dẫn đến mô hình Giai đoạn 1
    MODEL_PATH = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')

    # 1. Khởi tạo Predictor (Tải mô hình 1 lần)
    try:
        predictor = Predictor(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Lỗi khi khởi tạo Predictor: {e}")
        import traceback

        traceback.print_exc()
        exit()

    # 2. Chạy dự đoán
    print(f"\n--- Đang dự đoán cho ---")
    print(f"Ảnh: {TEST_IMAGE_PATH}")
    print(f"Text: {TEST_TEXT_CONTENT}")

    if not os.path.exists(TEST_IMAGE_PATH):
        print("\nLỖI: Đường dẫn TEST_IMAGE_PATH không tồn tại. Vui lòng sửa lại.")
    else:
        results = predictor.predict(
            text_content=TEST_TEXT_CONTENT,
            image_path=TEST_IMAGE_PATH,
            threshold=PREDICTION_THRESHOLD
        )

        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"(Với ngưỡng = {PREDICTION_THRESHOLD})")
        print(f"\nCác nhãn được dự đoán:")
        if results["predicted_labels"]:
            for label in results["predicted_labels"]:
                print(f"  - {label} (Score: {results['all_probabilities'][label]:.4f})")
        else:
            print("  (Không có nhãn nào vượt ngưỡng)")

        print("\n--- (Chi tiết tất cả xác suất) ---")
        # Sắp xếp để xem các nhãn có score cao nhất
        sorted_probs = sorted(results["all_probabilities"].items(), key=lambda item: item[1], reverse=True)
        for label, prob in sorted_probs[:5]:  # In ra 5 nhãn cao nhất
            print(f"  {label}: {prob:.4f}")