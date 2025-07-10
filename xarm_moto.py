import torch
import cv2
import time
from transformers import AutoTokenizer
import torchvision.transforms as T
from moto_gpt.model import MotoGPT  # 👈 Your actual model class
from moto_gpt.config import get_moto_gpt_config  # 👈 Your config loading function

# ------------------------
# Load model and tokenizer
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("[INFO] Loading model config...")
config = get_moto_gpt_config()  # <-- Implement or replace with OmegaConf.load or similar

print("[INFO] Initializing model...")
model = 
state_dict = torch.load("/workspace/Moto/moto_gpt/checkpoints/moto_gpt_finetuned_on_rt1/pytorch_model.bin", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ------------------------
# Preprocessing function
# ------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),  # Adjust to your model's input size
    T.ToTensor(),  # Converts to (C, H, W), values in [0,1]
])

# ------------------------
# Video Recorder Class
# ------------------------
class VideoRecorder:
    def __init__(self, device_id=0):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open.")

    def record_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Failed to capture frame.")
            return None
        return frame

# ------------------------
# Run live loop
# ------------------------
prompt = "pick up blue cube place on green plate"
video_recorder = VideoRecorder()

print("[INFO] Starting real-time inference loop.")
try:
    while True:
        frame = video_recorder.record_frame()
        if frame is None:
            continue

        # Preprocess image
        image_tensor = transform(frame).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

        # Tokenize instruction
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        # Inference
        with torch.no_grad():
            # Assumes model takes image + input_ids + attention_mask
            action = model(image_tensor, input_ids=input_ids, attention_mask=attention_mask)

        print("Action:", action)

        # Optional: display frame
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Exiting...")

finally:
    video_recorder.cap.release()
    cv2.destroyAllWindows()
