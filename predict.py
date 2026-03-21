from model import EmotionClassifier
from torchvision import transforms
from PIL import Image
import cv2
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

model = EmotionClassifier(in_channels=1, num_classes=7)
model.load_state_dict(torch.load("models/emotion_classifier.pt", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # adjust index if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 🔥 Crop face
            face = gray[y:y+h, x:x+w]

            # Convert to PIL
            face_pil = Image.fromarray(face)

            # Apply transform
            x_input = transform(face_pil).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                logits = model(x_input)
                pred = logits.argmax(dim=1).item()

            label = emotion_labels[pred]

            # Display label
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()