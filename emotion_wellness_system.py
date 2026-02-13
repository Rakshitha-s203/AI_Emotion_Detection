import cv2
from deepface import DeepFace
from collections import Counter
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

emoji_dict = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😨",
    "neutral": "😐",
    "disgust": "🤢"
}

recommendation_dict = {
    "happy": "Keep spreading positivity!",
    "sad": "Try listening to music",
    "angry": "Take a deep breath",
    "fear": "Stay calm",
    "surprise": "Enjoy the moment",
    "neutral": "Stay focused",
    "disgust": "Take a short break"
}

cap = cv2.VideoCapture(0)

emotion_buffer = []
buffer_size = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        current_emotion = result['dominant_emotion']
        confidence = result['emotion'][current_emotion]

        emotion_buffer.append(current_emotion)
        if len(emotion_buffer) > buffer_size:
            emotion_buffer.pop(0)

        stable_emotion = Counter(emotion_buffer).most_common(1)[0][0]

        emoji = emoji_dict.get(stable_emotion, "")
        suggestion = recommendation_dict.get(stable_emotion, "")

        # Convert OpenCV image to PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Use default font (for emoji)
        try:
            font = ImageFont.truetype("seguiemj.ttf", 40)  # Windows emoji font
        except:
            font = ImageFont.load_default()

        draw.text((40, 40), f"Emotion: {stable_emotion}", font=font, fill=(0, 255, 0))
        draw.text((40, 80), f"Confidence: {confidence:.2f}%", font=font, fill=(0, 0, 0))
        draw.text((40, 120), f"Suggestion: {suggestion}", font=font, fill=((128, 0, 128)))
        draw.text((40, 160), f"{emoji}", font=font, fill=((255, 255, 0)))

        # Convert back to OpenCV
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Wellness System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
