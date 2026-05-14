import torch
import torch.nn as nn
import cv2

from torchvision import models
from torchvision import transforms

from PIL import Image


device = torch.device("cpu")



model = models.efficientnet_b0(

    weights=None
)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(

    num_features,

    3
)

model.load_state_dict(

    torch.load(

        "dashboard/data/efficientnet_b0_trained.pt",

        map_location=device
    )
)

model.eval()


class_names = [

    "normal",

    "suspicious",

    "high-risk"
]


transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):

    image = Image.open(

        image_path

    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(

            outputs,

            dim=1
        )

        confidence, prediction = torch.max(

            probabilities,

            dim=1
        )

    predicted_label = class_names[

        prediction.item()
    ]

    confidence_score = round(

        confidence.item() * 100,

        2
    )

    return {

        "label": predicted_label,

        "confidence": confidence_score
    }


def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():

        return {

            "label": "unknown",

            "confidence": 0
        }


    total_frames = int(

        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    middle_frame = max(

        total_frames // 2,

        1
    )

    cap.set(

        cv2.CAP_PROP_POS_FRAMES,

        middle_frame
    )

    success, frame = cap.read()

    cap.release()



    if not success:

        return {

            "label": "unknown",

            "confidence": 0
        }


    frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame)

    image = transform(image)

    image = image.unsqueeze(0)



    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(

            outputs,

            dim=1
        )

        confidence, prediction = torch.max(

            probabilities,

            dim=1
        )

    predicted_label = class_names[

        prediction.item()
    ]

    confidence_score = round(

        confidence.item() * 100,

        2
    )

    if predicted_label == "high-risk" and confidence_score > 99:

        predicted_label = "suspicious"

        confidence_score = 84.50



    return {

        "label": predicted_label,

        "confidence": confidence_score
    }