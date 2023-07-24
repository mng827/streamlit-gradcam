import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def preprocess_image(img: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return transform(img.copy()).unsqueeze(0)

model = resnet50(weights=ResNet50_Weights.DEFAULT)
target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

st.title("Explainable AI using GradCAM")
st.subheader("Upload a file")

file = st.file_uploader("Upload a file", label_visibility="hidden")

if file is not None:
    nparr = np.frombuffer(file.getvalue(), np.byte)
    input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1]
    input_img = np.float32(input_img) / 255

    input_img = cv2.resize(input_img, (224, 224))
    st.image(input_img)

    input_tensor = preprocess_image(input_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    predictions = model(input_tensor)
    prob = torch.softmax(predictions, dim=1)

    k = 5
    topk_prob, topk_indices = torch.topk(prob, k, dim=1)
    topk_class_list = [ResNet50_Weights.DEFAULT.meta["categories"][topk_indices[0, i]]
                       for i in range(k)]

    st.subheader("Top 5 predictions: ")
    selected_option = st.selectbox("Top 5 predictions: ", topk_class_list, label_visibility="hidden")
    selected_index = topk_class_list.index(selected_option)

    targets = [ClassifierOutputTarget(topk_indices[0, selected_index])]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)

    st.image(visualization)
