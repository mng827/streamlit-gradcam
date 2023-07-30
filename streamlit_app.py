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


def read_and_resize_img(filename, size):
    img = cv2.imread(filename)[:, :, ::-1]
    img = np.float32(img) / 255
    img = cv2.resize(img, size)
    return img


model = resnet50(weights=ResNet50_Weights.DEFAULT)
target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

st.title("Explainable AI using GradCAM")

st.write("We use a ResNet-50 model trained on the ImageNet-1K dataset. For "
         "the list of all labels, see [here]"
         "(https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/).")

st.subheader("Choose a picture below or upload your own file")

if 'image_selected' not in st.session_state:
    st.session_state['image_selected'] = -1

def set_image_selected(i):
    st.session_state['image_selected'] = i

def on_file_changed():
    st.session_state['image_selected'] = -1


input_img = None
file = None

sample_images = ["images/dog_cat.png",
                 "images/hotdog_cheeseburger.jpg",
                 "images/scuba_diver_coral_reef.jpg"]

col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    img = read_and_resize_img(sample_images[0], (224, 224))
    st.image(img, use_column_width="always")
    st.button("Select", "select_image0_button", use_container_width=True,
              on_click=set_image_selected, args=[0])

with col2:
    img = read_and_resize_img(sample_images[1], (224, 224))
    st.image(img, use_column_width="always")
    st.button("Select", "select_image1_button", use_container_width=True,
              on_click=set_image_selected, args=[1])

with col3:
    img = read_and_resize_img(sample_images[2], (224, 224))
    st.image(img, use_column_width="always")
    st.button("Select", "select_image2_button", use_container_width=True,
              on_click=set_image_selected, args=[2])

with col4:
    file = st.file_uploader("Upload a file", label_visibility="hidden",
                            on_change=on_file_changed)


if st.session_state['image_selected'] > -1:
    image_selected = st.session_state['image_selected']
    input_img = read_and_resize_img(sample_images[image_selected], (224, 224))

elif file is not None:
    nparr = np.frombuffer(file.getvalue(), np.byte)
    input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1]
    input_img = np.float32(input_img) / 255
    input_img = cv2.resize(input_img, (224, 224))
    st.image(input_img)


if input_img is not None:
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
