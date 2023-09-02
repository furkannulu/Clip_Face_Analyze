import numpy as np
import torch
import clip
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from deepface import DeepFace
import cv2


def sample_frames(path: str, freq: float=1.0) -> list[Image.Image]:
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []
    ret, frame = video.read()
    count = 0
    while ret:
        if count % (int(fps/freq)) == 0:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame_time = count / fps  # Hesaplama burada yapılıyor
            frames.append({"frame": Image.fromarray(frame), "time": frame_time})
 
        ret, frame = video.read()
        count = count + 1
    return frames

class ClipIndex:
    def __init__(self, model_name: str="ViT-L/14@336px"):#########++++++
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = clip.load(model_name, self.device)
        self.model.eval()

    def encode_image(self, frames: Union[Image.Image,list[Image.Image]]) -> torch.Tensor:
        """Indexes a frame and returns a tensor of frame features"""
        frames = [frames] if isinstance(frames, Image.Image) else frames
        frame_images = [frame_dict["frame"] for frame_dict in frames]
        preprocessed_frames = torch.stack([self.preprocess(frame) for frame in frame_images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed_frames).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        del preprocessed_frames

        return image_features.detach().cpu()
    
    def encode_text(self, query: list[str]) -> torch.Tensor:
        tokens = clip.tokenize(query).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens).float()   
            text_features /= text_features.norm(dim=-1, keepdim=True)
        del tokens

        return text_features.detach().cpu()

    def search(self, query: Union[str, list[str]], frames: List[Image.Image], threshold: int=15) -> torch.Tensor:
        """Gets a string and searches across frame features"""
        query = [query] if isinstance(query, str) else query
        
        # video_features = [self.encode_image(frame) for frame in frames]
        video_features = self.encode_image(frames)
        
        text_features = self.encode_text(query)

        similarities = 100.0 * text_features @ video_features.T
        sims = torch.where(similarities > threshold, 1, 0)
        return similarities, sims

def highest_similarities(similarities: Union[torch.Tensor, list[torch.Tensor]]) -> list[int]:
    similarities = [similarities] if isinstance(similarities, torch.Tensor) else similarities # is that necessary???
    highest_sim_values, highest_sim_indices = torch.topk(similarities[0], k=2, dim=1)
    # En yüksek benzerlik değerlerini numpy dizisi olarak al
    highest_sim_values = highest_sim_values.cpu().numpy()
    # En yüksek benzerlik değerlerine sahip indeksleri al
    highest_sim_indices = highest_sim_indices.cpu().numpy()
    # En yüksek benzerlik değerlerine sahip indeksleri frame indeksleri olarak döndür
    highest_sim_frame_indices = [frame_indices.tolist() for frame_indices in highest_sim_indices]
    return highest_sim_frame_indices


def get_video_paths(path: str) -> list[str]:
    folder_path = Path.cwd() / path
    video_paths = []
    for path in folder_path.glob("*.mp4"):
        video_paths.append(str(path))
    
    return video_paths

def detect_face(frame_arr: np.array):
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yunet']
    faces = DeepFace.extract_faces(img_path=frame_arr,
                                  target_size=(224,224),
                                  detector_backend=backends[3],
                                  enforce_detection=False)
    return faces


def face_analyze(frame_arr: np.array, category: str) -> list[dict]:
    categories = ['age','gender','race','emotion']
    obj = []
    if category in categories:
        obj = DeepFace.analyze(img_path=frame_arr, actions=[category],enforce_detection=False, detector_backend='mtcnn')
        return obj
    else:
        raise ValueError("Invalid Category")

# #TODO Test this function
# def verify_images(highest_sim_list: list[int], matching_indices: list, frames: List[Image.Image]):
#     verified = []
#     full_frame_arrays = [np.array(frames[idx]["frame"]) for idx in matching_indices]
#     high_frmae_arr = [np.array(frames[idx]["frame"]) for idx in highest_sim_list[0]]
#     for i in range(0,len(high_frmae_arr)):
#         for j in range(0,len(full_frame_arrays)):
#             result = DeepFace.verify(img1_path = high_frmae_arr[i] , img2_path = full_frame_arrays[j],enforce_detection=False)
#             if result['verified'] == True:
#                 verified.append(full_frame_arrays[j])
#                 print("blabla\n")
#             del(result)
#     #TODO verify done but appending in verified returns empty. Make sure that works.
#     return verified
def verify_images(highest_sim_list: list[int], matching_indices: list, frames: List[Image.Image]):
    verified = []
    full_frame_arrays = [np.array(frames[idx]["frame"]) for idx in matching_indices]
    high_frmae_arr = [np.array(frames[idx]["frame"]) for idx in highest_sim_list[0]]

    # full_frame_arrays = [cv2.cvtColor(frames[idx]["frame"],cv2.COLOR_BGR2RGB) for idx in full_frame_arrays]
    # high_frmae_arr = [cv2.cvtColor(frames[idx]["frame"],cv2.COLOR_BGR2RGB) for idx in high_frmae_arr]

    # frame = cv2.cvtColor(frames[idx]["frame"],cv2.COLOR_BGR2RGB)

    for i in range(len(high_frmae_arr)):
        for j in range(len(full_frame_arrays)):
            try:
                result = DeepFace.verify(img1_path=high_frmae_arr[i][:,:,::-1], img2_path=full_frame_arrays[j][:,:,::-1],enforce_detection=True)
                if result['verified'] == True:
                    verified.append(full_frame_arrays[j])
                    print("Face detected in frame")
                del(result)
            except Exception as e:
                print(f"\nFace not detected in frame because: {e}")
            

    return verified

def main():
    video_paths = get_video_paths("videos")

    indexer = ClipIndex()
    query = "a photo of a people's face" 
    for path in video_paths:
        frames = sample_frames(path, 1.0)

        similarities, sims = indexer.search(query, frames)
        matching_indices = torch.nonzero(sims[0]).squeeze().tolist()

        highest_sim_list = highest_similarities(similarities)

        #Uyumlu frame'leri göster
        plt.figure(figsize=(15, 10))

        for i, idx in enumerate(matching_indices):
            frame_arr = np.array(frames[idx]["frame"])
            plt.subplot(1, len(matching_indices), i+1)
            plt.imshow(frames[idx]["frame"])
            plt.axis('off')
        
        plt.show()

        verified = verify_images(highest_sim_list, matching_indices, frames)

        for frame in verified:
            plt.imshow(frame)

        categories = ['age', 'gender', 'race', 'emotion']
        for category in categories:
            faces = detect_face(frame_arr)
            result = face_analyze(faces[0]['face'], category)
            print(f"\n{category}: {result}\n\n")

if __name__ == "__main__":
    main()

