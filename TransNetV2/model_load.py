from .modules import TransNetV2
from typing import Tuple, Any
import numpy as np
import os
import sys


class TransNetV2ModelLoader:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = self.load_model()

    def load_model(self):
        self.model = TransNetV2(model_dir=self.model_dir)
        return self.model

    def predict_video_frames(self, video_path, save_path) -> Tuple[str, str, str]:
        # predict
        # 获取video_path的文件名
        file_name = os.path.basename(video_path)
        predictions_path = save_path + f"{file_name}.predictions.txt"
        scenes_path = save_path + f"{file_name}.scenes.txt"
        vis_path = save_path + f"{file_name}.vis.png"
        if os.path.exists(predictions_path) or os.path.exists(scenes_path):
            raise Exception(f"[TransNetV2] {scenes_path} or {scenes_path} already exists. ")
        if os.path.exists(vis_path):
            print(f"[TransNetV2] {vis_path} already exists. ")

        video_frames, single_frame_predictions, all_frame_predictions = \
            self.model.predict_video(video_path)

        predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
        np.savetxt(predictions_path, predictions, fmt="%.6f")

        scenes = self.model.predictions_to_scenes(single_frame_predictions)
        np.savetxt(scenes_path, scenes, fmt="%d")

        pil_image = self.model.visualize_predictions(
            video_frames, predictions=(single_frame_predictions, all_frame_predictions))
        pil_image.save(vis_path)
        return predictions_path, scenes_path, vis_path
