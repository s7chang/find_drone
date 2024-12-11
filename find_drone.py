import subprocess
import cv2
import os
import yaml
import json
from ultralytics import YOLO

class DroneFinding:
    def __init__(self, pre_trained_model_name, path_best_model, path_output_video):
        self.pre_trained_model_name = pre_trained_model_name
        self.path_best_model = path_best_model
        self.path_output_video = path_output_video

    @staticmethod
    def download_youtube_videos(url_list, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        video_paths = []
        for idx, url in enumerate(url_list):
            video_file = os.path.join(output_folder, f"video_{idx + 1}.mp4")
            command = [
                "yt-dlp",
                "-o", video_file,
                "-f", "bestvideo+bestaudio",
                "--merge-output-format", "mp4",
                url
            ]
            subprocess.run(command, check=True)
            video_paths.append(video_file)

        return video_paths

    @staticmethod
    def extract_frames_from_videos(video_paths, frames_output_folder, interval=1):
        if not os.path.exists(frames_output_folder):
            os.makedirs(frames_output_folder)

        for idx, video_path in enumerate(video_paths):
            video_folder = os.path.join(frames_output_folder, f"video_{idx + 1}")
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)

            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            extracted_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % (fps * interval) == 0:
                    frame_filename = os.path.join(video_folder, f"frame_{extracted_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    extracted_count += 1

                frame_count += 1

            cap.release()
            print(f"Extracted {extracted_count} frames to {video_folder}")

    @staticmethod
    def create_yaml_file(train_path, val_path, test_path, nc, class_names, yaml_path):
        data = {
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "nc": nc,
            "names": class_names
        }
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file)
        print(f"YAML file created at {yaml_path}")


    @staticmethod
    def create_json_file(data, json_path):
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"JSON file created at {json_path}")

    def fine_tune_yolo(self, data_yaml, project, name, epochs=30, batch_size=32, optimizer='AdamW', patience=5, img_size=832):
        model = YOLO(self.pre_trained_model_name)
        model.train(
            data=data_yaml,
            epochs=epochs,
            project=project,
            name=name,
            exist_ok=False,
            seed=42,
            optimizer=optimizer,
            patience=patience,
            batch=batch_size,
            imgsz=img_size,
            degrees=0.15,
            fliplr=0
        )
        print(f"Model fine-tuned and saved under project {project}, experiment {name}")

    def detect_and_draw(self, video_path):
        model = YOLO(self.path_best_model)
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.path_output_video, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, stream=True)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{int(cls)}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow('Drone Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to {self.path_output_video}")
