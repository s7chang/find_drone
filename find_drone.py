import subprocess
import cv2
import os
import yaml
import json
import shutil
import sys
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

    def fine_tune_yolo(self, data_yaml, project, name, epochs=30, batch_size=32, optimizer='AdamW', patience=5, img_size=832, custom_best_model_path=None):
        model = YOLO(self.pre_trained_model_name)
        model.train(
            data=data_yaml,
            epochs=epochs,
            project=project,
            name=name,
            exist_ok=False,
            seed=42,
            optimizer=optimizer,
            patience=patience,         # Early stopping patience
            batch=batch_size,
            imgsz=img_size,
            degrees=0.15,
            fliplr=0,
            save=True,                 # Ensure best model is saved
            save_period=-1,            # Save only the best model
        )

        # 실제 저장된 디렉토리 경로 확인
        actual_save_dir = model.trainer.args.save_dir
        best_model_path = os.path.join(actual_save_dir, "weights", "best.pt")
        
        # 사용자 정의 경로로 이동 (필요 시)
        if self.path_best_model:
            if os.path.exists(best_model_path):
                shutil.move(best_model_path, self.path_best_model)
                print(f"Best model moved to: {self.path_best_model}")
            else:
                print(f"Default best model not found at: {best_model_path}")
        else:
            print(f"Best model saved at: {best_model_path}")

    def detect_and_draw(self, video_path):
        # 1. 비디오 파일 존재 여부 확인
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return

        # 2. 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file at {video_path}")
            return

        # 3. 비디오 속성 확인
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0 or width == 0 or height == 0:
            print("Error: Video file properties could not be determined. Check if the file is corrupted.")
            cap.release()
            return

        print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")

        # 4. 비디오 코덱 설정 및 확인
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        codec_test_path = "codec_test.mp4"
        codec_test = cv2.VideoWriter(codec_test_path, fourcc, fps, (width, height))

        if not codec_test.isOpened():
            print("Error: Required codec (mp4v) is not installed or supported.")
            cap.release()
            return
        else:
            print("Codec test passed. Codec is supported.")
            codec_test.release()
            if os.path.exists(codec_test_path):
                os.remove(codec_test_path)

        # 5. YOLO 모델 로드
        if not os.path.exists(self.path_best_model):
            print(f"Error: Best model file not found at {self.path_best_model}")
            cap.release()
            return

        model = YOLO(self.path_best_model)
        out = cv2.VideoWriter(self.path_output_video, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 현재 처리된 프레임 인덱스와 총 프레임 수 계산
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 진행 상황 출력
            progress = f"Processing frame {current_frame}/{total_frames} ({(current_frame/total_frames)*100:.2f}%)"
            sys.stdout.write("\r" + progress)  # \r로 줄 바꿈 없이 같은 줄에 덮어씌움
            sys.stdout.flush()

            # YOLO 모델로 예측
            results = model.predict(source=frame, stream=True)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"conf: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # 프레임 저장
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to {self.path_output_video}")
