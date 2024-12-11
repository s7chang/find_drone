# set_yolo.py
from find_drone import DroneFinding

# image_video_urls	= [
#     'https://www.youtube.com/watch?v=sUL386RYrgM',
#     'https://www.youtube.com/watch?v=DOWDNBu9DkU',
#     'https://www.youtube.com/watch?v=N_XneaFmOmU',
#     'https://www.youtube.com/watch?v=i8VUOslJISM'
# ]
# test_video_url		= 'https://www.youtube.com/watch?v=EEXI6r08908'


# Define a class to encapsulate the functions
class DroneSetup:
    def __init__(self, dir_data_set, pre_trained_model_name='yolo11n.pt', class_names=['drone']):
        self.dir_data_set = dir_data_set
        self.pre_trained_model_name = pre_trained_model_name
        self.class_names = class_names

        # Paths
        self.path_data_yaml = f"{dir_data_set}data.yaml"
        self.path_test_video = f"{dir_data_set}test_video.mp4"
        self.path_best_model = f"{dir_data_set}best.pt"
        self.path_output_video = f"{dir_data_set}output_video.mp4"

        self.dir_train = f"{dir_data_set}train/"
        self.dir_val = f"{dir_data_set}valid/"
        self.dir_test = f"{dir_data_set}test/"

        self.dir_downloaded_videos = f"{dir_data_set}downloaded_videos/"
        self.dir_test_video_dir = f"{dir_data_set}test_video/"
        self.dir_frames = f"{dir_data_set}frames/"

        # Initialize DroneFinding
        self.drone_finding = DroneFinding(self.pre_trained_model_name, self.path_best_model, self.path_output_video)

    def download_training_videos(self, image_video_urls):
        return self.drone_finding.download_youtube_videos(image_video_urls, self.dir_downloaded_videos)

    def extract_frames(self, videos):
        self.drone_finding.extract_frames_from_videos(videos, self.dir_frames)

    def download_test_video(self, test_video_url):
        return self.drone_finding.download_youtube_videos([test_video_url], self.dir_test_video_dir)[0]

    def create_yaml(self):
        self.drone_finding.create_yaml_file(
            train_path=self.dir_train,
            val_path=self.dir_val,
            test_path=self.dir_test,
            nc=len(self.class_names),
            class_names=self.class_names,
            yaml_path=self.path_data_yaml,
        )

    def fine_tune_model(self, epochs=30, batch_size=32, optimizer='AdamW', patience=5, img_size=832):
        self.drone_finding.fine_tune_yolo(
            data_yaml=self.path_data_yaml,
            project=f"{self.dir_data_set}project",
            name='drone_tracking_exp',
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            patience=patience,
            img_size=img_size,
        )

    def detect_and_draw(self):
        self.drone_finding.detect_and_draw(self.path_test_video)
