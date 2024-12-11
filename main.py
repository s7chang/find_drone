# main.py
from find_drone import DroneFinding

# User-defined variables
pre_trained_model_name	= 'yolo11n.pt'

# Directory and dataset definitions
dir_data_set			= 'D:/dataset/find_drone.v1i.yolov11/'
image_video_urls		= [
	'https://www.youtube.com/watch?v=sUL386RYrgM',
	'https://www.youtube.com/watch?v=DOWDNBu9DkU',
	'https://www.youtube.com/watch?v=N_XneaFmOmU',
	'https://www.youtube.com/watch?v=i8VUOslJISM'
]

test_video_url			= 'https://www.youtube.com/watch?v=EEXI6r08908'

path_data_yaml			= dir_data_set + 'data.yaml'
path_test_video			= dir_data_set + 'test_video.mp4'
path_best_model			= dir_data_set + 'best.pt'
path_output_video		= dir_data_set + 'output_video.mp4'

# 이미지 폴더 경로
dir_train				= dir_data_set + 'train/'
dir_val					= dir_data_set + 'valid/'
dir_test				= dir_data_set + 'test/'

dir_downloaded_videos	= dir_data_set + 'downloaded_videos/'
dir_test_video			= dir_data_set + 'test_video/'
dir_frames				= dir_data_set + 'frames/'

# Define class names for dataset
class_names				= ['drone']

if __name__ == '__main__':
	# Initialize the DroneTracking class
	drone_finding			= DroneFinding(pre_trained_model_name, path_best_model, path_output_video)

	# # Download videos for training
	# image_videos			= drone_finding.download_youtube_videos(image_video_urls, dir_downloaded_videos)
	#
	# # Extract frames from downloaded videos
	# drone_finding.extract_frames_from_videos(image_videos, dir_frames)

	# # Download and process the test video
	# test_video_path			= drone_finding.download_youtube_videos([test_video_url], dir_test_video)[0]

	# # Create YAML file for training
	# drone_finding.create_yaml_file(
	# 	train_path			= dir_train,
	# 	val_path			= dir_val,
	# 	test_path			= dir_test,
	# 	nc					= len(class_names),
	# 	class_names			= class_names,
	# 	yaml_path			= path_data_yaml,
	# )

	# Fine-tune the YOLO model
	drone_finding.fine_tune_yolo(
		data_yaml			= path_data_yaml,
		project				= dir_data_set + 'project',
		name				= 'drone_tracking_exp',
		epochs				= 30,
		batch_size			= 32,
		optimizer			= 'AdamW',
		patience			= 5,
		img_size			= 832,
	)

	drone_finding.detect_and_draw(path_test_video)
