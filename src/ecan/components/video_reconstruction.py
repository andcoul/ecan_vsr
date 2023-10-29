import cv2
import os

# Directory containing video frames
frame_directory = 'results/Settings/nFr7_batch64_K20L10M20_ecan_lab_dpw3/vid4/calendar'

# Output video file
output_video = 'vid4_calendar_output_video.mp4'

# Get the list of frame filenames
frame_filenames = sorted([filename for filename in os.listdir(
    frame_directory) if filename.endswith('.png')])

# Read the first frame to get frame dimensions
first_frame = cv2.imread(os.path.join(frame_directory, frame_filenames[0]))
height, width, channels = first_frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Iterate through the frame filenames and write frames to the video
for frame_filename in frame_filenames:
    frame_path = os.path.join(frame_directory, frame_filename)
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

# Release the video writer and close the video file
video_writer.release()
