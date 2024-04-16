import argparse
import h5py as h5
import numpy as np
from pandas import read_csv
import pickle as pkl
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from utils import DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--task-name", "-t", type=str, required=True)
args = parser.parse_args()

TASK_NAME = args.task_name

PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data/"
SAVE_DATA_PATH = Path(DATA_DIR) / "processed_data_pkl_aa/"

camera_indices = [1, 2, 51, 52]
img_size = (128, 128)
NUM_DEMOS = None

# Create the save path
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path(f"{PROCESSED_DATA_PATH}/{TASK_NAME}")

if (SAVE_DATA_PATH / f"{TASK_NAME}.pkl").exists():
    print(f"Data for {TASK_NAME} already exists. Appending to it...")
    input("Press Enter to continue...")
    data = pkl.load(open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "rb"))
    observations = data["observations"]
    max_cartesian = data["max_cartesian"]
    min_cartesian = data["min_cartesian"]
    max_gripper = data["max_gripper"]
    min_gripper = data["min_gripper"]
else:
    # Init storing variables
    observations = []

    # Store max and min
    max_cartesian, min_cartesian = None, None
    max_sensor, min_sensor = None, None
    # max_rel_cartesian, min_rel_cartesian = None, None
    max_gripper, min_gripper = None, None

# Load each data point and save in a list
dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
for i, data_point in enumerate(sorted(dirs)):
    use_sensor = True
    print(f"Processing data point {i+1}/{len(dirs)}")

    if NUM_DEMOS is not None:
        if int(str(data_point).split("_")[-1]) >= NUM_DEMOS:
            print(f"Skipping data point {data_point}")
            continue

    observation = {}
    # images
    image_dir = data_point / "videos"
    if not image_dir.exists():
        print(f"Data point {data_point} is incomplete")
        continue
    for save_idx, idx in enumerate(camera_indices):
        # Read the frames in the video
        video_path = image_dir / f"camera{idx}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Video {video_path} could not be opened")
            continue
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx == 52:
                # crop the right side of the image for the gripper cam
                shape = frame.shape
                crop_percent = 0.2
                frame = frame[:, : int(shape[1] * (1 - crop_percent))]
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
        if idx < 80:
            observation[f"pixels{idx}"] = np.array(frames)
        else:
            observation[f"digit{idx}"] = np.array(frames)
    # read cartesian and gripper states from csv
    state_csv_path = data_point / "states.csv"
    sensor_csv_path = data_point / "sensor.csv"
    state = read_csv(state_csv_path)
    try:
        sensor_data = read_csv(sensor_csv_path)
        sensor_states = sensor_data["sensor_values"].values
        sensor_states = np.array(
            [
                np.array([float(x.strip()) for x in sensor[1:-1].split(",")])
                for sensor in sensor_states
            ],
            dtype=np.float32,
        )
    except FileNotFoundError:
        use_sensor = False
        print(f"Sensor data not found for {data_point}")

    # Read cartesian state where every element is a 6D pose
    # Separate the pose into values instead of string
    cartesian_states = state["pose_aa"].values
    cartesian_states = np.array(
        [
            np.array([float(x.strip()) for x in pose[1:-1].split(",")])
            for pose in cartesian_states
        ],
        dtype=np.float32,
    )

    gripper_states = state["gripper_state"].values.astype(np.float32)
    observation["cartesian_states"] = cartesian_states.astype(np.float32)
    observation["gripper_states"] = gripper_states.astype(np.float32)
    if use_sensor:
        observation["sensor_states"] = sensor_states.astype(np.float32)
        if max_sensor is None:
            max_sensor = np.max(sensor_states)
            min_sensor = np.min(sensor_states)
        else:
            max_sensor = np.maximum(max_sensor, np.max(sensor_states))
            min_sensor = np.minimum(min_sensor, np.min(sensor_states))
        max_sensor = np.max(sensor_states, axis=0)
        min_sensor = np.min(sensor_states, axis=0)

    # update max and min
    if max_cartesian is None:
        max_cartesian = np.max(cartesian_states, axis=0)
        min_cartesian = np.min(cartesian_states, axis=0)
    else:
        max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
        min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))
    if max_gripper is None:
        max_gripper = np.max(gripper_states)
        min_gripper = np.min(gripper_states)
    else:
        max_gripper = np.maximum(max_gripper, np.max(gripper_states))
        min_gripper = np.minimum(min_gripper, np.min(gripper_states))

    # append to observations
    observations.append(observation)

# Save the data
data = {
    "observations": observations,
    "max_cartesian": max_cartesian,
    "min_cartesian": min_cartesian,
    "max_gripper": max_gripper,
    "min_gripper": min_gripper,
    "max_sensor": max_sensor,
    "min_sensor": min_sensor,
}
pkl.dump(data, open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "wb"))
