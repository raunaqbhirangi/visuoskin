import gym
from gym import spaces
import cv2
import numpy as np

# import pybullet
# import pybullet_data
import pickle
from scipy.spatial.transform import Rotation as R

from openteach.utils.network import create_request_socket, ZMQCameraSubscriber
from xarm_env.envs.constants import *


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    pos = cartesian[:3]
    ori = cartesian[3:]
    r = R.from_rotvec(ori)
    quat = r.as_quat()
    return np.concatenate([pos, quat], axis=-1)


class RobotEnv(gym.Env):
    def __init__(
        self,
        height=224,
        width=224,
        use_robot=True,  # True when robot used
        use_egocentric=False,  # True when egocentric camera used
        use_fisheye=True,
        sensor_type="reskin",
        subtract_sensor_baseline=False,
    ):
        super(RobotEnv, self).__init__()
        self.height = height
        self.width = width

        self.use_robot = use_robot
        self.use_fish_eye = use_fisheye
        self.use_egocentric = use_egocentric
        self.sensor_type = sensor_type
        self.digit_keys = ["digit80", "digit81"]

        self.subtract_sensor_baseline = subtract_sensor_baseline
        self.sensor_prev_state = None
        self.sensor_baseline = None

        self.feature_dim = 8  # 10  # 7
        self.proprio_dim = 8

        self.n_sensors = 2
        self.sensor_dim = 15
        self.action_dim = 7

        # Robot limits
        # self.cartesian_delta_limits = np.array([-10, 10])

        self.n_channels = 3
        self.reward = 0

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        if self.use_robot:
            # camera subscribers
            self.image_subscribers = {}
            for cam_idx in list(CAM_SERIAL_NUMS.keys()):
                port = CAMERA_PORT_OFFSET + cam_idx
                self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST_ADDRESS,
                    port=port,
                    topic_type="RGB",
                )

            # for fish_eye_cam_idx in range(len(FISH_EYE_CAM_SERIAL_NUMS)):
            if use_fisheye:
                for fish_eye_cam_idx in list(FISH_EYE_CAM_SERIAL_NUMS.keys()):
                    port = FISH_EYE_CAMERA_PORT_OFFSET + fish_eye_cam_idx
                    self.image_subscribers[fish_eye_cam_idx] = ZMQCameraSubscriber(
                        host=HOST_ADDRESS,
                        port=port,
                        topic_type="RGB",
                    )

            # action request port
            self.action_request_socket = create_request_socket(
                HOST_ADDRESS, DEPLOYMENT_PORT
            )

    def step(self, action):
        print("current step's action is: ", action)
        action = np.array(action)

        action_dict = {
            "xarm": {
                "cartesian": action[:-1],
                "gripper": action[-1:],
            }
        }

        # send action
        self.action_request_socket.send(pickle.dumps(action_dict, protocol=-1))
        ret = self.action_request_socket.recv()
        ret = pickle.loads(ret)
        if ret == "Command failed!":
            print("Command failed!")
            # return None, 0, True, None
            self.action_request_socket.send(b"get_state")
            ret = pickle.loads(self.action_request_socket.recv())
        #     robot_state = pickle.loads(self.action_request_socket.recv())["robot_state"]["xarm"]
        # else:
        #     # robot_state = ret["robot_state"]["xarm"]
        #     robot_state = ret["robot_state"]["xarm"]
        robot_state = ret["robot_state"]["xarm"]

        # cartesian_pos = robot_state[:3]
        # cartesian_ori = robot_state[3:6]
        # gripper = robot_state[6]
        # cartesian_ori_sin = np.sin(cartesian_ori)
        # cartesian_ori_cos = np.cos(cartesian_ori)
        # robot_state = np.concatenate(
        #     [cartesian_pos, cartesian_ori_sin, cartesian_ori_cos, [gripper]], axis=0
        # )
        cartesian = robot_state[:6]
        quat_cartesian = get_quaternion_orientation(cartesian)
        robot_state = np.concatenate([quat_cartesian, robot_state[6:]], axis=0)

        # subscribe images
        image_dict = {}
        for cam_idx, img_sub in self.image_subscribers.items():
            image_dict[cam_idx] = img_sub.recv_rgb_image()[0]

        obs = {}
        obs["features"] = np.array(robot_state, dtype=np.float32)
        obs["proprioceptive"] = np.array(robot_state, dtype=np.float32)
        if self.sensor_type == "reskin":
            try:
                sensor_state = ret["sensor_state"]["reskin"]["sensor_values"]
                sensor_state_sub = (
                    np.array(sensor_state, dtype=np.float32) - self.sensor_baseline
                )
                self.sensor_prev_state = sensor_state_sub
                sensor_keys = [
                    f"sensor{sensor_idx}" for sensor_idx in range(self.n_sensors)
                ]
                for sidx, sensor_key in enumerate(sensor_keys):
                    if self.subtract_sensor_baseline:
                        obs[sensor_key] = sensor_state_sub[
                            sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                        ]
                    else:
                        obs[sensor_key] = sensor_state[
                            sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                        ]
            except KeyError:
                pass
        elif self.sensor_type == "digit":
            for dkey in self.digit_keys:
                obs[dkey] = np.array(ret["sensor_state"][dkey])
                obs[dkey] = cv2.resize(obs[dkey], (self.width, self.height))
                if self.subtract_sensor_baseline:
                    obs[dkey] = obs[dkey] - self.sensor_baseline

        for cam_idx, image in image_dict.items():
            if cam_idx == 52:
                # crop the right side of the image for the gripper cam
                img_shape = image.shape
                crop_percent = 0.2
                image = image[:, : int(img_shape[1] * (1 - crop_percent))]
            obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
        return obs, self.reward, False, False, {}

    def reset(self, seed=None):  # currently same positions, with gripper opening
        if self.use_robot:
            print("resetting")
            self.action_request_socket.send(b"reset")
            reset_state = pickle.loads(self.action_request_socket.recv())

            # subscribe robot state
            self.action_request_socket.send(b"get_state")
            ret = pickle.loads(self.action_request_socket.recv())
            robot_state = ret["robot_state"]["xarm"]
            # robot_state = np.array(robot_state, dtype=np.float32)
            # cartesian_pos = robot_state[:3]
            # cartesian_ori = robot_state[3:6]
            # gripper = robot_state[6]
            # cartesian_ori_sin = np.sin(cartesian_ori)
            # cartesian_ori_cos = np.cos(cartesian_ori)
            # robot_state = np.concatenate(
            #     [cartesian_pos, cartesian_ori_sin, cartesian_ori_cos, [gripper]], axis=0
            # )
            cartesian = robot_state[:6]
            quat_cartesian = get_quaternion_orientation(cartesian)
            robot_state = np.concatenate([quat_cartesian, robot_state[6:]], axis=0)

            # subscribe images
            image_dict = {}
            for cam_idx, img_sub in self.image_subscribers.items():
                image_dict[cam_idx] = img_sub.recv_rgb_image()[0]

            obs = {}
            obs["features"] = robot_state
            obs["proprioceptive"] = robot_state
            if self.sensor_type == "reskin":
                try:
                    sensor_state = np.array(
                        ret["sensor_state"]["reskin"]["sensor_values"]
                    )
                    # obs["sensor"] = np.array(sensor_state)
                    if self.subtract_sensor_baseline:
                        baseline_meas = []
                        while len(baseline_meas) < 5:
                            self.action_request_socket.send(b"get_sensor_state")
                            ret = pickle.loads(self.action_request_socket.recv())
                            sensor_state = ret["reskin"]["sensor_values"]
                            baseline_meas.append(sensor_state)
                        self.sensor_baseline = np.median(baseline_meas, axis=0)
                        sensor_state = sensor_state - self.sensor_baseline
                    self.sensor_prev_state = sensor_state
                    sensor_keys = [
                        f"sensor{sensor_idx}" for sensor_idx in range(self.n_sensors)
                    ]
                    for sidx, sensor_key in enumerate(sensor_keys):
                        obs[sensor_key] = sensor_state[
                            sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                        ]
                except KeyError:
                    pass
            elif self.sensor_type == "digit":
                for dkey in self.digit_keys:
                    obs[dkey] = np.array(ret["sensor_state"][dkey])
                    obs[dkey] = cv2.resize(obs[dkey], (self.width, self.height))
                    if self.subtract_sensor_baseline:
                        baseline_meas = []
                        while len(baseline_meas) < 5:
                            self.action_request_socket.send(b"get_sensor_state")
                            ret = pickle.loads(self.action_request_socket.recv())
                            sensor_state = cv2.resize(
                                ret[dkey], (self.width, self.height)
                            )
                            baseline_meas.append(sensor_state)
                        self.sensor_baseline = np.median(baseline_meas, axis=0)
                        obs[dkey] = sensor_state - self.sensor_baseline
                        # obs["sensor"] = sensor_state - self.sensor_baseline
            for cam_idx, image in image_dict.items():
                if cam_idx == 52:
                    # crop the right side of the image for the gripper cam
                    img_shape = image.shape
                    crop_percent = 0.2
                    image = image[:, : int(img_shape[1] * (1 - crop_percent))]
                obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))

            return obs
        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["proprioceptive"] = np.zeros(self.proprio_dim)
            for sensor_idx in range(self.n_sensors):
                obs[f"sensor{sensor_idx}"] = np.zeros(self.sensor_dim)
            self.sensor_baseline = np.zeros(self.sensor_dim * self.n_sensors)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            return obs

    def render(self, mode="rgb_array", width=640, height=480):
        print("rendering")
        # subscribe images
        image_list = []
        for _, img_sub in self.image_subscribers.items():
            image = img_sub.recv_rgb_image()[0]
            image_list.append(cv2.resize(image, (width, height)))

        obs = np.concatenate(image_list, axis=1)
        return obs


if __name__ == "__main__":
    env = RobotEnv()
    obs = env.reset()
    import ipdb

    ipdb.set_trace()

    for i in range(30):
        action = obs["features"]
        action[0] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[1] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[2] += 2
        obs, reward, done, _ = env.step(action)
