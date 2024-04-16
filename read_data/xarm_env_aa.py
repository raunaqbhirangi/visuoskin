from collections import defaultdict
import cv2
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R


def get_relative_action(actions, action_after_steps):
    """
    Convert absolute axis angle actions to relative axis angle actions
    Action has both position and orientation. Convert to transformation matrix, get
    relative transformation matrix, convert back to axis angle
    """

    relative_actions = []
    for i in range(len(actions)):
        # Get relative transformation matrix
        # previous pose
        pos_prev = actions[i, :3]
        ori_prev = actions[i, 3:6]
        r_prev = R.from_rotvec(ori_prev).as_matrix()
        matrix_prev = np.eye(4)
        matrix_prev[:3, :3] = r_prev
        matrix_prev[:3, 3] = pos_prev
        # current pose
        next_idx = min(i + action_after_steps, len(actions) - 1)
        pos = actions[next_idx, :3]
        ori = actions[next_idx, 3:6]
        gripper = actions[next_idx, 6:]
        r = R.from_rotvec(ori).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = r
        matrix[:3, 3] = pos
        # relative transformation
        matrix_rel = np.linalg.inv(matrix_prev) @ matrix
        # relative pose
        # pos_rel = matrix_rel[:3, 3]
        pos_rel = pos - pos_prev
        r_rel = R.from_matrix(matrix_rel[:3, :3]).as_rotvec()
        # # compute relative rotation
        # r_prev = R.from_rotvec(ori_prev).as_matrix()
        # r = R.from_rotvec(ori).as_matrix()
        # r_rel = np.linalg.inv(r_prev) @ r
        # r_rel = R.from_matrix(r_rel).as_rotvec()
        # # compute relative translation
        # pos_rel = pos - pos_prev
        relative_actions.append(np.concatenate([pos_rel, r_rel, gripper]))
        # next_idx = min(i + action_after_steps, len(actions) - 1)
        # curr_pose, _ = actions[i, :6], actions[i, 6:]
        # next_pose, next_gripper = actions[next_idx, :6], actions[next_idx, 6:]

    # last action
    last_action = np.zeros_like(actions[-1])
    last_action[-1] = actions[-1][-1]
    while len(relative_actions) < len(actions):
        relative_actions.append(last_action)
    return np.array(relative_actions, dtype=np.float32)


def get_absolute_action(rel_actions, base_action):
    """
    Convert relative axis angle actions to absolute axis angle actions
    """
    actions = np.zeros((len(rel_actions) + 1, rel_actions.shape[-1]))
    actions[0] = base_action
    for i in range(1, len(rel_actions) + 1):
        # if i == 0:
        #     actions.append(base_action)
        #     continue
        # Get relative transformation matrix
        # previous pose
        pos_prev = actions[i - 1, :3]
        ori_prev = actions[i - 1, 3:6]
        r_prev = R.from_rotvec(ori_prev).as_matrix()
        matrix_prev = np.eye(4)
        matrix_prev[:3, :3] = r_prev
        matrix_prev[:3, 3] = pos_prev
        # relative pose
        pos_rel = rel_actions[i - 1, :3]
        r_rel = rel_actions[i - 1, 3:6]
        # compute relative transformation matrix
        matrix_rel = np.eye(4)
        matrix_rel[:3, :3] = R.from_rotvec(r_rel).as_matrix()
        matrix_rel[:3, 3] = pos_rel
        # compute absolute transformation matrix
        matrix = matrix_prev @ matrix_rel
        # absolute pose
        pos = matrix[:3, 3]
        # r = R.from_matrix(matrix[:3, :3]).as_rotvec()
        r = R.from_matrix(matrix[:3, :3]).as_euler("xyz")
        actions[i] = np.concatenate([pos, r, rel_actions[i - 1, 6:]])
    return np.array(actions, dtype=np.float32)


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    new_cartesian = []
    for i in range(len(cartesian)):
        pos = cartesian[i, :3]
        ori = cartesian[i, 3:]
        quat = R.from_rotvec(ori).as_quat()
        new_cartesian.append(np.concatenate([pos, quat], axis=-1))
    return np.array(new_cartesian, dtype=np.float32)


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        tasks,
        num_demos_per_task,
        temporal_agg,
        num_queries,
        img_size,
        action_after_steps,
        store_actions,
        pixel_keys,
        aux_keys,
        subsample,
        skip_first_n,
        relative_actions,
        random_mask_proprio,
        sensor_params,
    ):
        self._img_size = img_size
        self._action_after_steps = action_after_steps
        self._store_actions = store_actions
        self._pixel_keys = pixel_keys
        self._aux_keys = aux_keys
        self._random_mask_proprio = random_mask_proprio

        self._sensor_type = sensor_params.sensor_type
        self._subtract_sensor_baseline = sensor_params.subtract_sensor_baseline
        if self._sensor_type == "digit":
            use_anyskin_data = False
        elif self._sensor_type == "reskin":
            use_anyskin_data = True
        else:
            assert self._sensor_type == None

        self._num_anyskin_sensors = 2

        # temporal aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = []
        self._paths.extend([Path(path) / f"{task}.pkl" for task in tasks])

        paths = {}
        idx = 0
        for path in self._paths:
            paths[idx] = path
            idx += 1
        del self._paths
        self._paths = paths

        # store actions
        if self._store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 7
        self._num_samples = 0
        min_stat, max_stat = None, None
        min_sensor_stat, max_sensor_stat = None, None
        digit_mean_stat, digit_std_stat = defaultdict(list), defaultdict(list)
        min_sensor_diff_stat, max_sensor_diff_stat = None, None
        min_act, max_act = None, None
        self.prob = []
        sensor_states = []
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # Add to prob
            if "fridge" in str(self._paths[_path_idx]):
                # self.prob.append(25.0/11.0)
                self.prob.append(22.0 / 9.0)
            else:
                self.prob.append(1)
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = data["observations"]
            # store
            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                # compute actions
                # absolute actions
                actions = np.concatenate(
                    [
                        observations[i]["cartesian_states"],
                        observations[i]["gripper_states"][:, None],
                    ],
                    axis=1,
                )
                if len(actions) == 0:
                    continue
                # skip first n
                if skip_first_n is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][skip_first_n:]
                    actions = actions[skip_first_n:]
                # subsample
                if subsample is not None:
                    for key in observations[i].keys():
                        observations[i][key] = observations[i][key][::subsample]
                    actions = actions[::subsample]
                # action after steps
                if relative_actions:
                    actions = get_relative_action(actions, self._action_after_steps)
                else:
                    actions = actions[self._action_after_steps :]
                # Convert cartesian states to quaternion orientation
                observations[i]["cartesian_states"] = get_quaternion_orientation(
                    observations[i]["cartesian_states"]
                )
                if use_anyskin_data:
                    try:
                        sensor_baseline = np.median(
                            observations[i]["sensor_states"][:5], axis=0, keepdims=True
                        )
                        if self._subtract_sensor_baseline:
                            observations[i]["sensor_states"] = (
                                observations[i]["sensor_states"] - sensor_baseline
                            )
                            if max_sensor_stat is None:
                                max_sensor_stat = np.max(
                                    observations[i]["sensor_states"], axis=0
                                )
                                min_sensor_stat = np.min(
                                    observations[i]["sensor_states"], axis=0
                                )
                            else:
                                max_sensor_stat = np.maximum(
                                    max_sensor_stat,
                                    np.max(observations[i]["sensor_states"], axis=0),
                                )
                                min_sensor_stat = np.minimum(
                                    min_sensor_stat,
                                    np.min(observations[i]["sensor_states"], axis=0),
                                )
                        for sensor_idx in range(self._num_anyskin_sensors):
                            observations[i][
                                f"sensor{sensor_idx}_states"
                            ] = observations[i]["sensor_states"][
                                ..., sensor_idx * 15 : (sensor_idx + 1) * 15
                            ]

                    except KeyError:
                        print("WARN: Sensor data not found.")
                        use_anyskin_data = False
                elif self._sensor_type == "digit":
                    for key in self._aux_keys:
                        if key.startswith("digit"):
                            observations[i][key] = (
                                observations[i][key].astype(np.float32) / 255.0
                            )
                            if self._subtract_sensor_baseline:
                                sensor_baseline = np.median(
                                    observations[i][key][:5], axis=0, keepdims=True
                                )  # .astype(observations[i][key].dtype)
                                observations[i][key] = (
                                    observations[i][key] - sensor_baseline
                                )
                                delta_filter = np.abs(observations[i][key]) > (
                                    5.0 / 255.0
                                )
                                digit_std_stat[key].append(
                                    observations[i][key][delta_filter]
                                )
                            else:
                                pass

                for key in observations[i].keys():
                    observations[i][key] = np.concatenate(
                        [
                            [observations[i][key][0]],
                            observations[i][key],
                        ],
                        axis=0,
                    )

                remaining_actions = actions[0]
                if relative_actions:
                    pos = remaining_actions[:-1]
                    ori_gripper = remaining_actions[-1:]
                    remaining_actions = np.concatenate(
                        [np.zeros_like(pos), ori_gripper]
                    )
                actions = np.concatenate(
                    [
                        [remaining_actions],
                        actions,
                    ],
                    axis=0,
                )
                # store
                episode = dict(
                    observation=observations[i],
                    action=actions,
                    # task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i][self._pixel_keys[0]])
                    ),
                )
                self._num_samples += len(observations[i][self._pixel_keys[0]])

                # max, min action
                if min_act is None:
                    min_act = np.min(actions, axis=0)
                    max_act = np.max(actions, axis=0)
                else:
                    min_act = np.minimum(min_act, np.min(actions, axis=0))
                    max_act = np.maximum(max_act, np.max(actions, axis=0))

                # store actions
                if self._store_actions:
                    self.actions.append(actions)

            # keep record of max and min stat
            max_cartesian = data["max_cartesian"]
            min_cartesian = data["min_cartesian"]
            max_cartesian = np.concatenate(
                [data["max_cartesian"][:3], [1] * 4]
            )  # for quaternion
            min_cartesian = np.concatenate(
                [data["min_cartesian"][:3], [-1] * 4]
            )  # for quaternion
            max_gripper = data["max_gripper"]
            min_gripper = data["min_gripper"]
            max_val = np.concatenate([max_cartesian, max_gripper[None]], axis=0)
            min_val = np.concatenate([min_cartesian, min_gripper[None]], axis=0)
            if max_stat is None:
                max_stat = max_val
                min_stat = min_val
            else:
                max_stat = np.maximum(max_stat, max_val)
                min_stat = np.minimum(min_stat, min_val)
            if use_anyskin_data:
                # If baseline is subtracted, use zero as shift and max as scale
                if self._subtract_sensor_baseline:
                    max_sensor_stat = np.maximum(
                        np.abs(max_sensor_stat), np.abs(min_sensor_stat)
                    )
                    min_sensor_stat = np.zeros_like(max_sensor_stat)
                # If baseline isn't subtracted, use usual min and max values
                else:
                    if max_sensor_stat is None:
                        max_sensor_stat = data["max_sensor"]
                        min_sensor_stat = data["min_sensor"]
                    else:
                        max_sensor_stat = np.maximum(
                            max_sensor_stat, data["max_sensor"]
                        )
                        min_sensor_stat = np.minimum(
                            min_sensor_stat, data["min_sensor"]
                        )
        min_act[3:6], max_act[3:6] = 0, 1  #################################
        self.stats = {
            "actions": {
                "min": min_act,  # min_stat,
                "max": max_act,  # max_stat,
            },
            "proprioceptive": {
                "min": min_stat,
                "max": max_stat,
            },
        }
        if use_anyskin_data:
            for sensor_idx in range(self._num_anyskin_sensors):
                sensor_mask = np.zeros_like(min_sensor_stat, dtype=bool)
                sensor_mask[sensor_idx * 15 : (sensor_idx + 1) * 15] = True
                self.stats[f"sensor{sensor_idx}"] = {
                    "min": min_sensor_stat[sensor_mask],
                    "max": max_sensor_stat[sensor_mask],
                }

            if not self._subtract_sensor_baseline:
                raise NotImplementedError(
                    "Normalization not implemented without baseline subtraction"
                )
            for key in self.stats:
                if key.startswith("sensor"):
                    sensor_states = np.concatenate(
                        [
                            observations[i][f"{key}_states"]
                            for i in range(len(observations))
                        ],
                        axis=0,
                    )
                    sensor_std = (
                        np.std(sensor_states, axis=0).reshape((5, 3)).max(axis=0)
                    )
                    sensor_std[:2] = sensor_std[:2].max()
                    sensor_std = np.clip(sensor_std * 3, a_min=100, a_max=None)
                    # max_xyz = np.clip(max_xyz, a_min=400, a_max=None)
                    self.stats[key]["max"] = np.tile(
                        sensor_std, int(self.stats[key]["max"].shape[0] / 3)
                    )
        elif self._sensor_type == "digit":
            if self._subtract_sensor_baseline:
                shared_mean, shared_std = None, None
                for key in digit_std_stat:
                    digit_std_stat[key] = [
                        3 * np.concatenate(digit_std_stat[key], axis=0).std()
                    ] * 3
                    digit_mean_stat[key] = [0.0] * 3
                self.stats[key] = {
                    "mean": np.array(digit_mean_stat[key])[:, None, None],
                    "std": np.array(digit_std_stat[key])[:, None, None],
                }
                if shared_std is None:
                    shared_std = digit_std_stat[key]
                    shared_mean = digit_mean_stat[key]
                else:
                    shared_std = np.maximum(shared_std, digit_std_stat[key])
                    shared_mean = np.minimum(shared_mean, digit_mean_stat[key])
            else:
                shared_mean = [0.485, 0.456, 0.406]
                shared_std = [0.229, 0.224, 0.225]
            self.stats["digit"] = {
                "mean": np.array(shared_mean)[:, None, None],
                "std": np.array(shared_std)[:, None, None],
            }
            self.digit_aug = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        # augmentation
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(self._img_size, padding=4),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
            ]
        )

        # Samples from envs
        self.envs_till_idx = len(self._episodes)

        self.prob = np.array(self.prob) / np.sum(self.prob)

    def preprocess(self, key, x):
        if key.startswith("digit"):
            return (x - self.stats["digit"]["mean"]) / (
                self.stats["digit"]["std"] + 1e-5
            )
        return (x - self.stats[key]["min"]) / (
            self.stats[key]["max"] - self.stats[key]["min"] + 1e-5
        )

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx

        # sample idx with probability
        idx = np.random.choice(list(self._episodes.keys()), p=self.prob)

        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episodes, env_idx = self._sample_episode()
        observations = episodes["observation"]
        actions = episodes["action"]
        sample_idx = np.random.randint(1, len(observations[self._pixel_keys[0]]) - 1)
        # Sample obs, action
        sampled_pixel = {}
        for key in self._pixel_keys:
            sampled_pixel[key] = observations[key][-(sample_idx + 1) : -sample_idx]
            sampled_pixel[key] = torch.stack(
                [
                    self.aug(sampled_pixel[key][i])
                    for i in range(len(sampled_pixel[key]))
                ]
            )
            sampled_state = {}
            sampled_state = {}

        sampled_state = {}
        sampled_state["proprioceptive"] = np.concatenate(
            [
                observations["cartesian_states"][-(sample_idx + 1) : -sample_idx],
                observations["gripper_states"][-(sample_idx + 1) : -sample_idx][
                    :, None
                ],
            ],
            axis=1,
        )

        if self._random_mask_proprio and np.random.rand() < 0.5:
            sampled_state["proprioceptive"] = (
                np.ones_like(sampled_state["proprioceptive"])
                * self.stats["proprioceptive"]["min"]
            )
        if self._sensor_type == "reskin":
            try:
                for sensor_idx in range(self._num_anyskin_sensors):
                    skey = f"sensor{sensor_idx}"
                    sampled_state[f"{skey}"] = observations[f"{skey}_states"][
                        -(sample_idx + 1) : -sample_idx
                    ]
            except KeyError:
                pass
        elif self._sensor_type == "digit":
            try:
                for sensor_idx in range(self._num_anyskin_sensors):
                    key = f"digit{80 + sensor_idx}"
                    sampled_state[key] = observations[key][
                        -(sample_idx + 1) : -sample_idx
                    ]
                    sampled_state[key] = torch.stack(
                        [
                            self.digit_aug(sampled_state[key][i])
                            # self.aug(sampled_state[key][i])
                            for i in range(len(sampled_state[key]))
                        ]
                    )
            except KeyError as e:
                pass

        if self._temporal_agg:
            # arrange sampled action to be of shape (1, num_queries, action_dim)
            sampled_action = np.zeros((1, self._num_queries, actions.shape[-1]))
            num_actions = 1 + self._num_queries - 1
            act = np.zeros((num_actions, actions.shape[-1]))
            if num_actions - sample_idx < 0:
                act[:num_actions] = actions[-(sample_idx) : -sample_idx + num_actions]
            else:
                act[:sample_idx] = actions[-sample_idx:]
                act[sample_idx:] = actions[-1]
            sampled_action = np.lib.stride_tricks.sliding_window_view(
                act, (self._num_queries, actions.shape[-1])
            )
            sampled_action = sampled_action[:, 0]
        else:
            sampled_action = actions[-(sample_idx + 1) : -sample_idx]

        return_dict = {}
        for key in self._pixel_keys:
            return_dict[key] = sampled_pixel[key]
        for key in self._aux_keys:
            return_dict[key] = self.preprocess(key, sampled_state[key])
            return_dict["actions"] = self.preprocess("actions", sampled_action)
            return_dict["actions"] = self.preprocess("actions", sampled_action)
        return_dict["actions"] = self.preprocess("actions", sampled_action)
        return return_dict

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples
