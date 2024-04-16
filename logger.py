import csv
import datetime
from collections import defaultdict

import numpy as np
import torch
import torchvision
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

BC_TRAIN_FORMAT = [
    ("step", "S", "int"),
    ("actor_loss", "L", "float"),
    ("total_time", "T", "time"),
]
BC_EVAL_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("imitation_reward", "R_i", "float"),
    ("total_time", "T", "time"),
]
SSL_TRAIN_FORMAT = [
    ("step", "S", "int"),
    ("loss", "L", "float"),
    ("total_time", "T", "time"),
]
SSL_EVAL_FORMAT = [
    ("epoch", "E", "int"),
    ("step", "S", "int"),
    ("loss", "E", "float"),
    ("total_time", "T", "time"),
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train_vq"):
                key = key[len("train_vq") + 1 :]
            elif key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["step"]) >= data["step"]:
                    break
                rows.append(row)
        with self._csv_file_name.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(data.keys()), restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open("a")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(data.keys()), restval=0.0
            )
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.04f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise f"invalid format type: {ty}"

    def _dump_to_console(self, data, prefix):
        prefix = colored(
            prefix, "yellow" if prefix in ["train", "train_vq"] else "green"
        )
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["frame"] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb, mode="bc"):
        """
        mode: bc, ssl
        """
        self._log_dir = log_dir
        if mode == "bc":
            self._train_mg = MetersGroup(
                log_dir / "train.csv", formating=BC_TRAIN_FORMAT
            )
            self._eval_mg = MetersGroup(log_dir / "eval.csv", formating=BC_EVAL_FORMAT)
        elif mode == "ssl":
            self._train_mg = MetersGroup(
                log_dir / "train.csv", formating=SSL_TRAIN_FORMAT
            )
            self._train_vq_mg = MetersGroup(
                log_dir / "train_vq.csv", formating=SSL_TRAIN_FORMAT
            )
            self._eval_mg = MetersGroup(log_dir / "eval.csv", formating=SSL_EVAL_FORMAT)
        if use_tb:
            self._sw = SummaryWriter(str(log_dir / "tb"))
        else:
            self._sw = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        # mg = self._train_mg if key.startswith('train') else self._eval_mg
        if key.startswith("train_vq"):
            mg = self._train_vq_mg
        else:
            mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f"{ty}/{key}", value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == "eval":
            self._eval_mg.dump(step, "eval")
        if ty is None or ty == "train":
            self._train_mg.dump(step, "train")
        if ty is None or ty == "train_vq":
            self._train_vq_mg.dump(step, "train_vq")

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f"{self._ty}/{key}", value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)
