from __future__ import annotations

from copy import deepcopy

TARGETS = ("cellulose", "hemicellulose", "lignin")


def se_attention_switches(*enabled: int) -> tuple[bool, bool, bool]:
    """Return Conv1/Conv2/Conv3 SE switches from Table S1 SE mode numbers."""

    enabled_set = {int(x) for x in enabled}
    return tuple(i in enabled_set for i in (1, 2, 3))


def se_mode_labels(se_after_conv: tuple[bool, bool, bool]) -> tuple[str, ...]:
    """Return paper-style SE labels: SE1, SE2, SE3."""

    return tuple(f"SE{i}" for i, enabled in enumerate(se_after_conv, start=1) if enabled)


PAPER_BASELINE_PARAMS = {
    "cellulose": {
        "plsr": {"preprocessing": [{"method": "msc"}], "n_components": 14},
        "svr": {"preprocessing": [{"method": "snv"}], "kernel": "linear", "C": 15.8407, "epsilon": 0.7806, "gamma": "scale"},
        "xgboost": {
            "preprocessing": [{"method": "snv"}, {"method": "sg", "window": 21, "deriv": 1}],
            "n_estimators": 565,
            "max_depth": 2,
            "learning_rate": 0.0193,
            "subsample": 0.83,
            "colsample_bytree": 0.8641,
            "min_child_weight": 10,
            "reg_alpha": 7.86e-05,
            "reg_lambda": 0.0019,
            "gamma": 0.8914,
        },
        "cnn": {
            "preprocessing": [{"method": "sg", "window": 33, "deriv": 1}],
            "filters": (32, 32, 16),
            "kernels": (17, 17, 5),
            "strides": (1, 2),
            "se_mode": ("SE3",),
            "se_after_conv": se_attention_switches(3),
            "dense": 64,
            "dropout": 0.35,
            "batch_size": 128,
            "lr": 6.08e-05,
            "weight_decay": 3.08e-06,
        },
    },
    "hemicellulose": {
        "plsr": {"preprocessing": [{"method": "msc"}], "n_components": 15},
        "svr": {"preprocessing": [{"method": "msc"}], "kernel": "linear", "C": 16.7775, "epsilon": 0.6284, "gamma": "scale"},
        "xgboost": {
            "preprocessing": [{"method": "sg", "window": 27, "deriv": 2}],
            "n_estimators": 1014,
            "max_depth": 5,
            "learning_rate": 0.022,
            "subsample": 0.8639,
            "colsample_bytree": 0.9405,
            "min_child_weight": 10,
            "reg_alpha": 0.04,
            "reg_lambda": 0.0012,
            "gamma": 4.5107,
        },
        "cnn": {
            "preprocessing": [{"method": "sg", "window": 5, "deriv": 1}],
            "filters": (64, 128, 32),
            "kernels": (7, 19, 13),
            "strides": (2, 2),
            "se_mode": ("SE3",),
            "se_after_conv": se_attention_switches(3),
            "dense": 128,
            "dropout": 0.3,
            "batch_size": 128,
            "lr": 1.18e-04,
            "weight_decay": 1.54e-05,
        },
    },
    "lignin": {
        "plsr": {"preprocessing": [{"method": "msc"}], "n_components": 20},
        "svr": {
            "preprocessing": [{"method": "snv"}, {"method": "sg", "window": 21, "deriv": 1}],
            "kernel": "linear",
            "C": 0.4861,
            "epsilon": 0.1346,
            "gamma": "scale",
        },
        "xgboost": {
            "preprocessing": [{"method": "snv"}],
            "n_estimators": 1048,
            "max_depth": 2,
            "learning_rate": 0.0142,
            "subsample": 0.7876,
            "colsample_bytree": 0.9043,
            "min_child_weight": 2,
            "reg_alpha": 6.91e-08,
            "reg_lambda": 5.2037,
            "gamma": 2.776,
        },
        "cnn": {
            "preprocessing": [{"method": "snv"}, {"method": "sg", "window": 19, "deriv": 1}],
            "filters": (64, 32, 32),
            "kernels": (7, 25, 17),
            "strides": (2, 2),
            "se_mode": ("SE3",),
            "se_after_conv": se_attention_switches(3),
            "dense": 256,
            "dropout": 0.4,
            "batch_size": 128,
            "lr": 2.23e-04,
            "weight_decay": 4.89e-06,
        },
    },
}


PAPER_SREM_PARAMS = {
    "cellulose": {
        "preprocessing": [{"method": "sg", "window": 9, "deriv": 1}],
        "se_mode": ("SE2", "SE3"),
        "se_after_conv": se_attention_switches(2, 3),
        "se_reduction": 2,
        "dropout": 0.05,
        "dense": 256,
        "block_filters": ((16, 8, 32), (8, 8, 32), (16, 32, 4)),
        "block_kernels": ((23, 3, 11), (13, 23, 13), (5, 23, 21)),
        "strides": (1, 2),
        "batch_size": 256,
        "lr": 2.85e-05,
        "weight_decay": 0.0073,
    },
    "hemicellulose": {
        "preprocessing": [{"method": "sg", "window": 5, "deriv": 1}],
        "se_mode": ("SE2", "SE3"),
        "se_after_conv": se_attention_switches(2, 3),
        "se_reduction": 16,
        "dropout": 0.1,
        "dense": 128,
        "block_filters": ((128, 128, 8), (16, 64, 128), (8, 8, 128)),
        "block_kernels": ((5, 9, 5), (27, 21, 27), (19, 17, 21)),
        "strides": (1, 2),
        "batch_size": 512,
        "lr": 1.67e-06,
        "weight_decay": 0.0,
    },
    "lignin": {
        "preprocessing": [{"method": "snv"}, {"method": "sg", "window": 7, "deriv": 1}],
        "se_mode": ("SE3",),
        "se_after_conv": se_attention_switches(3),
        "se_reduction": 8,
        "dropout": 0.45,
        "dense": 64,
        "block_filters": ((128, 4, 32), (128, 8, 64), (32, 4, 8)),
        "block_kernels": ((25, 11, 19), (3, 23, 7), (13, 23, 17)),
        "strides": (2, 1),
        "batch_size": 128,
        "lr": 5.42e-06,
        "weight_decay": 1.82e-05,
    },
}


def paper_baseline_params(target: str, model: str) -> dict:
    return deepcopy(PAPER_BASELINE_PARAMS[target][model])


def paper_srem_params(target: str) -> dict:
    return deepcopy(PAPER_SREM_PARAMS[target])
