# lerobot_alohamini

Shared software layer for the AlohaMini product line, built on HuggingFace LeRobot. Supports both the full AlohaMini robot (dual-arm + mobile base + lift) and the AM-ARM200 arm.

> Haven't assembled your hardware yet? Start here: [AlohaMini](https://github.com/liyiteng/alohamini) · [AM-ARM200](https://github.com/liyiteng/AM-ARM)

## Updates
- **[2025-05-21]** Add support for AM-ARM200 and AlohaMini 2 / 2 Pro
- **[2025-04-10]** Compatible with LeRobot 0.5.2

## Documentation

| Guide | Description |
|-------|-------------|
| [Install](docs/alohamini/install.md) | Environment setup, serial port permissions, HuggingFace configuration |
| [Hardware Profiles](docs/alohamini/profiles.md) | `--arm_profile` and `--robot_model` flag reference |
| [AM-ARM200](docs/alohamini/am-arm200.md) | Calibration → teleoperation → dataset recording → training → evaluation (single arm, one PC) |
| [AlohaMini 1 / 2 / 2 Pro](docs/alohamini/alohamini.md) | Calibration → teleoperation → dataset recording → training → evaluation (dual arm, Pi + PC) |

---

## Team & Contact

AlohaMini is created by **Li Yiteng** and **Wu Zhiyong**.

- Email: liyiteng+github@gmail.com
- WeChat: liyiteng

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) — the software stack this repository targets
- [ALOHA](https://tonyzhaozh.github.io/aloha/) — the bimanual teleoperation paradigm
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — pioneered the low-cost open arm design pattern
