from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "alohamini_control"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/worker", ["alohamini_control/hardware_worker.py"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="Guarded mock and single-arm hardware trajectory bridges for AlohaMini2Pro.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mock_trajectory_bridge = alohamini_control.mock_trajectory_bridge:main",
            "hardware_trajectory_bridge = alohamini_control.hardware_trajectory_bridge:main",
        ]
    },
)
