from setuptools import find_packages, setup

PACKAGE_NAME = "alohamini_kinematics"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="Dry-run ROS 2 inverse kinematics for AlohaMini2Pro.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ik_dry_run = alohamini_kinematics.ik_dry_run_node:main",
        ]
    },
)
