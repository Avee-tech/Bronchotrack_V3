from setuptools import find_packages, setup

setup(
    name="bronchotrack",
    version="0.1.0",
    description=(
        "Modular re-implementation of the BronchoTrack pipeline "
        "(airway lumen tracking for branch-level bronchoscopic localization), "
        "designed to plug in an existing YOLOv11 detector and 3D-Slicer-derived "
        "airway graph."
    ),
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "opencv-python>=4.7",
    ],
    extras_require={
        "detection": ["ultralytics>=8.3"],
        "reid": ["torch>=2.0", "torchvision>=0.15"],
    },
    entry_points={
        "console_scripts": [
            "bronchotrack=bronchotrack.cli:main",
        ]
    },
)
