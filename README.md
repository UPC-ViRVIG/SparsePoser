# SparsePoser: Real-time Full-body Motion Reconstruction from Sparse Data

[Jose Luis Ponton<sup>*,1</sup>](https://joseluisponton.com/), [Haoran Yun<sup>*,1</sup>](https://haoranyun.com), [Andreas Aristidou<sup>2,3</sup>](http://andreasaristidou.com), [Carlos Andujar<sup>1</sup>](https://www.cs.upc.edu/~andujar), [Nuria Pelechano<sup>1</sup>](https://www.cs.upc.edu/~npelechano)<br/>

<sup>1</sup> [Universitat Politècnica de Catalunya (UPC)](https://www.upc.edu/en?set_language=en), Spain <br/>
<sup>2</sup> [University of Cyprus](https://www.ucy.ac.cy/?lang=en), Cyprus <br/>
<sup>3</sup> [CYENS Centre of Excellence](https://www.cyens.org.cy/en-gb/), Cyprus <br/>
<sup>*</sup> [Jose Luis Ponton](https://joseluisponton.com/) and [Haoran Yun](https://haoranyun.com) are joint first authors.

<p align="center">
</p>

---

<p align="center">
  <img 
    width="940"
    height="231"
    src="docs/assets/img/teaser.jpg"
  >
</p>

This repository contains the implementation of the method shown in the paper *SparsePoser: Real-time Full-body Motion Reconstruction from Sparse Data* published at the **ACM Transactions on Graphics**.

Get an overview of the paper by visiting the [project website](https://upc-virvig.github.io/SparsePoser/) or watching the [video](https://youtu.be/BAi4KoHtehY)!

Download the paper [here](https://doi.org/10.1145/3625264)!

---

## Contents

1. [Project Architecture](#project-architecture)
2. [Getting Started](#getting-started)
   * [Evaluating BVH Files](#evaluating-bvh-files)
   * [Unity Live Demo](#unity-live-demo)
   * [Training Procedure](#training-procedure)
3. [Data](#data)
4. [Citation](#citation)
5. [License](#license)

## Project Architecture

The project is structured into two primary directories: `SparsePoserUnity`, which contains the Unity project, and `python`, where the network implementation using PyTorch is located. The network runs in a Python environment, and it communicates with Unity using a TCP connection for result delivery.

## Getting Started

1. Clone the repository onto your local system.
2. Navigate to the `python` directory with the command: ``cd SparsePoser/python/``.
3. Create a virtual environment with: ``python -m venv env`` (tested on Python 3.9).
4. Activate the created virtual environment.
5. Install the necessary packages from the requirements file with: ``pip install -r requirements.txt``.
6. Download and install [PyTorch](https://pytorch.org/get-started/locally/).
7. Retrieve the [motion dataset](https://zenodo.org/record/8427980/files/data.zip?download=1) and decompress it.

At this stage, your `python/` directory should be organized as follows:

```
└───python
    ├───data
    │   └───xsens
    │       ├───eval
    │       └───train
    ├───env
    ├───models 
    │   ├───model_dancedb
    │   └───model_xsens
    └───src
```

### Evaluating BVH Files

1. Use the following command to evaluate BVH files: ``python src/eval.py models/model_xsens/ data/xsens/eval/S02_A04.bvh ik``.
   
   > Feel free to synthesize motion from any other .bvh file in the `.\data\xsens\` directory.
   
   > Replace ``ik`` with ``generator`` to exclusively synthesize motion using the generator network.

2. The output will be stored in ``data/eval_S02_A04.bvh``.

### Unity Live Demo

**Installation Process**

1. Download and install Unity 2021.2.13f1. (Note: Other versions may work but have not been tested.)

2. Launch the Unity Hub application. Select ``Open`` and navigate to ``SparsePoser/SparsePoserUnity/``.
   
   > Upon opening, Unity may issue a warning about compilation errors in the project. Please select ``Ignore`` to proceed. Unity should handle the dependencies automatically, with the exception of SteamVR.

3. Import SteamVR from the Asset Store using the following link: [SteamVR Plugin](https://assetstore.unity.com/packages/tools/integration/steamvr-plugin-32647).
   
   > Unity should configure the project for VR use and install OpenVR during the import of SteamVR. Any previously encountered errors should be resolved at this stage and the console should be clear.

4. [Only VR] Within Unity, navigate to ``Edit/Project Settings/XR Plug-in Management`` and select ``OpenVR Loader`` from the ``Plug-in Providers`` list. Navigate to ``Edit/Project Settings/XR Plug-in Management/OpenVR`` and change the ``Stereo Rendering Mode`` from ``Single-Pass Instanced`` to ``Multi Pass``.

**Running the Simulator**

1. Within the ``Project Window``, navigate to ``Assets/Scenes/SampleSceneSimulator``.

2. Pick a skeleton (a ``.txt`` file) from ``Assets/Data/`` and refer to it in the ``T Pose BVH`` field in the ``PythonCommunication`` component of the Unity scene.
   
   > We provide different skeleton configurations based on real human dimensions. For usability, the skeleton files are named as follows: ``<gender>_<height>_<hips_height>.txt``.

3. Open a Terminal or Windows PowerShell, go to ``SparsePoser/python/``, activate the virtual environment, and execute the following command: ``python src/unity.py /models/model_xsens/ ../SparsePoserUnity/Assets/Data/male_180_94.txt ik``. Replace ``male_180_94.txt`` with the skeleton selected in step 2.
   
   > This command facilitates communication between Unity and Python. Note: Repeatedly playing and stopping Unity may disrupt this and result in an error in both Unity and Python. If this occurs, re-execute the Terminal command.

4. Press ``Play`` in Unity. Manipulate the GameObjects (``Root``, ``LFoot``, ``RFoot``, ``Head``, ``LHand`` and ``RHand``) located within the ``Trackers`` GameObject to adjust the end-effectors.
   
   > The initial position and rotation of the trackers are utilized during calibration, so refrain from modifying them.

**Running the Virtual Reality Demo**

1. In the ``Project Window``, select the scene ``Assets/Scenes/SampleSceneVR``.

2. Choose a skeleton (a ``.txt`` file) from ``Assets/Data/`` and reference it in the ``T Pose BVH`` field in the ``PythonCommunication`` component of the Unity scene.
   
   > We provide different skeleton configurations based on real human dimensions. For usability, the skeleton files are named as follows: ``<gender>_<height>_<hips_height>.txt``.
   
   > Warning: the system may fail when the user dimensions are considerably different from those of the skeleton. Please, choose the closest skeleton to the participant.

3. Open a Terminal or Windows PowerShell, go to ``SparsePoser/python/``, activate the virtual environment, and execute the following command: ``python src/unity.py /models/model_xsens/ ../SparsePoserUnity/Assets/Data/male_180_94.txt ik``. Replace ``male_180_94.txt`` with the skeleton chosen in step 2.
   
   > This command facilitates communication between Unity and Python. Note: Repeatedly playing and stopping Unity may disrupt this and result in an error in both Unity and Python. If this occurs, re-execute the Terminal command.

4. Initiate SteamVR and connect one Head-Mounted Display (HMD), two HTC VIVE hand-held controllers, and three HTC VIVE Trackers 3.0. (Note: Other versions might work but have not been tested.)

5. Press ``Play`` in Unity.

6. The first time the application is played, a SteamVR pop-up will appear to generate input actions, select accept. Similarly, a TextMeshPro pop-up will appear, select ``Import TMP Essentials``.

7. Within the VR environment, locate the mirror. Stand in a T-Pose and press the ``Trigger`` button on any handheld controller.

8. A yellow skeleton will appear. Position your feet and body within this skeleton and face forward. Aligning hands is not necessary. Press the ``Trigger`` button on any handheld controller once you're in position.
   
   > If the yellow skeleton doesn't animate within a few seconds, consider restarting Unity or the Terminal executing Python.

### Training Procedure

1. [Optional] If needed, modify the hyperparameters in the `src/train.py` parameter dictionary.

2. Initiate the training process with the following command: ``python src/train.py data/xsens/ train_test all``.
   
   > The syntax for the command is as follows: ``python src/train.py <train_db> <name> <generator|ik|all>``

3. The training outcome will be stored in the following directory: ``models/model_<name>_<train_db>``.

## Data

Data used for training can be downloaded from [here](https://zenodo.org/record/8427980/files/data.zip?download=1).
It contains over **1GB** of high-quality **motion capture data** recorded with an **Xsens Awinda** system.

## Citation

If you find this code useful for your research, please cite our paper:

```
@article {ponton2023sparseposer,
	author = {Ponton, Jose Luis and Yun, Haoran and Aristidou, Andreas and Andujar, Carlos and Pelechano, Nuria},
	title = {SparsePoser: Real-Time Full-Body Motion Reconstruction from Sparse Data},
	year = {2023},
	issue_date = {February 2024},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	volume = {43},
	number = {1},
	issn = {0730-0301},
	url = {https://doi.org/10.1145/3625264},
	doi = {10.1145/3625264},
	journal = {ACM Trans. Graph.},
	month = {oct},
	articleno = {5},
	numpages = {14}
}
```

## License

- The code in this repository is released under the MIT license. Please, see the [LICENSE](LICENSE) for further details.

- The model weights in this repository and the data are licensed under CC BY-SA 4.0 license as found in the [LICENSE_data](LICENSE_data) file.
