# Vive Stereo Rendering Toolkit - v1.1.2
Copyright 2016-2018, HTC Corporation. All rights reserved.

## Introduction

- Vive Stereo Rendering Toolkit provides drag-and-drop components for developers to create stereoscopic rendering effects in a few minutes. 
- With this toolkit, effects such as mirrors or portal doors can be easily achieved in your VR application.

## System Requirements

1. Unity 5.5.0 or higher.

2. Appropriate SDK for target VR devices
 - Vive or other SteamVR-compatible HMDs: SteamVR Unity plugin, version 1.2.0 or higher
 - Vive Focus: WaveVR plugin, version 2.0.13 or higher
 - Oculus Rift: OVR plugin, version 1.20.0 or higher (**only supported for Unity 2017.2 or higher**)

3. **To run the examples, add camera rig prefabs from respective platform SDKs**
 - Vive or other SteamVR-compatible HMDs: no need to modify.
 - Vive Focus: add **WaveVR** from "WaveVR/Prefabs" to the scenes; remember to expand the camera rig.
 - Oculus Rift: add **OVRCameraRig** from "OVR/Prefabs" to the scenes.

For tutorial and API reference, please see the attached Developer's Guide.

## Change log

v1.1.2
 - Add option to make StereoRenderers unlit by scene lights (default off)

v1.1.1
 - Add option to enable oblique near plane clipping for better performance (default on)
 - Add option to enable scissor test for better performance (default on)
 - Fix: "IsMirror" option in Unity editor would fail to set anchor position under certain conditions

v1.1.0
 - [Backward Compatibility Break Change] drop support for Unity 5.3/5.4
 - Add multi-platform support: we currently support Vive, Vive Focus and Oculus Rift
 - Add an example for two-way portal; remove obsolete examples
 
v1.0.9
 - fix wrong rendering under single-pass stereo rendering for Unity 5.6.0

v1.0.8
 - new icon :)

v1.0.7
 - fix compatiblity for SteamVR plugin v1.2.1
 - bug fix: cannot build project due to SerializedProperty not exist outside of Unity Editor 

v1.0.6
 - Each StereoRenderer can have different "ignore object list"
   (for existing project, need to add proper layers to Unity "Tags and Layers Manager" manually)

v1.0.5
 - [Backward Compatibility Break Change] Minimal supported SteamVR plugin version changed to v1.2.0
 - fix compatibility issue for SteamVR plugin v1.2.0

v1.0.4
 - support double-sided StereoRenderer
 - bug fix: fix right-eye rendering of mirrors (for Unity 5.4+)

v1.0.3
 - Unity 5.5 compatibility
 - enable creation of StereoRenderer at runtime
 - add parameter for tweaking resolution of rendertextures
 - bug fix: fix "drifting" when entering portals (for Unity 5.4)
 - bug fix: fix rendering of near-anchor objects (for Unity 5.4+)

v1.0.2
 - bug fix: fix wrong shadow in canvases when using directional light (for Unity 5.4)

v1.0.1
 - add functions to query the up, forward and right vectors of canvas origin
 - support moving or rotating mirrors at run time
 - support mirrors of arbitrary orientation
 - bug fix: only remove StereoRenderers from manager when their host object is destroyed