﻿//========= Copyright 2016-2017, HTC Corporation. All rights reserved. ===========

using System;
using System.Collections.Generic;
using UnityEngine;

namespace HTC.UnityPlugin.StereoRendering
{
    [DisallowMultipleComponent]
    public class StereoRenderManager : MonoBehaviour
    {
        // singleton
        private static StereoRenderManager instance = null;

        // flags
        private static bool isApplicationQuitting = false;

        // device types
        public HmdType hmdType;

        // factory for device-specific things
        public IDeviceParamFactory paramFactory;

        // all current stereo renderers
        public List<StereoRenderer> stereoRendererList = new List<StereoRenderer>();

        /////////////////////////////////////////////////////////////////////////////////
        // initialization

        // whehter we have initialized the singleton
        public static bool Active { get { return instance != null; } }

        // singleton interface
        public static StereoRenderManager Instance
        {
            get
            {
                Initialize();
                return instance;
            }
        }

        private static void Initialize()
        {
            if (Active || isApplicationQuitting) { return; }

            // try to get existing manager
            var instances = FindObjectsOfType<StereoRenderManager>();
            if (instances.Length > 0)
            {
                instance = instances[0];
                if (instances.Length > 1) { Debug.LogError("Multiple StereoRenderManager is not supported."); }
            }

            // pop warning if no VR device detected
#if (!(UNITY_ANDROID && VIVE_STEREO_WAVEVR))
    #if UNITY_2017_2_OR_NEWER
            if (!UnityEngine.XR.XRSettings.enabled) { Debug.LogError("VR is not enabled for this application."); }
    #else
            if (!UnityEngine.VR.VRSettings.enabled) { Debug.LogError("VR is not enabled for this application."); }
    #endif
#endif

            // get HMD head
            Camera head = GetHmdRig();
            if (head == null) { return; }
            if (head.transform.parent == null)
            {
                Debug.LogError("HMD rig is not of proper hierarchy. You need a \"rig\" object as its root.");
                return;
            }

            // if no exsiting instance, attach a new one to HMD camera
            if (!Active)
            {
                instance = head.gameObject.AddComponent<StereoRenderManager>();
            }
        }

        private static Camera GetHmdRig()
        {
            Camera target = null;

#if (UNITY_ANDROID && VIVE_STEREO_WAVEVR)
            if (WaveVR_Render.Instance != null)
            {
                var left = WaveVR_Render.Instance.lefteye.gameObject.AddComponent<VRRenderEventDetector>();
                left.Initialize(0);

                var right = WaveVR_Render.Instance.righteye.gameObject.AddComponent<VRRenderEventDetector>();
                right.Initialize(1);

                target = WaveVR_Render.Instance.GetComponent<Camera>();
            }
            else
            {
                Debug.LogError("No WaveVR_Render found.");
            }
#else
            if (Camera.main != null)
            {
                var head = Camera.main.gameObject.AddComponent<VRRenderEventDetector>();
                head.Initialize(0);

                target = Camera.main;
            }
            else
            {
                Debug.LogWarning("No Camera tagged as \"MainCamera\" found.");
            }
#endif
            return target;
        }

        public void InitParamFactory()
        {
            // if not yet initialized
            if (paramFactory == null)
            {
                // get device type
                hmdType = StereoRenderDevice.GetHmdType();

                // create parameter factory
                paramFactory = StereoRenderDevice.InitParamFactory(hmdType);
                if (paramFactory == null)
                {
                    Debug.LogError("Current VR device is unsupported.");
                }
            }
        }

        private void OnApplicationQuit()
        {
            isApplicationQuitting = true;
        }

        /////////////////////////////////////////////////////////////////////////////////
        // render related

        public void InvokeStereoRenderers(VRRenderEventDetector detector)
        {
            // render registored stereo cameras
            for (int renderIter = 0; renderIter < stereoRendererList.Count; renderIter++)
            {
                StereoRenderer stereoRenderer = stereoRendererList[renderIter];

                if (stereoRenderer.shouldRender)
                {
                    stereoRenderer.Render(detector);
                }
            }
        }

        /////////////////////////////////////////////////////////////////////////////////
        // callbacks

        public void AddToManager(StereoRenderer stereoRenderer)
        {
            stereoRendererList.Add(stereoRenderer);
        }

        public void RemoveFromManager(StereoRenderer stereoRenderer)
        {
            stereoRendererList.Remove(stereoRenderer);
        }
    }
}