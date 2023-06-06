//========= Copyright 2016-2018, HTC Corporation. All rights reserved. ===========

using UnityEngine;
using HTC.UnityPlugin.StereoRendering;

public class CreateStereoRenderer : MonoBehaviour
{
    public GameObject target;

    void Update ()
    {
        StereoRenderer r = target.GetComponent<StereoRenderer>();

        if (r == null && Input.GetKeyDown(KeyCode.Space))
        {
            StereoRenderer stereoRenderer = target.AddComponent<StereoRenderer>();
            stereoRenderer.SetUnlit(true);

            stereoRenderer.anchorPos = new Vector3(-0.22f, 0.0f, -2.39f);
            stereoRenderer.anchorEuler = new Vector3(0, 180.0f, 0);

            stereoRenderer.canvasOriginPos = new Vector3(0, 0, 6.57f);
        }
    }
}
