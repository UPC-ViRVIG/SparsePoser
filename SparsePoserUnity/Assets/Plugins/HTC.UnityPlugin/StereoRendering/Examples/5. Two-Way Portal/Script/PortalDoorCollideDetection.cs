//========= Copyright 2016-2017, HTC Corporation. All rights reserved. ===========
using HTC.UnityPlugin.StereoRendering;
using UnityEngine;

[RequireComponent(typeof(StereoRenderer))]
public class PortalDoorCollideDetection : MonoBehaviour
{
    public Collider playerCollider;
    public PortalSet portalManger;
    private StereoRenderer stereoRenderer;

    /////////////////////////////////////////////////////////////////////////////////

    private void OnEnable()
    {
        stereoRenderer = GetComponent<StereoRenderer>();
    }

    private void OnTriggerEnter(Collider other)
    {
        // if hmd has collided with portal door, notify parent
        if (other == playerCollider)
        {
            Debug.Log("**" + stereoRenderer.gameObject.transform.parent.name);
            portalManger.OnPlayerEnter(stereoRenderer);
        }
    }
}
