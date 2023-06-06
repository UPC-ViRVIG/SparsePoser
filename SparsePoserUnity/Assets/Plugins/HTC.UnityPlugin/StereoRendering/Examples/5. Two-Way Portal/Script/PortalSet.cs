//========= Copyright 2016-2017, HTC Corporation. All rights reserved. ===========
using HTC.UnityPlugin.StereoRendering;
using System.Collections;
using UnityEngine;

public class PortalSet : MonoBehaviour
{
    public GameObject hmdRig;
    public GameObject hmdEye;

    private bool isColliding = false;
    
    public void OnPlayerEnter(StereoRenderer stereoRenderer)
    {
        if(!isColliding)
        {
            StartCoroutine(MovePlayer(stereoRenderer));
        }
    }

    IEnumerator MovePlayer(StereoRenderer stereoRenderer)
    {
        isColliding = true;

        Debug.Log(stereoRenderer.gameObject.transform.parent.name);

        stereoRenderer.shouldRender = false;

        // adjust rotation
        Quaternion rotEntryToExit = stereoRenderer.anchorRot * Quaternion.Inverse(stereoRenderer.canvasOriginRot);
        hmdRig.transform.rotation = rotEntryToExit * hmdRig.transform.rotation;

        // adjust position
        Vector3 posDiff = new Vector3(stereoRenderer.stereoCameraHead.transform.position.x - hmdEye.transform.position.x,
                                      stereoRenderer.stereoCameraHead.transform.position.y - hmdEye.transform.position.y,
                                      stereoRenderer.stereoCameraHead.transform.position.z - hmdEye.transform.position.z);
        Vector3 offset = stereoRenderer.anchorRot * Vector3.forward;
        offset.Normalize();

        Vector3 camRigTargetPos = hmdRig.transform.position + posDiff;

        // assign the target position to camera rig
        hmdRig.transform.position = camRigTargetPos;

        stereoRenderer.shouldRender = true;

        yield return new WaitForSeconds(0.1f);
        isColliding = false;
    }
}
