using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using BVH;
using Unity.Mathematics;

public class BVHPlayback : MonoBehaviour
{
    public int Frame = 0;
    public int TargetFramerate = 60;
    public TextAsset BVH;
    public Transform Root, LFoot, RFoot, Head, LHand, RHand;

    private static readonly int ROOT_ID = 0;
    private static readonly int LFOOT_ID = 4;
    private static readonly int RFOOT_ID = 8;
    private static readonly int HEAD_ID = 13;
    private static readonly int LHAND_ID = 17;
    private static readonly int RHAND_ID = 21;

    private BVHAnimation Animation;

    private void Awake()
    {
        Application.targetFrameRate = TargetFramerate;
    }

    private void Start()
    {
        BVHImporter importer = new BVHImporter();
        Animation = importer.Import(BVH);
    }

    private void Update()
    {
        int animationLength = Animation.Frames.Length;

        PlayTracker(ROOT_ID, Root);
        PlayTracker(LFOOT_ID, LFoot);
        PlayTracker(RFOOT_ID, RFoot);
        PlayTracker(HEAD_ID, Head);
        PlayTracker(LHAND_ID, LHand);
        PlayTracker(RHAND_ID, RHand);

        Frame = (Frame + 1) % animationLength;
    }

    private void PlayTracker(int id, Transform t)
    {
        Animation.GetWorldPositionAndRotation(Animation.Skeleton.Joints[id], Frame, out quaternion worldRot, out float3 worldPos);
        t.position = worldPos;
        t.rotation = worldRot;
    }

    private void OnDrawGizmos()
    {
        if (Animation == null) return;

        int updateFrame = Frame - 1;

        Gizmos.color = Color.blue;
        foreach (Skeleton.Joint joint in Animation.Skeleton.Joints)
        {
            if (joint.Index == 0) continue;
            Animation.GetWorldPositionAndRotation(joint, updateFrame, out quaternion worldRot, out float3 worldPos);
            Skeleton.Joint parent = Animation.Skeleton.GetParent(joint);
            Animation.GetWorldPositionAndRotation(parent, updateFrame, out quaternion parentWorldRot, out float3 parentWorldPos);
            GizmosExtensions.DrawLine(worldPos, parentWorldPos, 3);
        }
    }
}
