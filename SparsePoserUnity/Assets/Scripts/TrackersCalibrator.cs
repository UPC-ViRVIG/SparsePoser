using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using BVH;

[DefaultExecutionOrder(-100)]
public class TrackersCalibrator : MonoBehaviour
{
    public bool CalibrateOnStart;
    public PythonCommunication Controller;
    public SkeletonRenderer SkeletonRenderer;
    public Transform Root, LFoot, RFoot, Head, LHand, RHand;
    public Transform RetRoot, RetLFoot, RetRFoot, RetHead, RetLHand, RetRHand;

    private quaternion[] SourceTPose = new quaternion[6];
    private quaternion[] InverseTargetTPose = new quaternion[6];
    private quaternion RootAlign;
    private quaternion InverseRootAlign;

    private void Start()
    {
        if (CalibrateOnStart)
        {
            Calibrate();
        }
    }

    // Call this function when the user is in T-Pose
    [ContextMenu("Calibrate")]
    public void Calibrate()
    {
        BVHImporter importer = new BVHImporter();
        BVHAnimation tposeAnimation = importer.Import(Controller.TPoseBVH, 1.0f, true);

        Skeleton skeleton = tposeAnimation.Skeleton;
        HumanBodyBones[] unity = new HumanBodyBones[] { HumanBodyBones.Hips, HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot,
                                                        HumanBodyBones.Head, HumanBodyBones.LeftHand, HumanBodyBones.RightHand };
        Transform[] trackers = new Transform[] { Root, LFoot, RFoot, Head, LHand, RHand };
        Debug.Assert(unity[0] == HumanBodyBones.Hips, "hips must be the first bone in the array, so that RootAlign is computed first");
        for (int i = 0; i < unity.Length; ++i)
        {
            if (SkeletonRenderer.UnityToName(unity[i], out string jointName) &&
                skeleton.Find(jointName, out Skeleton.Joint joint))
            {
                quaternion worldRot = tposeAnimation.GetWorldRotation(joint, 0);

                if (unity[i] == HumanBodyBones.Hips)
                {
                    // Root Alignment
                    float3 targetWorldForward = math.mul(Root.rotation, SkeletonRenderer.TargetWorldForward);
                    float3 targetWorldUp = math.mul(Root.rotation, SkeletonRenderer.TargetWorldUp);
                    float3 sourceWorldForward = math.mul(worldRot, SkeletonRenderer.BVHForwardLocalVector);
                    float3 sourceWorldUp = math.mul(worldRot, SkeletonRenderer.BVHUpLocalVector);
                    quaternion targetLookAt = quaternion.LookRotation(targetWorldForward, targetWorldUp);
                    quaternion sourceLookAt = quaternion.LookRotation(sourceWorldForward, sourceWorldUp);
                    // RootAlign -> [target tpose world] to [source tpose world]
                    RootAlign = math.mul(sourceLookAt, math.inverse(targetLookAt));
                    InverseRootAlign = math.inverse(RootAlign);
                }

                // TargetTPose[i]  -> [target local] to [target tpose world]
                // InverseSourceTPose[i] -> [source tpose world] to [source local]
                InverseTargetTPose[i] = math.inverse(trackers[i].rotation);
                SourceTPose[i] = worldRot;
            }
            else Debug.Assert(false, "Joint not found: " + unity[i]);
        }
        Controller.SetCalibrated();
    }

    private void Update()
    {
        // Retarget Positions
        float3 rootPos = Root.position;
        float3 retRootPos = math.mul(RootAlign, rootPos);
        RetRoot.position = retRootPos;
        RetLFoot.position = math.mul(RootAlign, (float3)LFoot.position - rootPos) + retRootPos;
        RetRFoot.position = math.mul(RootAlign, (float3)RFoot.position - rootPos) + retRootPos;
        RetHead.position = math.mul(RootAlign, (float3)Head.position - rootPos) + retRootPos;
        RetLHand.position = math.mul(RootAlign, (float3)LHand.position - rootPos) + retRootPos;
        RetRHand.position = math.mul(RootAlign, (float3)RHand.position - rootPos) + retRootPos;
        // Retarget Rotations -> [source local] to [source world]
        // [source local] -> [source world] -> [target world] -> [target local] -> [target world] -> [source world]
        RetRoot.rotation = math.mul(RootAlign, math.mul(Root.rotation, math.mul(InverseTargetTPose[0], math.mul(InverseRootAlign, SourceTPose[0]))));
        RetLFoot.rotation = math.mul(RootAlign, math.mul(LFoot.rotation, math.mul(InverseTargetTPose[1], math.mul(InverseRootAlign, SourceTPose[1]))));
        RetRFoot.rotation = math.mul(RootAlign, math.mul(RFoot.rotation, math.mul(InverseTargetTPose[2], math.mul(InverseRootAlign, SourceTPose[2]))));
        RetHead.rotation = math.mul(RootAlign, math.mul(Head.rotation, math.mul(InverseTargetTPose[3], math.mul(InverseRootAlign, SourceTPose[3]))));
        RetLHand.rotation = math.mul(RootAlign, math.mul(LHand.rotation, math.mul(InverseTargetTPose[4], math.mul(InverseRootAlign, SourceTPose[4]))));
        RetRHand.rotation = math.mul(RootAlign, math.mul(RHand.rotation, math.mul(InverseTargetTPose[5], math.mul(InverseRootAlign, SourceTPose[5]))));
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.yellow;
        GizmosExtensions.DrawLine(Root.position, Head.position, 3.0f);
        GizmosExtensions.DrawLine(Root.position, LHand.position, 3.0f);
        GizmosExtensions.DrawLine(Root.position, RHand.position, 3.0f);
        GizmosExtensions.DrawLine(Root.position, LFoot.position, 3.0f);
        GizmosExtensions.DrawLine(Root.position, RFoot.position, 3.0f);
        Gizmos.color = Color.green;
        GizmosExtensions.DrawLine(RetRoot.position, RetHead.position, 3.0f);
        GizmosExtensions.DrawLine(RetRoot.position, RetLHand.position, 3.0f);
        GizmosExtensions.DrawLine(RetRoot.position, RetRHand.position, 3.0f);
        GizmosExtensions.DrawLine(RetRoot.position, RetLFoot.position, 3.0f);
        GizmosExtensions.DrawLine(RetRoot.position, RetRFoot.position, 3.0f);
    }
}
