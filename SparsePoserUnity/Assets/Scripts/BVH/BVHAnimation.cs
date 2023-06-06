using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using System;

namespace BVH
{
    using Joint = Skeleton.Joint;

    /// <summary>
    /// Stores the BVH animation data in Unity format.
    /// </summary>
    public class BVHAnimation
    {
        public float FrameTime { get; private set; }
        public Skeleton Skeleton { get; private set; }
        public List<EndSite> EndSites { get; private set; }
        public Frame[] Frames { get; private set; }

        public BVHAnimation()
        {
            Skeleton = new Skeleton();
            EndSites = new List<EndSite>();
        }

        public void SetFrameTime(float frameTime)
        {
            FrameTime = frameTime;
        }

        public void InitFrames(int numberFrames)
        {
            Frames = new Frame[numberFrames];
        }

        public void AddFrame(int index, Frame frame)
        {
            Frames[index] = frame;
        }

        public void AddJoint(Joint joint)
        {
            Skeleton.AddJoint(joint);
        }

        public void AddEndSite(EndSite endSite)
        {
            EndSites.Add(endSite);
        }

        //public void UpdateMecanimInformation(MotionMatchingData motionMatchingData)
        //{
        //    for (int i = 0; i < Skeleton.Joints.Count; i++)
        //    {
        //        Joint joint = Skeleton.Joints[i];
        //        if (motionMatchingData.GetMecanimBone(joint.Name, out HumanBodyBones bone))
        //        {
        //            joint.Type = bone;
        //            Skeleton.Joints[i] = joint;
        //        }
        //    }
        //}

        /// <summary>
        /// Apply forward kinematics to obtain the quaternion rotating from the local
        /// coordinate system of the joint to the world coordinate system.
        /// </summary>
        public quaternion GetWorldRotation(Joint joint, int frameIndex)
        {
            Frame frame = Frames[frameIndex];
            quaternion worldRot = quaternion.identity;

            while (joint.Index != 0) // while not root
            {
                worldRot = frame.LocalRotations[joint.Index] * worldRot;
                joint = Skeleton.GetParent(joint);
            }
            worldRot = frame.LocalRotations[0] * worldRot; // root

            return worldRot;
        }

        /// <summary>
        /// Apply forward kinematics
        /// </summary>
        public void GetWorldPositionAndRotation(Joint joint, int frameIndex, out quaternion worldRot, out float3 worldPos)
        {
            Frame frame = Frames[frameIndex];
            worldPos = float3.zero;
            worldRot = quaternion.identity;

            while (joint.Index != 0) // while not root
            {
                quaternion rot = frame.LocalRotations[joint.Index];
                worldRot = math.mul(rot, worldRot);
                worldPos = math.mul(rot, worldPos) + (float3)Skeleton.Joints[joint.Index].LocalOffset;
                joint = Skeleton.GetParent(joint);
            }
            quaternion rootRot = frame.LocalRotations[0];
            worldRot = math.mul(rootRot, worldRot); // root
            worldPos = math.mul(rootRot, worldPos) + (float3)frame.RootMotion;
        }

        public struct EndSite
        {
            public int ParentIndex;
            public Vector3 Offset;

            public EndSite(int parentIndex, Vector3 offset)
            {
                ParentIndex = parentIndex;
                Offset = offset;
            }
        }

        public struct Frame
        {
            public Vector3 RootMotion;
            public Quaternion[] LocalRotations;

            public Frame(Vector3 rootMotion, Quaternion[] localRotations)
            {
                RootMotion = rootMotion;
                LocalRotations = localRotations;
            }
        }
    }
}