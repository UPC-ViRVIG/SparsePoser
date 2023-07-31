using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using System;

namespace BVH
{
    [DefaultExecutionOrder(10000)]
    public class SkeletonRenderer : MonoBehaviour
    {
        public PythonCommunication Controller;
        public Vector3 TargetWorldForward = new Vector3(0, 0, 1);
        [Tooltip("Local vector (axis) pointing in the forward direction of the character")] public Vector3 BVHForwardLocalVector = new Vector3(0, 0, 1);
        public Vector3 TargetWorldUp = new Vector3(0, 1, 0);
        [Tooltip("Local vector (axis) pointing in the up direction of the character")] public Vector3 BVHUpLocalVector = new Vector3(0, 1, 0);
        public BoneMap[] BoneMapping;

        private Transform[] SkeletonTransforms;

        private Quaternion HipsCorrection;

        private void OnEnable()
        {
            Controller.OnSkeletonTransformUpdated += OnSkeletonTransformUpdated;
        }

        private void OnDisable()
        {
            Controller.OnSkeletonTransformUpdated -= OnSkeletonTransformUpdated;
        }

        private void Start()
        {
            InitSkeleton();
        }

        private void InitSkeleton()
        {
            // Create Skeleton
            BVHImporter importer = new BVHImporter();
            BVHAnimation tpose = importer.Import(Controller.TPoseBVH, 1.0f, true);
            SkeletonTransforms = new Transform[Controller.SkeletonTransforms.Length];
            for (int j = 0; j < SkeletonTransforms.Length; j++)
            {
                // Joints
                Skeleton.Joint joint = tpose.Skeleton.Joints[j];
                Transform t = (new GameObject()).transform;
                t.name = joint.Name;
                t.SetParent(j == 0 ? transform : SkeletonTransforms[joint.ParentIndex], false);
                t.localPosition = joint.LocalOffset;
                tpose.GetWorldPositionAndRotation(joint, 0, out quaternion worldRot, out _);
                t.rotation = worldRot;
                SkeletonTransforms[j] = t;
                // Visual
                Transform visual = (new GameObject()).transform;
                visual.name = "Visual";
                visual.SetParent(t, false);
                visual.localScale = new Vector3(0.1f, 0.1f, 0.1f);
                visual.localPosition = Vector3.zero;
                visual.localRotation = Quaternion.identity;
                // Sphere
                Color color = Color.yellow;
                Transform sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
                sphere.name = "Sphere";
                sphere.SetParent(visual, false);
                sphere.localScale = Vector3.one;
                sphere.localPosition = Vector3.zero;
                sphere.localRotation = Quaternion.identity;
                sphere.GetComponent<MeshRenderer>().material.color = color;
                // Capsule
                Transform capsule = GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
                capsule.name = "Capsule";
                capsule.SetParent(SkeletonTransforms[joint.ParentIndex].Find("Visual"), false);
                float distance = Vector3.Distance(t.position, t.parent.position) * (1.0f / visual.localScale.y) * 0.5f;
                capsule.localScale = new Vector3(0.5f, distance, 0.5f);
                Vector3 up = (t.position - t.parent.position).normalized;
                capsule.localPosition = t.parent.InverseTransformDirection(up) * distance;
                capsule.localRotation = Quaternion.Inverse(t.parent.rotation) * Quaternion.LookRotation(new Vector3(-up.y, up.x, 0.0f), up);
                capsule.GetComponent<MeshRenderer>().material.color = color;
            }
            // Hips Correction
            float3 targetWorldForward = TargetWorldForward;
            float3 targetWorldUp = TargetWorldUp;
            quaternion hipsRot = quaternion.identity;
            if (UnityToName(HumanBodyBones.Hips, out string jointName) &&
                tpose.Skeleton.Find(jointName, out Skeleton.Joint skjoint))
            {
                // Get the rotation for the first frame of the animation
                hipsRot = tpose.GetWorldRotation(skjoint, 0);
            }
            else Debug.Assert(false, "Joint not found");
            float3 sourceWorldForward = math.mul(hipsRot, BVHForwardLocalVector);
            float3 sourceWorldUp = math.mul(hipsRot, BVHUpLocalVector);
            quaternion targetLookAt = quaternion.LookRotation(targetWorldForward, targetWorldUp);
            quaternion sourceLookAt = quaternion.LookRotation(sourceWorldForward, sourceWorldUp);
            HipsCorrection = math.mul(sourceLookAt, math.inverse(targetLookAt));
        }

        private void OnSkeletonTransformUpdated()
        {
            for (int i = 0; i < SkeletonTransforms.Length; i++)
            {
                SkeletonTransforms[i].localPosition = Controller.SkeletonTransforms[i].localPosition;
                SkeletonTransforms[i].rotation = Quaternion.Inverse(HipsCorrection) * Controller.SkeletonTransforms[i].rotation;
            }
            SkeletonTransforms[0].position = Quaternion.Inverse(HipsCorrection) * Controller.SkeletonTransforms[0].position;
        }

        public bool UnityToName(HumanBodyBones humanBodyBone, out string name)
        {
            name = "";
            for (int i = 0; i < BoneMapping.Length; ++i)
            {
                if (BoneMapping[i].Unity == humanBodyBone)
                {
                    name = BoneMapping[i].Name;
                    return true;
                }
            }
            Debug.Assert(false, "Not found");
            return false;
        }

        public Transform GetBoneTransform(HumanBodyBones humanBodyBones)
        {
            if (UnityToName(humanBodyBones, out string jointName))
            {
                foreach (Transform t in SkeletonTransforms)
                {
                    if (t.name == jointName)
                    {
                        return t;
                    }
                }
            }
            return null;
        }

        private void OnValidate()
        {
            if (math.abs(math.length(TargetWorldForward)) < 1E-3f)
            {
                Debug.LogWarning("ForwardLocalVector is too close to zero. Object: " + name);
            }
            if (math.abs(math.length(BVHForwardLocalVector)) < 1E-3f)
            {
                Debug.LogWarning("BVHForwardLocalVector is too close to zero. Object: " + name);
            }
            if (BoneMapping == null || BoneMapping.Length != 22)
            {
                BVHImporter importer = new BVHImporter();
                BVHAnimation tpose = importer.Import(Controller.TPoseBVH, 1.0f, true);
                BoneMapping = new BoneMap[tpose.Skeleton.Joints.Count];
                for (int i = 0; i < BoneMapping.Length; ++i)
                {
                    BoneMapping[i].Name = tpose.Skeleton.Joints[i].Name;
                }
            }
        }

        [System.Serializable]
        public struct BoneMap
        {
            public string Name;
            public HumanBodyBones Unity;
        }
    }
}