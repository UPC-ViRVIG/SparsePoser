using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Unity.Mathematics;
using BVH;    

using Debug = UnityEngine.Debug;
using System.Threading;
using System;

public class PythonCommunication : MonoBehaviour
{
    public event Action OnSkeletonTransformUpdated;

    public Transform Root, LFoot, RFoot, Head, LHand, RHand;
    public TextAsset TPoseBVH;
    public float RotationSmooth = 30;
    public float PositionSmooth = 30;
    public bool EnforceRootRot = true;
    public bool DebugNetwork = false;
    public bool DebugSkeleton = true;

    private static readonly int WINDOW = 64;
    private static readonly int SPARSE = 6;

    private float[] PosBuffer = new float[WINDOW * SPARSE * 3]; // frames, joints, vec3
    private float[] RotBuffer = new float[WINDOW * SPARSE * 4]; // frames, joints, quat
    private byte[] SendBuffer = new byte[WINDOW * SPARSE * (3 + 4) * sizeof(float)];
    private byte[] ReceiveBuffer;
    private bool SentFrame;
    private Vector3 RootPosSent;
    private Quaternion RootRotSent;
    private Quaternion[] TargetRotations;
    private Vector3 TargetRootPos;
    private Quaternion TargetRootRot;

    private bool SocketConnected;
    private bool Calibrated;
    private Socket Client;

    public BVHAnimation TPose { get; private set; }
    public Transform[] SkeletonTransforms { get; private set; }

    private void Start()
    {
        // Initialize valid quaternions
        for (int f = 0; f < WINDOW; f++)
        {
            for (int j = 0; j < SPARSE; j++)
            {
                RotBuffer[f * SPARSE * 4 + j * 4 + 0] = 1.0f; // w (real part)
            }
        }
        SendAndReceivePython();
        // Import T-Pose BVH
        BVHImporter importer = new BVHImporter();
        TPose = importer.Import(TPoseBVH, 1.0f, true);
        // Create Skeleton
        SkeletonTransforms = new Transform[TPose.Skeleton.Joints.Count];
        for (int j = 0; j < TPose.Skeleton.Joints.Count; j++)
        {
            // Joints
            Skeleton.Joint joint = TPose.Skeleton.Joints[j];
            Transform t = (new GameObject()).transform;
            t.name = joint.Name;
            t.SetParent(j == 0 ? transform : SkeletonTransforms[joint.ParentIndex], false);
            t.localPosition = joint.LocalOffset;
            TPose.GetWorldPositionAndRotation(joint, 0, out quaternion worldRot, out _);
            t.rotation = worldRot;
            SkeletonTransforms[j] = t;
        }
        TargetRotations = new Quaternion[SkeletonTransforms.Length];
        for (int j = 0; j < SkeletonTransforms.Length; j++)
        {
            TargetRotations[j] = SkeletonTransforms[j].localRotation;
        }
        TargetRootPos = Root.position;
        TargetRootRot = Root.rotation;
        // Init Variables
        ReceiveBuffer = new byte[TPose.Skeleton.Joints.Count * 4 * sizeof(float)];
    }

    [ContextMenu("SetCalibrated()")]
    public void SetCalibrated()
    {
        Calibrated = true;
    }

    private async void SendAndReceivePython()
    {
        IPAddress ipAddress = IPAddress.Parse("127.0.0.1");
        int port = 2222;

        IPEndPoint ipEndPoint = new(ipAddress, port);

        Client = new(
            ipEndPoint.AddressFamily,
            SocketType.Stream,
            ProtocolType.Tcp);

        await Client.ConnectAsync(ipEndPoint);
        SocketConnected = true;
    }

    private void SendMessage()
    {
        if (SentFrame) return;
        // Fill SendBuffer array
        unsafe
        {
            fixed (byte* b = SendBuffer)
            {
                fixed (float* p = PosBuffer)
                {
                    fixed (float* r = RotBuffer)
                    {
                        for (int f = 0; f < WINDOW; ++f)
                        {
                            for (int j = 0; j < SPARSE; ++j)
                            {
                                int offset = f * SPARSE * (3 + 4) * sizeof(float) + j * (3 + 4) * sizeof(float);
                                int offsetPos = f * SPARSE * 3 + j * 3;
                                int offsetRot = f * SPARSE * 4 + j * 4;
                                *((float*)(b + offset + 0 * sizeof(float))) = *(p + offsetPos + 0);
                                *((float*)(b + offset + 1 * sizeof(float))) = *(p + offsetPos + 1);
                                *((float*)(b + offset + 2 * sizeof(float))) = *(p + offsetPos + 2);
                                *((float*)(b + offset + 3 * sizeof(float))) = *(r + offsetRot + 0);
                                *((float*)(b + offset + 4 * sizeof(float))) = *(r + offsetRot + 1);
                                *((float*)(b + offset + 5 * sizeof(float))) = *(r + offsetRot + 2);
                                *((float*)(b + offset + 6 * sizeof(float))) = *(r + offsetRot + 3);
                            }
                        }
                    }
                }
            }
        }
        _ = Client.SendAsync(SendBuffer, SocketFlags.None);
        SentFrame = true;
        RootPosSent = Root.position;
        RootRotSent = Root.rotation;
        if (DebugNetwork) Debug.Log("[Sent] " + Time.frameCount);
    }

    private async void ReceiveMessage()
    {
        if (!SentFrame || !SocketConnected) return; 
        int receivedCount = await Client.ReceiveAsync(ReceiveBuffer, SocketFlags.None);
        SentFrame = false;
        if (DebugNetwork) Debug.Log("[Received] " + Time.frameCount);
        Debug.Assert(receivedCount == ReceiveBuffer.Length, "Received number bytes does not match ReceiveBuffer length");
        unsafe
        {
            fixed (byte* b = ReceiveBuffer)
            {
                for (int j = 0; j < TPose.Skeleton.Joints.Count; ++j)
                {
                    int offset = j * 4 * sizeof(float);
                    float w = *((float*)(b + offset + 0 * sizeof(float)));
                    float x = *((float*)(b + offset + 1 * sizeof(float)));
                    float y = *((float*)(b + offset + 2 * sizeof(float)));
                    float z = *((float*)(b + offset + 3 * sizeof(float)));
                    if (SkeletonTransforms[j] != null)
                    {
                        quaternion newQ = new quaternion(-x, -y, z, w); // right to left-handed negate imaginary part (x,y,z), negate z again because BVH's z+ is Unity's z-
                        newQ = math.normalizesafe(newQ);
                        float4 currentRot = new float4(TargetRotations[j].x, TargetRotations[j].y,
                                                       TargetRotations[j].z, TargetRotations[j].w);
                        float4 newRot = new float4(newQ.value.x, newQ.value.y, newQ.value.z, newQ.value.w);
                        // if the distance (or, equivalently, maximal dot product) between the previous rotation
                        // and the flipped current quaternion is smaller than
                        // the distance between the previous rotation and the current quaternion, then flip the quaternion
                        if (math.dot(currentRot, -newRot) > math.dot(currentRot, newRot))
                        {
                            newRot = -newRot;
                        }
                        TargetRotations[j] = new Quaternion(newRot.x, newRot.y, newRot.z, newRot.w);
                    }
                }
            }
        }
        // Set Root
        //SkeletonTransforms[0].position = RootPosSent;
        TargetRootPos = RootPosSent;
        TargetRootRot = RootRotSent;
    }

    private float3 GetTrackerPosition(int index)
    {
        switch (index)
        {
            case 0: return Root.position;
            case 1: return math.mul(math.inverse(Root.rotation), LFoot.position - Root.position);
            case 2: return math.mul(math.inverse(Root.rotation), RFoot.position - Root.position);
            case 3: return math.mul(math.inverse(Root.rotation), Head.position - Root.position);
            case 4: return math.mul(math.inverse(Root.rotation), LHand.position - Root.position);
            case 5: return math.mul(math.inverse(Root.rotation), RHand.position - Root.position);
            default: Debug.Assert(false, "Invalid index"); return float3.zero;
        }
    }

    private quaternion GetTrackerRotation(int index)
    {
        switch (index)
        {
            case 0: return Root.rotation;
            case 1: return math.mul(math.inverse(Root.rotation), LFoot.rotation);
            case 2: return math.mul(math.inverse(Root.rotation), RFoot.rotation);
            case 3: return math.mul(math.inverse(Root.rotation), Head.rotation);
            case 4: return math.mul(math.inverse(Root.rotation), LHand.rotation);
            case 5: return math.mul(math.inverse(Root.rotation), RHand.rotation);
            default: Debug.Assert(false, "Invalid index"); return quaternion.identity;
        }
    }

    private void GetTrackersInfo()
    {
        // Shift frames
        int f;
        for (f = 0; f < WINDOW - 1; f++)
        {
            for (int j = 0; j < SPARSE; j++)
            {
                int offset = f * SPARSE * 3 + j * 3;
                int next = (f+1) * SPARSE * 3 + j * 3;
                PosBuffer[offset + 0] = PosBuffer[next + 0];
                PosBuffer[offset + 1] = PosBuffer[next + 1];
                PosBuffer[offset + 2] = PosBuffer[next + 2];
                offset = f * SPARSE * 4 + j * 4;
                next = (f + 1) * SPARSE * 4 + j * 4;
                RotBuffer[offset + 0] = RotBuffer[next + 0];
                RotBuffer[offset + 1] = RotBuffer[next + 1];
                RotBuffer[offset + 2] = RotBuffer[next + 2];
                RotBuffer[offset + 3] = RotBuffer[next + 3];
            }
        }
        // New frame
        f = WINDOW - 1;
        for (int j = 0; j < SPARSE; j++)
        {
            int offset = f * SPARSE * 3 + j * 3;
            float3 pos = GetTrackerPosition(j);
            PosBuffer[offset + 0] = pos.x;
            PosBuffer[offset + 1] = pos.y;
            PosBuffer[offset + 2] = -pos.z; // BVH's z+ axis is Unity's (z-) (Unity is left-handed BVH is right-handed)
            offset = f * SPARSE * 4 + j * 4;
            quaternion rot = GetTrackerRotation(j);
            // right to left-handed negate imaginary part (x,y,z), negate z again because BVH's z+ is Unity's z-
            float4 currentRot = new float4(rot.value.w, -rot.value.x,
                                           -rot.value.y, rot.value.z);
            // Ensure quaternion continuity
            int prevOffset = (f - 1) * SPARSE * 4 + j * 4;
            float4 prevRot = new float4(RotBuffer[prevOffset + 0], RotBuffer[prevOffset + 1],
                                        RotBuffer[prevOffset + 2], RotBuffer[prevOffset + 3]);
            // if the distance (or, equivalently, maximal dot product) between the previous rotation
            // and the flipped current quaternion is smaller than
            // the distance between the previous rotation and the current quaternion, then flip the quaternion
            if (math.dot(prevRot, -currentRot) > math.dot(prevRot, currentRot))
            {
                currentRot = -currentRot;
            }
            RotBuffer[offset + 0] = currentRot[0];
            RotBuffer[offset + 1] = currentRot[1];
            RotBuffer[offset + 2] = currentRot[2];
            RotBuffer[offset + 3] = currentRot[3];
        }
    }

    private void Update()
    {
        if (!SocketConnected || !Calibrated) return;
        GetTrackersInfo();
        SendMessage();
    }

    private void LateUpdate()
    {
        if (!SocketConnected || !Calibrated) return;
        ReceiveMessage();
        // Update Pose and Root
        for (int j = 0; j < SkeletonTransforms.Length; ++j)
        {
            SkeletonTransforms[j].localRotation = Quaternion.Slerp(SkeletonTransforms[j].localRotation, TargetRotations[j], Time.deltaTime * RotationSmooth);
        }
        if (EnforceRootRot) SkeletonTransforms[0].rotation = Quaternion.Slerp(SkeletonTransforms[0].rotation, TargetRootRot, Time.deltaTime * RotationSmooth);
        SkeletonTransforms[0].position = Vector3.Lerp(SkeletonTransforms[0].position, TargetRootPos, Time.deltaTime * PositionSmooth);
        // Update other scripts
        if (OnSkeletonTransformUpdated != null) OnSkeletonTransformUpdated.Invoke();
    }

    private void OnDisable()
    {
        if (SocketConnected)
        {
            Client.Shutdown(SocketShutdown.Both);
            Client.Close();
            SocketConnected = false;
        }
    }

    private void OnDrawGizmos()
    {
        // Skeleton
        if (SkeletonTransforms == null) return;

        if (DebugSkeleton)
        {
            Gizmos.color = Color.red;
            for (int i = 1; i < SkeletonTransforms.Length; i++)
            {
                Transform t = SkeletonTransforms[i];
                GizmosExtensions.DrawLine(t.parent.position, t.position, 3);
            }
        }
    }
}

    