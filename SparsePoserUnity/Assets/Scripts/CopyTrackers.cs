using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;

[DefaultExecutionOrder(-1000)]
public class CopyTrackers : MonoBehaviour
{
    public VRController VR;
    public Transform Root, LFoot, RFoot, Head, LHand, RHand;
    public bool Record;
    public bool Read;
    public int ReadTargetFramerate = 90;

    private List<Vector3>[] RecordPos;
    private List<Quaternion>[] RecordRot;
    private int FrameIndex;

    private void Awake()
    {
        if (Record || Read)
        {
            RecordPos = new List<Vector3>[6];
            for (int i = 0; i < RecordPos.Length; ++i)
            {
                RecordPos[i] = new List<Vector3>();
            }
            RecordRot = new List<Quaternion>[6];
            for (int i = 0; i < RecordRot.Length; ++i)
            {
                RecordRot[i] = new List<Quaternion>();
            }
        }
        if (Read)
        {
            Application.targetFrameRate = ReadTargetFramerate;
            ReadData();
            ApplyReadTrackers();
            FindObjectOfType<TrackersCalibrator>().Calibrate();
        }
    }

    private void Update()
    {
        if (Read)
        {
            ApplyReadTrackers();
        }
        else
        {
            if (VR == null || VR.TrackerRightJoint == null) return;

            Root.SetPositionAndRotation(VR.TrackerRootJoint.transform.position, VR.TrackerRootJoint.transform.rotation);
            LFoot.SetPositionAndRotation(VR.TrackerLeftJoint.transform.position, VR.TrackerLeftJoint.transform.rotation);
            RFoot.SetPositionAndRotation(VR.TrackerRightJoint.transform.position, VR.TrackerRightJoint.transform.rotation);
            Head.SetPositionAndRotation(VR.HMDJoint.transform.position, VR.HMDJoint.transform.rotation);
            LHand.SetPositionAndRotation(VR.ControllerLeftJoint.transform.position, VR.ControllerLeftJoint.transform.rotation);
            RHand.SetPositionAndRotation(VR.ControllerRightJoint.transform.position, VR.ControllerRightJoint.transform.rotation);

            if (Record)
            {
                RecordPos[0].Add(VR.TrackerRootJoint.transform.position);
                RecordPos[1].Add(VR.TrackerLeftJoint.transform.position);
                RecordPos[2].Add(VR.TrackerRightJoint.transform.position);
                RecordPos[3].Add(VR.HMDJoint.transform.position);
                RecordPos[4].Add(VR.ControllerLeftJoint.transform.position);
                RecordPos[5].Add(VR.ControllerRightJoint.transform.position);

                RecordRot[0].Add(VR.TrackerRootJoint.transform.rotation);
                RecordRot[1].Add(VR.TrackerLeftJoint.transform.rotation);
                RecordRot[2].Add(VR.TrackerRightJoint.transform.rotation);
                RecordRot[3].Add(VR.HMDJoint.transform.rotation);
                RecordRot[4].Add(VR.ControllerLeftJoint.transform.rotation);
                RecordRot[5].Add(VR.ControllerRightJoint.transform.rotation);
            }
        }
    }

    private void ApplyReadTrackers()
    {
        Root.SetPositionAndRotation(RecordPos[0][FrameIndex], RecordRot[0][FrameIndex]);
        LFoot.SetPositionAndRotation(RecordPos[1][FrameIndex], RecordRot[1][FrameIndex]);
        RFoot.SetPositionAndRotation(RecordPos[2][FrameIndex], RecordRot[2][FrameIndex]);
        Head.SetPositionAndRotation(RecordPos[3][FrameIndex], RecordRot[3][FrameIndex]);
        LHand.SetPositionAndRotation(RecordPos[4][FrameIndex], RecordRot[4][FrameIndex]);
        RHand.SetPositionAndRotation(RecordPos[5][FrameIndex], RecordRot[5][FrameIndex]);
        //LAnkle.SetPositionAndRotation(RecordPos[6][FrameIndex], RecordRot[6][FrameIndex]);
        //RAnkle.SetPositionAndRotation(RecordPos[7][FrameIndex], RecordRot[7][FrameIndex]);

        FrameIndex = (FrameIndex + 1) % RecordPos[0].Count;
    }

    private void ReadData()
    {
        using (var stream = File.Open("htcvive_data", FileMode.Open))
        {
            long numFloats = stream.Length / sizeof(float);
            long numFrames = numFloats / (3 * 6 + 4 * 6); // Each frame contains 6 positions and 8 rotations
            using (var reader = new BinaryReader(stream, Encoding.UTF8, false))
            {
                for (int i = 0; i < RecordPos.Length; ++i)
                {
                    for (int f = 0; f < numFrames; ++f)
                    {
                        float x = reader.ReadSingle();
                        float y = reader.ReadSingle();
                        float z = reader.ReadSingle();
                        RecordPos[i].Add(new Vector3(x, y, z));
                    }
                }
                for (int i = 0; i < RecordRot.Length; ++i)
                {
                    for (int f = 0; f < numFrames; ++f)
                    {
                        float x = reader.ReadSingle();
                        float y = reader.ReadSingle();
                        float z = reader.ReadSingle();
                        float w = reader.ReadSingle();
                        RecordRot[i].Add(new Quaternion(x, y, z, w));
                    }
                }
            }
        }
    }

    private void OnDisable()
    {
        if (Record)
        {
            using (var stream = File.Open("htcvive_data", FileMode.Create))
            {
                using (var writer = new BinaryWriter(stream, Encoding.UTF8, false))
                {
                    for (int i = 0; i < RecordPos.Length; ++i)
                    {
                        foreach (Vector3 v in RecordPos[i])
                        {
                            writer.Write(v.x);
                            writer.Write(v.y);
                            writer.Write(v.z);
                        }
                    }
                    for (int i = 0; i < RecordRot.Length; ++i)
                    {
                        foreach (Quaternion q in RecordRot[i])
                        {
                            writer.Write(q.x);
                            writer.Write(q.y);
                            writer.Write(q.z);
                            writer.Write(q.w);
                        }
                    }
                }
            }
        }
    }
}
