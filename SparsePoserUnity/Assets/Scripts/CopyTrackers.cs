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
    public Transform C_Root, C_LFoot, C_RFoot, C_Head, C_LHand, C_RHand;
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
            if (VR == null) return;

            if (VR.TrackerRightJoint != null)
            {
                if (C_Root.gameObject.activeSelf)
                {
                    C_Root.gameObject.SetActive(false);
                    C_LFoot.gameObject.SetActive(false);
                    C_RFoot.gameObject.SetActive(false);
                    C_Head.gameObject.SetActive(false);
                    C_LHand.gameObject.SetActive(false);
                    C_RHand.gameObject.SetActive(false);
                    Root.gameObject.SetActive(true);
                    LFoot.gameObject.SetActive(true);
                    RFoot.gameObject.SetActive(true);
                    Head.gameObject.SetActive(true);
                    LHand.gameObject.SetActive(true);
                    RHand.gameObject.SetActive(true);
                }

                Root.SetPositionAndRotation(VR.TrackerRootJoint.transform.position, VR.TrackerRootJoint.transform.rotation);
                LFoot.SetPositionAndRotation(VR.TrackerLeftJoint.transform.position, VR.TrackerLeftJoint.transform.rotation);
                RFoot.SetPositionAndRotation(VR.TrackerRightJoint.transform.position, VR.TrackerRightJoint.transform.rotation);
                Head.SetPositionAndRotation(VR.HMDJoint.transform.position, VR.HMDJoint.transform.rotation);
                LHand.SetPositionAndRotation(VR.ControllerLeftJoint.transform.position, VR.ControllerLeftJoint.transform.rotation);
                RHand.SetPositionAndRotation(VR.ControllerRightJoint.transform.position, VR.ControllerRightJoint.transform.rotation);
            }
            else if (C_Root != null)
            {
                if (Root.gameObject.activeSelf)
                {
                    Root.gameObject.SetActive(false);
                    LFoot.gameObject.SetActive(false);
                    RFoot.gameObject.SetActive(false);
                    Head.gameObject.SetActive(false);
                    LHand.gameObject.SetActive(false);
                    RHand.gameObject.SetActive(false);
                    C_Root.gameObject.SetActive(true);
                    C_LFoot.gameObject.SetActive(true);
                    C_RFoot.gameObject.SetActive(true);
                    C_Head.gameObject.SetActive(true);
                    C_LHand.gameObject.SetActive(true);
                    C_RHand.gameObject.SetActive(true);
                }

                C_Root.SetPositionAndRotation(VR.TrackerRoot.transform.position, VR.TrackerRoot.transform.rotation);
                C_LFoot.SetPositionAndRotation(VR.TrackerLeft.transform.position, VR.TrackerLeft.transform.rotation);
                C_RFoot.SetPositionAndRotation(VR.TrackerRight.transform.position, VR.TrackerRight.transform.rotation);
                C_Head.SetPositionAndRotation(VR.HMD.transform.position, VR.HMD.transform.rotation);
                C_LHand.SetPositionAndRotation(VR.ControllerLeft.transform.position, VR.ControllerLeft.transform.rotation);
                C_RHand.SetPositionAndRotation(VR.ControllerRight.transform.position, VR.ControllerRight.transform.rotation);
            }

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
