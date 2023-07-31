using BVH;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using UnityEngine;
using Valve.VR;

using Color = UnityEngine.Color;

public class VRController : MonoBehaviour
{
    private static readonly uint MAX_DEVICE_COUNT = 16;
    private static readonly float HEAD_COSINE_DEVIATION_THRESHOLD = 0.5f;
    private static float MAX_HEAD_TO_WAIST_DISTANCE = 0.8f;

    public GameObject SteamVRSetup;
    public SkeletonRenderer Skeleton;
    public bool ComputeOffsetsHands = false;

    // SteamVR controllers
    private SteamVR_Behaviour_Pose SteamVRControllerLeft;
    private SteamVR_Behaviour_Pose SteamVRControllerRight;

    // Devices
    public GameObject HMD { get; private set; }
    public GameObject HMDJoint { get; private set; }
    public GameObject ControllerLeft { get; private set; }
    public GameObject ControllerLeftJoint { get; private set; }
    public GameObject ControllerRight { get; private set; }
    public GameObject ControllerRightJoint { get; private set; }
    public GameObject TrackerRoot { get; private set; }
    public GameObject TrackerRootJoint { get; private set; }
    public GameObject TrackerLeft { get; private set; }
    public GameObject TrackerLeftJoint { get; private set; }
    public GameObject TrackerRight { get; private set; }
    public GameObject TrackerRightJoint { get; private set; }

    // Mirror
    private GameObject Mirror;
    private DisplayMirror DisplayMirror;

    // Input actions
    private SteamVR_Action_Boolean SteamVRTrigger = SteamVR_Input.GetAction<SteamVR_Action_Boolean>("default", "GrabPinch");

    private uint NumControllerConnected, NumTrackersConnected;
    private int[] TrackerIndices = new int[3];
    // Whether the SteamVR controllers have been assigned
    private bool ControllersStarted = false;
    private bool ControllersIdentified = false;
    private bool WalkInAvatarStarted = false;
    private bool WalkInAvatarDone = false;
    private float CooldownTime = -10000.0f;

    private void Start()
    {
        // Create and place mirror
        Mirror = Instantiate(Resources.Load("Prefabs/DisplayMirror"), transform) as GameObject;
        Mirror.name = "DisplayMirror";
        Mirror.transform.localPosition = new Vector3(0.0f, 1.5f, 2.0f);
        Mirror.transform.localEulerAngles = new Vector3(0.0f, 180.0f, 0.0f);
        Mirror.transform.localScale = new Vector3(0.25f, 0.25f, 0.1f);
        Mirror.SetActive(true);
        DisplayMirror = Mirror.transform.Find("Mirror").GetComponent<DisplayMirror>();

        // Find Devices
        SteamVRSetup.SetActive(true);
        Transform t;
        // Init devices with one random object (here we still have to assign the right devices for each end effector)
        string[] headNames = { "HMD", "Neck", "Head", "Camera", "Camera (eye)" };
        t = FindChildRecursive(SteamVRSetup.transform, headNames);
        if (t != null)
        {
            HMD = t.gameObject;
        }

        string[] handLeftNames = { "LeftHand", "HandLeft", "Controller (left)", "Controller1" };
        t = FindChildRecursive(SteamVRSetup.transform, handLeftNames);
        if (t != null)
        {
            ControllerLeft = t.gameObject;
        }

        string[] handRightNames = { "RightHand", "HandRight", "Controller (right)", "Controller2" };
        t = FindChildRecursive(SteamVRSetup.transform, handRightNames);
        if (t != null)
        {
            ControllerRight = t.gameObject;
        }

        string[] pelvisNames = { "Pelvis", "Root", "Hips", "Tracker (root)", "Tracker1" };
        t = FindChildRecursive(SteamVRSetup.transform, pelvisNames);
        if (t != null)
        {
            TrackerRoot = t.gameObject;
        }

        string[] footLeftNames = { "LeftFoot", "FootLeft", "Tracker (left)", "Tracker2" };
        t = FindChildRecursive(SteamVRSetup.transform, footLeftNames);
        if (t != null)
        {
            TrackerLeft = t.gameObject;
        }

        string[] footRightNames = { "RightFoot", "FootRight", "Tracker (right)", "Tracker3" };
        t = FindChildRecursive(SteamVRSetup.transform, footRightNames);
        if (t != null)
        {
            TrackerRight = t.gameObject;
        }

        // Layers
#if UNITY_EDITOR
        // Create a layer for everything that should not be seen from the HMD
        LayerUtils.CreateLayer("NotHMD");
        if (HMD)
        {
            LayerUtils.HideLayerInCamera("NotHMD", HMD.GetComponentInChildren<Camera>());
        }
        // Hide HMD model from HMD
        if (HMD)
        {
            if (HMD.transform.Find("Model"))
            {
                LayerUtils.MoveToLayer(HMD.transform.Find("Model").gameObject, "NotHMD");
            }
        }
#endif
    }

    private void Update()
    {
        if (CooldownTime + 0.5f > Time.time)
        {
            return;
        }

        if (!ControllersStarted)
        {
            bool allDevicesConnected = DetectDevices(DisplayMirror);
            if (allDevicesConnected) StartControllers();
            return;
        }

        if (!ControllersIdentified && ControllersStarted && IsTriggerDown())
        {
            IdentifyDevices(DisplayMirror);
            StartControllers();
            ControllersIdentified = true;
            CooldownTime = Time.time;
            return;
        }

        if (!WalkInAvatarStarted && ControllersStarted && ControllersIdentified)
        {
            Transform hips = Skeleton.GetBoneTransform(HumanBodyBones.Hips);
            hips.position = new Vector3(0.0f, TrackerRoot.transform.position.y, 0.0f);
            hips.rotation = Quaternion.Euler(0.0f, 180.0f, 0.0f) * hips.rotation;
            string message1 = "Measures were correctly captured!";
            string message2 = "\n\n\n\n\nSetting up root... Please, stand on a T-pose inside the avatar shown. Press TRIGGER when ready!";
            DisplayMirror.ShowTextAgain(message1, new Color(0.0f, 1.0f, 0.0f, 0.5f), 2, message2, new Color(1.0f, 1.0f, 1.0f, 0.5f), 0, true);
            WalkInAvatarStarted = true;
            CooldownTime = Time.time;
            return;
        }

        if (!WalkInAvatarDone && WalkInAvatarStarted && IsTriggerDown())
        {
            SetupJoints();
            FindObjectOfType<TrackersCalibrator>().Calibrate();
            WalkInAvatarDone = true;
            enabled = false;
        }
    }

    private void SetupJoints()
    {
        Transform head = Skeleton.GetBoneTransform(HumanBodyBones.Head);
        Transform leftHand = Skeleton.GetBoneTransform(HumanBodyBones.LeftHand);
        Transform rightHand = Skeleton.GetBoneTransform(HumanBodyBones.RightHand);
        Transform leftToes = Skeleton.GetBoneTransform(HumanBodyBones.LeftToes);
        Transform rightToes = Skeleton.GetBoneTransform(HumanBodyBones.RightToes);
        Transform pelvis = Skeleton.GetBoneTransform(HumanBodyBones.Hips);
        // Create joints
        HMDJoint = new GameObject("HMDJoint");
        HMDJoint.transform.parent = HMD.transform;
        HMDJoint.transform.localPosition = HMD.transform.InverseTransformVector(head.position - HMD.transform.position);
        HMDJoint.transform.localRotation = Quaternion.Inverse(HMD.transform.localRotation);
        ControllerLeftJoint = new GameObject("ControllerLeftJoint");
        ControllerLeftJoint.transform.parent = ControllerLeft.transform;
        if (ComputeOffsetsHands)
        {
            ControllerLeftJoint.transform.localPosition = ControllerLeft.transform.InverseTransformVector(leftHand.position - ControllerLeft.transform.position);
        }
        else
        {
            ControllerLeftJoint.transform.localPosition = new Vector3(0.0f, 0.0f, -0.175f);
        }
        ControllerLeftJoint.transform.localRotation = Quaternion.Inverse(ControllerLeft.transform.localRotation);
        ControllerRightJoint = new GameObject("ControllerRightJoint");
        ControllerRightJoint.transform.parent = ControllerRight.transform;
        if (ComputeOffsetsHands)
        {
            ControllerRightJoint.transform.localPosition = ControllerRight.transform.InverseTransformVector(rightHand.position - ControllerRight.transform.position);
        }
        else
        {
            ControllerRightJoint.transform.localPosition = new Vector3(0.0f, 0.0f, -0.175f);
        }
        ControllerRightJoint.transform.localRotation = Quaternion.Inverse(ControllerRight.transform.localRotation);
        TrackerRootJoint = new GameObject("TrackerRootJoint");
        TrackerRootJoint.transform.parent = TrackerRoot.transform;
        TrackerRootJoint.transform.localPosition = TrackerRoot.transform.InverseTransformVector(pelvis.position - TrackerRoot.transform.position);
        TrackerRootJoint.transform.localRotation = Quaternion.Inverse(TrackerRoot.transform.localRotation);
        TrackerLeftJoint = new GameObject("TrackerLeftJoint");
        TrackerLeftJoint.transform.parent = TrackerLeft.transform;
        TrackerLeftJoint.transform.localPosition = TrackerLeft.transform.InverseTransformVector(leftToes.position - TrackerLeft.transform.position);
        TrackerLeftJoint.transform.localRotation = Quaternion.Inverse(TrackerLeft.transform.localRotation);
        TrackerRightJoint = new GameObject("TrackerRightJoint");
        TrackerRightJoint.transform.parent = TrackerRight.transform;
        TrackerRightJoint.transform.localPosition = TrackerRight.transform.InverseTransformVector(rightToes.position - TrackerRight.transform.position);
        TrackerRightJoint.transform.localRotation = Quaternion.Inverse(TrackerRight.transform.localRotation);
    }

    private void StartControllers()
    {
        SteamVRControllerLeft = null;
        SteamVRControllerRight = null;
        if (ControllerLeft.activeInHierarchy)
        {
            SteamVRControllerLeft = ControllerLeft.GetComponent<SteamVR_Behaviour_Pose>();
            ControllersStarted = true;
        }
        if (ControllerRight.activeInHierarchy)
        {
            SteamVRControllerRight = ControllerRight.GetComponent<SteamVR_Behaviour_Pose>();
            ControllersStarted = true;
        }
    }

    // Finds connected devices and sets temporal device indices
    public bool DetectDevices(DisplayMirror displayMirror)
    {
        // Init global vars
        NumControllerConnected = 0;
        NumTrackersConnected = 0;

        // Get pose relative to the safe bounds defined by the user
        TrackedDevicePose_t[] trackedDevicePoses = new TrackedDevicePose_t[MAX_DEVICE_COUNT];
        if (OpenVR.Settings != null)
        {
            OpenVR.System.GetDeviceToAbsoluteTrackingPose(ETrackingUniverseOrigin.TrackingUniverseStanding, 0, trackedDevicePoses);
        }

        // Loop over connected devices
        for (uint i = 0; i < MAX_DEVICE_COUNT; ++i)
        {
            // deviceClass sometimes returns the wrong class for a device, hence we use a string
            /*ETrackedDeviceClass deviceClass = ETrackedDeviceClass.Invalid;
            if (OpenVR.Settings != null)
            {
                deviceClass = OpenVR.System.GetTrackedDeviceClass(i);
            }*/
            ETrackingResult status = trackedDevicePoses[i].eTrackingResult;
            var result = new System.Text.StringBuilder((int)64);
            var error = ETrackedPropertyError.TrackedProp_Success;
            if (OpenVR.System != null)
            {
                OpenVR.System.GetStringTrackedDeviceProperty(i, ETrackedDeviceProperty.Prop_RenderModelName_String, result, 64, ref error);
            }
            // Handle HMD
            if (result.ToString().Contains("hmd") && status == ETrackingResult.Running_OK)
            //else if (deviceClass == ETrackedDeviceClass.HMD && status == ETrackingResult.Running_OK)
            {
                continue;
            }
            // Handle controllers
            else if (result.ToString().Contains("controller") && status == ETrackingResult.Running_OK)
            //else if (deviceClass == ETrackedDeviceClass.Controller && status == ETrackingResult.Running_OK)
            {
                NumControllerConnected++;
            }
            // Handle trackers
            else if (result.ToString().Contains("tracker_vive") && status == ETrackingResult.Running_OK)
            //else if (deviceClass == ETrackedDeviceClass.GenericTracker && status == ETrackingResult.Running_OK)
            {
                TrackerIndices[NumTrackersConnected] = (int)i;
                NumTrackersConnected++;
            }
        }

        string message = string.Format("Found {0} controller(s) and {1} tracker(s).", NumControllerConnected, NumTrackersConnected);
        if (NumControllerConnected >= 2 && NumTrackersConnected >= 3)
        {
            string message2 = "Setting up device indices and taking some measures... Please, stand on a T-pose. Press TRIGGER when ready!";
            if (displayMirror)
            {
                displayMirror.ShowTextAgain(message, new Color(1.0f, 1.0f, 1.0f, 0.5f), 2, message2, new Color(1.0f, 1.0f, 1.0f, 0.5f), 0, true);
            }
        }
        else
        {
            message = message + " Please, connect more controllers and/or trackers.";
            if (displayMirror)
            {
                displayMirror.ShowText(message, new Color(1.0f, 0.0f, 0.0f, 0.5f), 0, true);
            }
        }

        if (NumControllerConnected < 2 || NumTrackersConnected < 3) return false; // not enough devices

        // Asign correct indices
        return SetDevicesIndex();
    }

    // Assigns device indices and enables/disables GameObjects accordingly
    public bool SetDevicesIndex()
    {
        if (NumControllerConnected >= 2)
        {
            ControllerLeft.SetActive(true);
        }
        else
        {
            ControllerLeft.SetActive(false);
        }

        if (NumControllerConnected >= 1)
        {
            ControllerRight.SetActive(true);
        }
        else
        {
            ControllerRight.SetActive(false);
        }

        if (NumTrackersConnected >= 1)
        {
            TrackerRoot.SetActive(true);
            TrackerRoot.GetComponent<SteamVR_TrackedObject>().index = (SteamVR_TrackedObject.EIndex)TrackerIndices[0];
        }
        else
        {
            TrackerRoot.SetActive(false);
        }

        if (NumTrackersConnected >= 3)
        {
            TrackerLeft.SetActive(true);
            TrackerLeft.GetComponent<SteamVR_TrackedObject>().index = (SteamVR_TrackedObject.EIndex)TrackerIndices[2];
        }
        else
        {
            TrackerLeft.SetActive(false);
        }

        if (NumTrackersConnected >= 2)
        {
            TrackerRight.SetActive(true);
            TrackerRight.GetComponent<SteamVR_TrackedObject>().index = (SteamVR_TrackedObject.EIndex)TrackerIndices[1];
        }
        else
        {
            TrackerRight.SetActive(false);
        }

        return true;
    }

    // Fixes indices of tracked devices
    public bool IdentifyDevices(DisplayMirror displayMirror)
    {
        if (NumControllerConnected + NumTrackersConnected < 2)
        {
            string message = "Not enough devices! Need at least two controllers and/or trackers.";
            if (displayMirror)
            {
                displayMirror.ShowText(message, new Color(1.0f, 0.0f, 0.0f, 0.5f), 2, true);
            }
            return false;
        }

        uint numPoints = 1 + NumControllerConnected + NumTrackersConnected;
        Vector3[] points = new Vector3[numPoints];
        GameObject[] deviceObjects = new GameObject[numPoints];
        //  points[0] = HMD position                deviceIndices[0] = HMD GameObject     
        //  points[1] = Controller 1 position       deviceIndices[1] = Controller 1 GameObject     
        //  points[2] = Controller 2 position       deviceIndices[2] = Controller 2 GameObject     
        // ...                                      ...
        //  points[n] = Controller n position       deviceIndices[n] = Controller n GameObject     
        //  points[n+1] = Tracker 1 position        deviceIndices[n+1] = Tracker 1 GameObject     
        //  points[n+2] = Tracker 2 position        deviceIndices[n+2] = Tracker 2 GameObject     
        //  points[n+3] = Tracker 3 position        deviceIndices[n+3] = Tracker 3 GameObject     
        // ...                                      ...
        uint controllerIndex0 = 1;
        uint trackerIndex0 = controllerIndex0 + NumControllerConnected;

        uint controllerIndex = controllerIndex0;
        uint trackerIndex = trackerIndex0;

        if (HMD.activeInHierarchy)
        {
            points[0] = HMD.transform.position;
            deviceObjects[0] = HMD;
        }

        if (ControllerLeft.activeInHierarchy)
        {
            points[controllerIndex] = ControllerLeft.transform.position;
            deviceObjects[controllerIndex] = ControllerLeft;
            controllerIndex++;
        }
        if (ControllerRight.activeInHierarchy)
        {
            points[controllerIndex] = ControllerRight.transform.position;
            deviceObjects[controllerIndex] = ControllerRight;
            controllerIndex++;
        }
        Debug.Assert(controllerIndex == NumControllerConnected + 1);

        if (TrackerRoot.activeInHierarchy)
        {
            points[trackerIndex] = TrackerRoot.transform.position;
            deviceObjects[trackerIndex] = TrackerRoot;
            trackerIndex++;
        }
        if (TrackerLeft.activeInHierarchy)
        {
            points[trackerIndex] = TrackerLeft.transform.position;
            deviceObjects[trackerIndex] = TrackerLeft;
            trackerIndex++;
        }
        if (TrackerRight.activeInHierarchy)
        {
            points[trackerIndex] = TrackerRight.transform.position;
            deviceObjects[trackerIndex] = TrackerRight;
            trackerIndex++;
        }
        Debug.Assert(trackerIndex == NumControllerConnected + NumTrackersConnected + 1);

        // Fit plane to tracked objects locations
        float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
        bool res = FitPlane(numPoints, points, ref a, ref b, ref c, ref d);
        if (!res)
        {
            string message = "Could not identify tracked objects! Make sure you're standing on a T-pose.";
            if (displayMirror)
            {
                displayMirror.ShowText(message, new Color(1.0f, 0.0f, 0.0f, 0.5f), 2, true);
            }
            return false;
        }
        Vector3 n = new Vector3(a, b, c);
        n = Vector3.Normalize(n);

        // Get HMD forward vector
        Vector3 f = HMD.transform.forward;
        f = Vector3.Normalize(f);

        //  Compute deviation between plane normal and HMD forward
        float deviation = Vector3.Dot(n, f);

        // Make sure plane points in the same direction 
        if (System.Math.Abs(deviation) < HEAD_COSINE_DEVIATION_THRESHOLD)
        {
            string message = "Your head is not aligned with the rest of your body! Make sure you're standing on a T-pose.";
            Debug.Log(message + "\n");
            if (displayMirror)
            {
                displayMirror.ShowText(message, new Color(1.0f, 0.0f, 0.0f, 0.5f), 2, true);
            }
            //ViveInput.blockControllers(false); // no need, displayPanel will unblock them
            return false;
        }
        if (deviation < 0.0f)
        {
            n = -1.0f * n;
        }

        // Get a point on the plane
        Vector3 p = new Vector3(0.0f, 0.0f, -d / c);

        // Project points on plane
        Vector3[] projectedPoints = new Vector3[numPoints];
        for (uint i = 0; i < numPoints; ++i)
        {
            Vector3 t = points[i] - p;
            float dist = Vector3.Dot(t, n);
            projectedPoints[i] = points[i] - dist * n;
        }

        // Build u,v coordinate system
        Vector3 v = Vector3.up;
        Vector3 u = Vector3.Cross(v, n);
        float u0 = Vector3.Dot(projectedPoints[0], u); // HMD
        float v0 = Vector3.Dot(projectedPoints[0], v);

        // Get uv coordinates
        Vector2[] planePoints = new Vector2[numPoints];
        planePoints[0] = new Vector2(0.0f, 0.0f); // HMD will be origin of uv space
        for (uint i = 1; i < numPoints; ++i)
        {
            float u_coord = Vector3.Dot(projectedPoints[i], u) - u0;
            float v_coord = Vector3.Dot(projectedPoints[i], v) - v0;
            Vector2 uv = new Vector2(u_coord, v_coord);
            planePoints[i] = uv;
        }

        // Identify controllers/trackers according to uv coordinates
        for (uint i = 0; i < NumControllerConnected; ++i)
        {
            if (planePoints[controllerIndex0 + i].x < 0.0f)
            {
                ControllerLeft = deviceObjects[controllerIndex0 + i];
            }
            else
            {
                ControllerRight = deviceObjects[controllerIndex0 + i];
            }
        }
        for (uint i = 0; i < NumTrackersConnected; ++i)
        {
            if (System.Math.Abs(planePoints[trackerIndex0 + i].y) < MAX_HEAD_TO_WAIST_DISTANCE)
            {
                TrackerRoot = deviceObjects[trackerIndex0 + i];
            }
            else if (planePoints[trackerIndex0 + i].x < 0.0f)
            {
                TrackerLeft = deviceObjects[trackerIndex0 + i];
            }
            else
            {
                TrackerRight = deviceObjects[trackerIndex0 + i];
            }
        }
        // Update variables
        displayMirror.CleanText();
        return true;
    }

    // Grab inputs
    public bool IsTriggerDown()
    {
        return
            (SteamVRControllerLeft != null && SteamVRTrigger.GetLastStateDown(SteamVRControllerLeft.inputSource)) ||
            (SteamVRControllerRight != null && SteamVRTrigger.GetLastStateDown(SteamVRControllerRight.inputSource));
    }

    public static Transform FindChildRecursive(Transform parent, string[] names)
    {
        foreach (Transform c in parent.GetComponentsInChildren<Transform>(true))
        {
            for (int i = 0; i < names.Length; i++)
            {
                if (c.name == names[i])
                {
                    return c;
                }
            }
        }
        return null;
    }

    // Fit least square errors plane to set of points
    public static bool FitPlane(uint numPoints, Vector3[] points, ref float a, ref float b, ref float c, ref float d)
    {
        // Check input
        if (numPoints < 3)
        {
            return false;
        }

        // Compute the mean of the points
        Vector3 mean = new Vector3(0.0f, 0.0f, 0.0f);
        for (uint i = 0; i < numPoints; ++i)
        {
            mean += points[i];
        }
        mean /= numPoints;

        // Compute the linear system matrix and vector elements
        float xxSum = 0.0f, xySum = 0.0f, xzSum = 0.0f, yySum = 0.0f, yzSum = 0.0f;
        for (uint i = 0; i < numPoints; ++i)
        {
            Vector3 diff = points[i] - mean;
            xxSum += diff[0] * diff[0];
            xySum += diff[0] * diff[1];
            xzSum += diff[0] * diff[2];
            yySum += diff[1] * diff[1];
            yzSum += diff[1] * diff[2];
        }

        // Solve the linear system
        float det = xxSum * yySum - xySum * xySum;
        if (det != 0.0f)
        {
            // Compute the fitted plane
            a = (yySum * xzSum - xySum * yzSum) / det;
            b = (xxSum * yzSum - xySum * xzSum) / det;
            c = -1;
            d = -a * mean[0] - b * mean[1] + mean[2];
            return true;
        }
        else
        {
            return false;
        }
    }
}
