using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ComputeLocalVector : MonoBehaviour
{
    public Vector3 WorldVector;
    
    [ContextMenu("Compute Local Vector")]
    public void GetLocalVector()
    {
        Debug.Log(transform.InverseTransformDirection(WorldVector).ToString("F6"));
    }
}
