using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif


public class LayerUtils
{
    // Assign layers recursively
    public static void MoveToLayer(GameObject root, string layername)
    {
        int layerNo = LayerMask.NameToLayer(layername);
        root.layer = layerNo;
        foreach (Transform child in root.transform)
        {
            MoveToLayer(child.gameObject, layername);
        }
    }

    // ~ BIT-NOT      (010111 becomes 101000) 
    // & BIT-AND      (011 & 001 becomes 001) 
    // | BIT-OR       (011 | 001 becomes 011) 
    // ^ EXCLUSIVE-OR (011 ^ 001 becomes 010)

     // Turn on the bit using an OR operation
    public static void ShowLayerInCamera(string layername, Camera camera)
    {
        camera.cullingMask |= 1 << LayerMask.NameToLayer(layername);
    }
 
    // Turn off the bit using an AND operation with the complement of the shifted int
    public static void HideLayerInCamera(string layername, Camera camera)
    {
        camera.cullingMask &= ~(1 << LayerMask.NameToLayer(layername));
    }
 
    // Toggle the bit using a XOR operation
    public static void ToggleLayerInCamera(string layername, Camera camera)
    {
        camera.cullingMask ^= 1 << LayerMask.NameToLayer(layername);
    }
 
#if UNITY_EDITOR
    // Create a layer at the next available index. Returns silently if layer already exists.
    public static void CreateLayer(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new System.ArgumentNullException("name", "New layer name string is either null or empty.");
        }
        var tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
        var layerProps = tagManager.FindProperty("layers");
        var propCount = layerProps.arraySize;

        SerializedProperty firstEmptyProp = null;

        for (var i = 0; i < propCount; i++)
        {
            var layerProp = layerProps.GetArrayElementAtIndex(i);

            var stringValue = layerProp.stringValue;

            if (stringValue == name)
            {
                return;
            }

            if (i < 8 || stringValue != string.Empty)
            {
                continue;
            }

            if (firstEmptyProp == null)
            {
                firstEmptyProp = layerProp;
            }
        }

        if (firstEmptyProp == null)
        {
            UnityEngine.Debug.LogError("Maximum limit of " + propCount + " layers exceeded. Layer \"" + name + "\" not created.");
            return;
        }

        firstEmptyProp.stringValue = name;
        tagManager.ApplyModifiedProperties();
    }
#endif
}
