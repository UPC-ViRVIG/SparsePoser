// original source from: http://wiki.unity3d.com/index.php/MirrorReflection4
// This is in fact just the Water script from Pro Standard Assets, just with refraction stuff removed.

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Valve.VR;

[ExecuteInEditMode]
public class DisplayMirror : MonoBehaviour
{
    public bool sideMirror = false;

    private const string DISPLAY_TEXTURE_ID = "_DisplayTex";

    /***************************************************************************************/

    private string oldMessage = "";
    private Texture oldTexture = null;

    private TMPro.TextMeshProUGUI textMesh;
    private Material material;

    /***************************************************************************************/

    // Mirror properties
    private Dictionary<Camera, Camera> m_ReflectionCameras = new Dictionary<Camera, Camera>();

    private RenderTexture m_ReflectionTextureLeft = null;
    private RenderTexture m_ReflectionTextureRight = null;

    /***************************************************************************************/


    void Awake()
    {
        // Find references among children
        GameObject textGO = transform.parent.Find("Display").GetChild(0).GetChild(0).gameObject; // Text GameObject
        textMesh = textGO.AddComponent<TMPro.TextMeshProUGUI>();
        textMesh.fontSize = 10.0f;
        textMesh.text = "";
        textMesh.rectTransform.offsetMin = Vector2.zero;
        textMesh.rectTransform.offsetMax = Vector2.zero;
        textMesh.color = Color.black;

#if UNITY_EDITOR
        // Create a layer for everything that should not be seen on the mirror
        LayerUtils.CreateLayer("NotMirror");
        LayerUtils.MoveToLayer(transform.parent.gameObject, "NotMirror");
#endif

        if (sideMirror) CleanText();
    }

    // Cleanup all the objects we possibly have created
    void OnDisable()
    {
        if (m_ReflectionTextureLeft)
        {
            DestroyImmediate(m_ReflectionTextureLeft);
            m_ReflectionTextureLeft = null;
        }

        if (m_ReflectionTextureRight)
        {
            DestroyImmediate(m_ReflectionTextureRight);
            m_ReflectionTextureRight = null;
        }

        foreach (var kvp in m_ReflectionCameras)
        {
            DestroyImmediate(((Camera)kvp.Value).gameObject);
        }
        m_ReflectionCameras.Clear();
    }

    /*************************************** Display ***************************************/

    public void CleanText()
    {
        textMesh.text = "";
        material.SetTexture(DISPLAY_TEXTURE_ID, null);
    }

    public void ShowText(string message, Color background, int secs, bool block = false, bool overwrite = true)
    {
        Texture2D bTexture = TextToTexture.CreateFillTexture2D(background, 1, 1);
        bTexture.Apply();
        try
        {
            StartCoroutine(ShowTextForSeconds(message, secs, bTexture, block, overwrite));
        }
        catch (System.Exception)
        {

        }
    }

    public void ShowTextAgain(string message, Color background, int secs, string message2, Color background2, int secs2, bool block = false, bool overwrite = true)
    {
        Texture2D bTexture = TextToTexture.CreateFillTexture2D(background, 1, 1);
        bTexture.Apply();
        Texture2D bTexture2 = TextToTexture.CreateFillTexture2D(background2, 1, 1);
        bTexture2.Apply();
        StartCoroutine(ShowTextForSecondsAgain(message, secs, bTexture, message2, secs2, bTexture2, block, overwrite));
    }

    /*********************************** Display helpers ***********************************/

    private IEnumerator ShowTextForSeconds(string message, int secs, Texture2D text, bool block = false, bool overwrite = true)
    {
        if (material == null) material = GetComponent<Renderer>().sharedMaterial;

        if (overwrite)
        {
            oldMessage = textMesh.text;
            oldTexture = material.GetTexture(DISPLAY_TEXTURE_ID);
        }
        textMesh.text = "\n\n\n\n\n" + message;
        material.SetTexture(DISPLAY_TEXTURE_ID, text);
        if (secs > 0)
        {
            yield return new WaitForSeconds(secs);
            textMesh.text = "\n\n\n\n\n" + oldMessage;
            material.SetTexture(DISPLAY_TEXTURE_ID, oldTexture);
        }
    }

    private IEnumerator ShowTextForSecondsAgain(string message, int secs, Texture2D text, string message2, int secs2, Texture2D text2, bool block = false, bool overwrite = true)
    {
        if (material == null) material = GetComponent<Renderer>().sharedMaterial;

        if (overwrite)
        {
            oldMessage = textMesh.text;
            oldTexture = material.GetTexture(DISPLAY_TEXTURE_ID);
        }
        textMesh.text = "\n\n\n\n\n" + message;
        material.SetTexture(DISPLAY_TEXTURE_ID, text);
        if (secs > 0)
        {
            yield return new WaitForSeconds(secs);
            textMesh.text = "\n\n\n\n\n" + oldMessage;
            material.SetTexture(DISPLAY_TEXTURE_ID, oldTexture);
        }

        if (overwrite)
        {
            oldMessage = textMesh.text;
            oldTexture = material.GetTexture(DISPLAY_TEXTURE_ID);
        }
        textMesh.text = "\n\n\n\n\n" + message2;
        material.SetTexture(DISPLAY_TEXTURE_ID, text2);
        if (secs2 > 0)
        {
            yield return new WaitForSeconds(secs2);
            textMesh.text = "\n\n\n\n\n" + oldMessage;
            material.SetTexture(DISPLAY_TEXTURE_ID, oldTexture);
        }
    }
}