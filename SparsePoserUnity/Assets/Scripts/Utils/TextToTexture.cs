using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class TextToTexture
{
    private const int ASCII_START_OFFSET = 32;
    private Font customFont;
    private Texture2D fontTexture;
    private int fontCountX;
    private int fontCountY;
    private float[] kerningValues;
    private bool supportSpecialCharacters;
	private Color backgroundColor;
	private Color fontColor;

	public TextToTexture(Font customFont, int fontCountX, int fontCountY, PerCharacterKerning[] perCharacterKerning, bool supportSpecialCharacters, Color bColor, Color fColor)
    {
        this.customFont = customFont;
        fontTexture = (Texture2D) customFont.material.mainTexture;
        this.fontCountX = fontCountX;
        this.fontCountY = fontCountY;
        kerningValues = GetCharacterKerningValuesFromPerCharacterKerning(perCharacterKerning);
        this.supportSpecialCharacters = supportSpecialCharacters;
		backgroundColor = bColor;
		fontColor = fColor;
    }

	// trailing buffer is to allow for area where the character might be at the end
	public int CalcTextWidthPlusTrailingBuffer(string text, int decalTextureSize, float characterSize)
	{
		char letter;
		bool nextCharacterSpecial = false;
		float width = 0;
		float old_width = 0;
		int fontItemWidth = (int)((fontTexture.width / fontCountX) * characterSize);

		for (int n = 0; n < text.Length; n++)
		{
			letter = text[n];
			nextCharacterSpecial = false;

			if (letter == '\\' && supportSpecialCharacters)
			{
				nextCharacterSpecial = true;
				if (n + 1 < text.Length)
				{
					n++;
					letter = text[n];
					if (letter == 'n' || letter == 'r') // new line or return
					{
						width += fontItemWidth;
						old_width = Math.Max(width, old_width);
						width = 0;
					}
					else if (letter == 't') // tab
					{
						width = fontItemWidth * GetKerningValue(' ') * 4; // 4 spaces
					}
					else if (letter == '\\') // this allows for printing of \
					{
						nextCharacterSpecial = false; 
					}
				}
			}

			if (!nextCharacterSpecial && customFont.HasCharacter(letter)) 
			{
				if (n < text.Length - 1) 
				{
					width += fontItemWidth * GetKerningValue(letter);
				}
				else // last letter ignore kerning for buffer
				{ 
					width += fontItemWidth;
				}
			}
		}

		return (int)Math.Max(width, old_width);
	}

    // placementX and Y - placement within texture size, texture size = textureWidth and textureHeight (square)
    public Texture2D CreateTextToTexture(string text, int textPlacementX, int textPlacementY, int textureSize, float characterSize, float lineSpacing)
    {
		Texture2D txtTexture = CreateFillTexture2D(backgroundColor, textureSize, textureSize);
        int fontGridCellWidth = (int)(fontTexture.width / fontCountX);
        int fontGridCellHeight = (int)(fontTexture.height / fontCountY);
        int fontItemWidth = (int)(fontGridCellWidth * characterSize);
        int fontItemHeight = (int)(fontGridCellHeight * characterSize);
        Vector2 charTexturePos;
        Color[] charPixels;
        float textPosX = textPlacementX;
        float textPosY = textPlacementY;
        float charKerning;
        bool nextCharacterSpecial = false;
        char letter;

        for (int n = 0; n < text.Length; n++)
        {
            letter = text[n];
            nextCharacterSpecial = false;

            if (letter == '\\' && supportSpecialCharacters)
            {
                nextCharacterSpecial = true;
                if (n + 1 < text.Length)
                {
                    n++;
                    letter = text[n];
                    if (letter == 'n' || letter == 'r') // new line or return
                    {
                        textPosY -= fontItemHeight * lineSpacing;
                        textPosX = textPlacementX;
                    }
                    else if (letter == 't') // tab
                    {
                        textPosX += fontItemWidth * GetKerningValue(' ') * 4; // 4 spaces
                    }
					else if (letter == '\\') // this allows for printing of \
                    {
                        nextCharacterSpecial = false; 
                    }
                }
            }

            if (!nextCharacterSpecial && customFont.HasCharacter(letter))
            {
                charTexturePos = GetCharacterGridPosition(letter);
                charTexturePos.x *= fontGridCellWidth;
                charTexturePos.y *= fontGridCellHeight;
                charPixels = fontTexture.GetPixels((int)charTexturePos.x, fontTexture.height - (int)charTexturePos.y - fontGridCellHeight, fontGridCellWidth, fontGridCellHeight);
				charPixels = RecolorFont(charPixels, fontGridCellWidth, fontGridCellHeight);
				charPixels = ChangeDimensions(charPixels, fontGridCellWidth, fontGridCellHeight, fontItemWidth, fontItemHeight);

                txtTexture = AddPixelsToTextureIfClear(txtTexture, charPixels, (int)textPosX, (int)textPosY, fontItemWidth, fontItemHeight);
                charKerning = GetKerningValue(letter);
                textPosX += (fontItemWidth * charKerning); //add kerning here
            }
            else if (!nextCharacterSpecial)
            {
                Debug.Log("Letter Not Found:" + letter);
            }

        }
        txtTexture.Apply();
        return txtTexture;
    }

	public static Texture2D CreateFillTexture2D(Color color, int textureWidth, int textureHeight)
	{
		Texture2D texture = new Texture2D(textureWidth, textureHeight);
		int numOfPixels = texture.width * texture.height;
		Color[] colors = new Color[numOfPixels];
		for (int x = 0; x < numOfPixels; x++)
		{
			colors[x] = color;
		}
		texture.SetPixels(colors);
		return texture;
	}

	private Color[] RecolorFont(Color[] originalColors, int originalWidth, int originalHeight)
	{
		int pixelCount;
		for (int y = 0; y < originalHeight; y++)
		{
			for (int x = 0; x < originalWidth; x++)
			{
				pixelCount = x + (y * originalWidth);
				if (originalColors[pixelCount] == Color.clear)
				{
					originalColors[pixelCount] = backgroundColor;
				}
				else
				{
					originalColors[pixelCount] = fontColor;
				}
			}
		}
		return originalColors;
	}

    private Color[] ChangeDimensions(Color[] originalColors, int originalWidth, int originalHeight, int newWidth, int newHeight)
    {
        Color[] newColors;
        Texture2D originalTexture;
        int pixelCount;
        float u;
        float v;

        if (originalWidth == newWidth && originalHeight == newHeight)
        {
			newColors = originalColors;
        }
        else
        {
            newColors = new Color[newWidth * newHeight];
            originalTexture = new Texture2D(originalWidth, originalHeight);

            originalTexture.SetPixels(originalColors);
            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    pixelCount = x + (y * newWidth);
                    u = (float)x / newWidth;
                    v = (float)y / newHeight;
                    newColors[pixelCount] = originalTexture.GetPixelBilinear(u, v);
                }
            }
        }

        return newColors;
    }

    private Texture2D AddPixelsToTextureIfClear(Texture2D texture, Color[] newPixels, int placementX, int placementY, int placementWidth, int placementHeight)
    {
        int pixelCount = 0;
        Color[] currPixels;

        if (placementX + placementWidth < texture.width)
        {
            currPixels = texture.GetPixels(placementX, placementY, placementWidth, placementHeight);

            for (int y = 0; y < placementHeight; y++) //vert
            {
                for (int x = 0; x < placementWidth; x++) //horiz
                {
                    pixelCount = x + (y * placementWidth);
					if (currPixels[pixelCount] != backgroundColor) //if pixel is not empty take the previous value
                    {
                        //newPixels[pixelCount] = currPixels[pixelCount];
                    }
                }
            }

            texture.SetPixels(placementX, placementY, placementWidth, placementHeight, newPixels);
        }
        else
        {
            Debug.Log("Letter Falls Outside Bounds of Texture" + (placementX + placementWidth));
        }
        return texture;
    }

    private Vector2 GetCharacterGridPosition(char c)
    {
        int codeOffset = c - ASCII_START_OFFSET;

        return new Vector2(codeOffset % fontCountX, (int)codeOffset / fontCountX);
    }

    private float GetKerningValue(char c)
    {
		//Debug.Log (c);
        return kerningValues[((int)c) - ASCII_START_OFFSET];
    }

    private float[] GetCharacterKerningValuesFromPerCharacterKerning(PerCharacterKerning[] perCharacterKerning)
    {
        float[] perCharKerning = new float[128 - ASCII_START_OFFSET];
        int charCode;

        foreach (PerCharacterKerning perCharKerningObj in perCharacterKerning)
        {
            if (perCharKerningObj.First != "")
            {
                charCode = (int) perCharKerningObj.GetChar();
                if (charCode >= 0 && charCode - ASCII_START_OFFSET < perCharKerning.Length)
                {
                    perCharKerning[charCode - ASCII_START_OFFSET] = perCharKerningObj.GetKerningValue();
                }
            }
        }
        return perCharKerning;
    }
}

[System.Serializable]
public class PerCharacterKerning
{
    public string First = ""; // character
    public float Second;      // kerning ex: 0.201

    public PerCharacterKerning(string character, float kerning)
    {
        this.First = character;
        this.Second = kerning;
    }

    public PerCharacterKerning(char character, float kerning)
    {
        this.First = "" + character;
        this.Second = kerning;
    }

    public char GetChar()
    {
        return First[0];
    }

    public float GetKerningValue() 
	{
		return Second; 
	}

	public static PerCharacterKerning[] ArialCharacterKerning()
	{
		PerCharacterKerning[] perCharacterKerning = new PerCharacterKerning[] 
		{
			new PerCharacterKerning(" ",   0.201f),
			new PerCharacterKerning("!",   0.201f),
			new PerCharacterKerning("\"",  0.256f),
			new PerCharacterKerning("#",   0.401f),
			new PerCharacterKerning("$",   0.401f),
			new PerCharacterKerning("%",   0.641f),
			new PerCharacterKerning("&",   0.481f),
			new PerCharacterKerning("'",   0.138f),
			new PerCharacterKerning("(",   0.240f),
			new PerCharacterKerning(")",   0.240f),
			new PerCharacterKerning("*",   0.281f),
			new PerCharacterKerning("+",   0.421f),
			new PerCharacterKerning(",",   0.201f),
			new PerCharacterKerning("-",   0.240f),
			new PerCharacterKerning(".",   0.201f),
			new PerCharacterKerning("/",   0.201f),
			new PerCharacterKerning("0",   0.401f),
			new PerCharacterKerning("1",   0.353f),
			new PerCharacterKerning("2",   0.401f),
			new PerCharacterKerning("3",   0.401f),
			new PerCharacterKerning("4",   0.401f),
			new PerCharacterKerning("5",   0.401f),
			new PerCharacterKerning("6",   0.401f),
			new PerCharacterKerning("7",   0.401f),
			new PerCharacterKerning("8",   0.401f),
			new PerCharacterKerning("9",   0.401f),
			new PerCharacterKerning(":",   0.201f),
			new PerCharacterKerning(";",   0.201f),
			new PerCharacterKerning("<",   0.421f),
			new PerCharacterKerning("=",   0.421f),
			new PerCharacterKerning(">",   0.421f),
			new PerCharacterKerning("?",   0.401f),
			new PerCharacterKerning("@",   0.731f),
			new PerCharacterKerning("A",   0.481f),
			new PerCharacterKerning("B",   0.481f),
			new PerCharacterKerning("C",   0.520f),
			new PerCharacterKerning("D",   0.481f),
			new PerCharacterKerning("E",   0.481f),
			new PerCharacterKerning("F",   0.440f),
			new PerCharacterKerning("G",   0.561f),
			new PerCharacterKerning("H",   0.520f),
			new PerCharacterKerning("I",   0.201f),
			new PerCharacterKerning("J",   0.360f),
			new PerCharacterKerning("K",   0.481f),
			new PerCharacterKerning("L",   0.401f),
			new PerCharacterKerning("M",   0.600f),
			new PerCharacterKerning("N",   0.520f),
			new PerCharacterKerning("O",   0.561f),
			new PerCharacterKerning("P",   0.481f),
			new PerCharacterKerning("Q",   0.561f),
			new PerCharacterKerning("R",   0.520f),
			new PerCharacterKerning("S",   0.481f),
			new PerCharacterKerning("T",   0.440f),
			new PerCharacterKerning("U",   0.520f),
			new PerCharacterKerning("V",   0.481f),
			new PerCharacterKerning("W",   0.680f),
			new PerCharacterKerning("X",   0.481f),
			new PerCharacterKerning("Y",   0.481f),
			new PerCharacterKerning("Z",   0.440f),
			new PerCharacterKerning("[",   0.201f),
			new PerCharacterKerning("\\",  0.201f),
			new PerCharacterKerning("]",   0.201f),
			new PerCharacterKerning("^",   0.338f),
			new PerCharacterKerning("_",   0.401f),
			new PerCharacterKerning("`",   0.240f),
			new PerCharacterKerning("a",   0.401f),
			new PerCharacterKerning("b",   0.401f),
			new PerCharacterKerning("c",   0.360f),
			new PerCharacterKerning("d",   0.401f),
			new PerCharacterKerning("e",   0.401f),
			new PerCharacterKerning("f",   0.189f),
			new PerCharacterKerning("g",   0.401f),
			new PerCharacterKerning("h",   0.401f),
			new PerCharacterKerning("i",   0.160f),
			new PerCharacterKerning("j",   0.160f),
			new PerCharacterKerning("k",   0.360f),
			new PerCharacterKerning("l",   0.160f),
			new PerCharacterKerning("m",   0.600f),
			new PerCharacterKerning("n",   0.401f),
			new PerCharacterKerning("o",   0.401f),
			new PerCharacterKerning("p",   0.401f),
			new PerCharacterKerning("q",   0.401f),
			new PerCharacterKerning("r",   0.240f),
			new PerCharacterKerning("s",   0.360f),
			new PerCharacterKerning("t",   0.201f),
			new PerCharacterKerning("u",   0.401f),
			new PerCharacterKerning("v",   0.360f),
			new PerCharacterKerning("w",   0.520f),
			new PerCharacterKerning("x",   0.360f),
			new PerCharacterKerning("y",   0.360f),
			new PerCharacterKerning("z",   0.360f),
			new PerCharacterKerning("{",   0.241f),
			new PerCharacterKerning("|",   0.188f),
			new PerCharacterKerning("}",   0.241f),
			new PerCharacterKerning("~",   0.421f)
		};

		return perCharacterKerning;
	}

	public static PerCharacterKerning[] DroidSansBoldCharacterKerning()
	{
		PerCharacterKerning[] perCharacterKerning = new PerCharacterKerning[] 
		{
			new PerCharacterKerning(" ",   0.188f),
			new PerCharacterKerning("!",   0.207f),
			new PerCharacterKerning("\"",  0.340f),
			new PerCharacterKerning("#",   0.466f),
			new PerCharacterKerning("$",   0.397f),
			new PerCharacterKerning("%",   0.635f),
			new PerCharacterKerning("&",   0.520f),
			new PerCharacterKerning("'",   0.192f),
			new PerCharacterKerning("(",   0.244f),
			new PerCharacterKerning(")",   0.244f),
			new PerCharacterKerning("*",   0.393f),
			new PerCharacterKerning("+",   0.397f),
			new PerCharacterKerning(",",   0.209f),
			new PerCharacterKerning("-",   0.232f),
			new PerCharacterKerning(".",   0.206f),
			new PerCharacterKerning("/",   0.298f),
			new PerCharacterKerning("0",   0.397f),
			new PerCharacterKerning("1",   0.397f),
			new PerCharacterKerning("2",   0.397f),
			new PerCharacterKerning("3",   0.397f),
			new PerCharacterKerning("4",   0.397f),
			new PerCharacterKerning("5",   0.397f),
			new PerCharacterKerning("6",   0.397f),
			new PerCharacterKerning("7",   0.397f),
			new PerCharacterKerning("8",   0.397f),
			new PerCharacterKerning("9",   0.397f),
			new PerCharacterKerning(":",   0.206f),
			new PerCharacterKerning(";",   0.209f),
			new PerCharacterKerning("<",   0.397f),
			new PerCharacterKerning("=",   0.397f),
			new PerCharacterKerning(">",   0.397f),
			new PerCharacterKerning("?",   0.331f),
			new PerCharacterKerning("@",   0.624f),
			new PerCharacterKerning("A",   0.468f),
			new PerCharacterKerning("B",   0.463f),
			new PerCharacterKerning("C",   0.446f),
			new PerCharacterKerning("D",   0.505f),
			new PerCharacterKerning("E",   0.404f),
			new PerCharacterKerning("F",   0.396f),
			new PerCharacterKerning("G",   0.522f),
			new PerCharacterKerning("H",   0.523f),
			new PerCharacterKerning("I",   0.281f),
			new PerCharacterKerning("J",   0.239f),
			new PerCharacterKerning("K",   0.457f),
			new PerCharacterKerning("L",   0.386f),
			new PerCharacterKerning("M",   0.658f),
			new PerCharacterKerning("N",   0.564f),
			new PerCharacterKerning("O",   0.545f),
			new PerCharacterKerning("P",   0.431f),
			new PerCharacterKerning("Q",   0.545f),
			new PerCharacterKerning("R",   0.454f),
			new PerCharacterKerning("S",   0.378f),
			new PerCharacterKerning("T",   0.402f),
			new PerCharacterKerning("U",   0.516f),
			new PerCharacterKerning("V",   0.440f),
			new PerCharacterKerning("W",   0.668f),
			new PerCharacterKerning("X",   0.452f),
			new PerCharacterKerning("Y",   0.421f),
			new PerCharacterKerning("Z",   0.389f),
			new PerCharacterKerning("[",   0.239f),
			new PerCharacterKerning("\\",  0.298f),
			new PerCharacterKerning("]",   0.239f),
			new PerCharacterKerning("^",   0.384f),
			new PerCharacterKerning("_",   0.297f),
			new PerCharacterKerning("`",   0.416f),
			new PerCharacterKerning("a",   0.414f),
			new PerCharacterKerning("b",   0.438f),
			new PerCharacterKerning("c",   0.360f),
			new PerCharacterKerning("d",   0.438f),
			new PerCharacterKerning("e",   0.419f),
			new PerCharacterKerning("f",   0.279f),
			new PerCharacterKerning("g",   0.404f),
			new PerCharacterKerning("h",   0.452f),
			new PerCharacterKerning("i",   0.220f),
			new PerCharacterKerning("j",   0.220f),
			new PerCharacterKerning("k",   0.425f),
			new PerCharacterKerning("l",   0.220f),
			new PerCharacterKerning("m",   0.679f),
			new PerCharacterKerning("n",   0.452f),
			new PerCharacterKerning("o",   0.432f),
			new PerCharacterKerning("p",   0.438f),
			new PerCharacterKerning("q",   0.438f),
			new PerCharacterKerning("r",   0.313f),
			new PerCharacterKerning("s",   0.347f),
			new PerCharacterKerning("t",   0.305f),
			new PerCharacterKerning("u",   0.452f),
			new PerCharacterKerning("v",   0.389f),
			new PerCharacterKerning("w",   0.581f),
			new PerCharacterKerning("x",   0.395f),
			new PerCharacterKerning("y",   0.389f),
			new PerCharacterKerning("z",   0.330f),
			new PerCharacterKerning("{",   0.262f),
			new PerCharacterKerning("|",   0.397f),
			new PerCharacterKerning("}",   0.262f),
			new PerCharacterKerning("~",   0.397f)
		};

		return perCharacterKerning;
	}
}

