//========= Copyright 2016-2018, HTC Corporation. All rights reserved. ===========

Shader "Custom/StereoRenderShader-Unlit"
{
	Properties
	{
		_LeftEyeTexture("Left Eye Texture", 2D) = "white" {}
		_RightEyeTexture("Right Eye Texture", 2D) = "white" {}
		_DisplayTex("_DisplayTex", 2D) = "white" {}
	}

	SubShader
	{
		Tags { "RenderType"="Opaque" }

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fog
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float4 screenPos : TEXCOORD1;
				float2 disp : TEXCOORD2;
			};

			sampler2D _LeftEyeTexture;
			sampler2D _RightEyeTexture;

			sampler2D _DisplayTex; ///
			float4 _DisplayTex_ST;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.screenPos = ComputeScreenPos(o.vertex);
				o.disp = TRANSFORM_TEX(v.uv, _DisplayTex);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				float2 screenUV = i.screenPos.xy / i.screenPos.w;

#if UNITY_SINGLE_PASS_STEREO
				float4 scaleOffset = unity_StereoScaleOffset[unity_StereoEyeIndex];
				screenUV = (screenUV - scaleOffset.zw) / scaleOffset.xy;
#endif

				fixed4 color = float4(0, 0, 0, 0);
				if (unity_StereoEyeIndex == 0)
				{
					color = tex2D(_LeftEyeTexture, screenUV);
				}
				else
				{
					color = tex2D(_RightEyeTexture, screenUV);
				}

				return color * saturate(tex2D(_DisplayTex, i.disp) + fixed4(0.5, 0.5, 0.5, 1.0));
			}
			ENDCG
		}
	}
}
