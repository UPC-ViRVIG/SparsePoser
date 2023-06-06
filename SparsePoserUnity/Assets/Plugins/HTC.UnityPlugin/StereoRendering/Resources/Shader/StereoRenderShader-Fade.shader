//========= Copyright 2016-2017, HTC Corporation. All rights reserved. ===========

Shader "Custom/StereoRenderShader-Fade"
{
	Properties
	{
		_LeftEyeTexture("Left Eye Texture", 2D) = "white" {}
		_RightEyeTexture("Right Eye Texture", 2D) = "white" {}
		_FadeTexture("Fade Texture", 2D) = "white" {}
	}

	CGINCLUDE
	#include "UnityCG.cginc"
	#include "UnityInstancing.cginc"
	ENDCG

	SubShader
	{
		Tags{"Queue" = "Transparent" "RenderType" = "Transparent"}

		ZWrite Off

		CGPROGRAM
		#pragma surface surf Standard fullforwardshadows alpha:fade

		#pragma multi_compile __ STEREO_RENDER
		#pragma target 3.0

		sampler2D _LeftEyeTexture;
		sampler2D _RightEyeTexture;
		sampler2D _FadeTexture;

		struct Input
		{
			float2 uv_FadeTexture;
			float4 screenPos;
		};

		void surf(Input IN, inout SurfaceOutputStandard o)
		{
			float2 screenUV = IN.screenPos.xy / IN.screenPos.w;

#if UNITY_SINGLE_PASS_STEREO
			float4 scaleOffset = unity_StereoScaleOffset[unity_StereoEyeIndex];
			screenUV = (screenUV - scaleOffset.zw) / scaleOffset.xy;
#endif

			if (unity_StereoEyeIndex == 0)
			{
				fixed4 color = tex2D(_LeftEyeTexture, screenUV);

				o.Albedo = color.xyz;
				o.Alpha = tex2D(_FadeTexture, IN.uv_FadeTexture).a;
			}
			else
			{
				fixed4 color = tex2D(_RightEyeTexture, screenUV);

				o.Albedo = color.xyz;
				o.Alpha = tex2D(_FadeTexture, IN.uv_FadeTexture).a;
			}
		}

		ENDCG
	}

	Fallback "Standard"
}