//========= Copyright 2016-2017, HTC Corporation. All rights reserved. ===========

Shader "Custom/StereoRenderShader-SingleTexture"
{
	Properties
	{
		_MainTexture("Eye Texture", 2D) = "white" {}
	}

	CGINCLUDE
	#include "UnityCG.cginc"
	#include "UnityInstancing.cginc"
	ENDCG

	SubShader
	{
		Tags{ "RenderType" = "Opaque" }

		//Cull OFF

		CGPROGRAM
		#pragma surface surf Standard 

		#pragma multi_compile __ STEREO_RENDER
		#pragma target 3.0

		sampler2D _MainTexture;

		struct Input
		{
			float2 uv_MainTex;
			float4 screenPos;
		};

		void surf(Input IN, inout SurfaceOutputStandard o)
		{
			float2 screenUV = IN.screenPos.xy / IN.screenPos.w;
			fixed4 color = tex2D(_MainTexture, screenUV);
			o.Albedo = color.xyz;
		}

		ENDCG
	}

	Fallback "Diffuse"
}