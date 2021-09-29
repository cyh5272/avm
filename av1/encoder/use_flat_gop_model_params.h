/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License and the
 * Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear License was
 * not distributed with this source code in the LICENSE file, you can obtain it
 * at aomedia.org/license/software-license/bsd-3-c-c/.  If the Alliance for Open Media Patent
 * License 1.0 was not distributed with this source code in the PATENTS file, you
 * can obtain it at aomedia.org/license/patent-license/.
 */

#ifndef AOM_AV1_ENCODER_USE_FLAT_GOP_MODEL_PARAMS_H_
#define AOM_AV1_ENCODER_USE_FLAT_GOP_MODEL_PARAMS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "av1/encoder/ml.h"

// A binary classifier that returns true (score > 0) if it is better to use a
// flat GOP structure, rather than a GOP structure that uses ALT-REFs and
// internal ARFs.

#define NUM_FEATURES 21
#define NUM_HIDDEN_LAYERS 1
#define NUM_HIDDEN_NODES_LAYER0 48
#define NUM_LABELS 1

static const float
    av1_use_flat_gop_nn_weights_layer0[NUM_FEATURES *
                                       NUM_HIDDEN_NODES_LAYER0] = {
      0.3801f,  -2.1832f, 1.7469f,  2.0130f,  2.1264f,  -0.7293f, -0.2814f,
      0.0692f,  -4.6589f, -1.4591f, 0.3023f,  -0.4310f, -0.1911f, -0.8284f,
      -1.3322f, -0.4621f, -0.1148f, -0.3531f, -0.0794f, -0.3114f, -0.1664f,
      -0.1615f, 0.2913f,  -0.0394f, -0.0620f, 0.1845f,  0.0204f,  -0.2124f,
      -0.1233f, -0.1685f, 0.1215f,  -0.2372f, -0.2865f, -0.1976f, 0.2137f,
      -0.1318f, -0.0324f, 0.0415f,  -0.1172f, 0.1077f,  -0.1135f, -0.2462f,
      -0.0743f, -0.1584f, -0.3267f, -0.0566f, -0.1615f, -0.3931f, -0.5200f,
      -0.1786f, -0.1811f, -0.2812f, -0.1986f, -0.4393f, -0.3941f, -0.2500f,
      -0.2029f, -0.4605f, -0.4973f, -0.2238f, -0.2599f, -0.1951f, -0.2034f,
      -0.3186f, -0.1368f, -0.5076f, -0.4718f, -0.1815f, -0.3338f, -0.0550f,
      -0.3920f, -0.5328f, -0.1658f, -0.2194f, -0.2867f, -0.0916f, -0.1678f,
      -0.1760f, -0.5055f, -0.2322f, -0.4668f, -0.0121f, -0.3903f, -0.2721f,
      -0.1306f, 0.1199f,  0.2894f,  0.1098f,  -0.0155f, -0.0844f, 0.0421f,
      -0.2364f, -0.1073f, -0.0878f, -0.2146f, -0.1713f, -0.2283f, 0.0342f,
      0.0394f,  -0.2808f, -0.0048f, 0.2640f,  -0.1371f, 0.1709f,  0.0155f,
      -0.3614f, -0.1843f, -0.3215f, -0.3121f, -0.2609f, -0.0254f, -0.2474f,
      -0.4674f, -0.3674f, -0.2076f, 0.0149f,  -0.3304f, -0.2678f, -0.0465f,
      -0.1326f, -0.4504f, -0.5101f, -0.1280f, -0.0416f, -0.4296f, -0.4568f,
      -0.6762f, -2.8105f, 0.7249f,  1.4288f,  1.3731f,  0.3034f,  0.1841f,
      -0.0912f, -0.1508f, 1.2637f,  -0.2009f, 0.3236f,  -0.2500f, -0.0736f,
      0.8655f,  -0.2599f, 0.1150f,  -0.0368f, -0.1122f, -0.7650f, -0.2004f,
      -0.0891f, -0.3832f, -0.2576f, -0.3532f, -0.1735f, -0.4018f, -0.0265f,
      -0.2988f, 0.2555f,  -0.1041f, -0.3391f, -0.5316f, -0.0171f, -0.3232f,
      -0.0565f, -0.3359f, -0.1842f, -0.0582f, 0.0073f,  -0.0278f, -0.5517f,
      0.0892f,  -0.1354f, 0.0548f,  -0.0401f, -0.1697f, 0.0432f,  0.0832f,
      -0.3538f, 0.2602f,  -0.0066f, -0.2130f, -0.3085f, 0.0025f,  0.2464f,
      -0.0103f, -0.3082f, -0.1136f, -0.2359f, -0.3421f, 0.1335f,  -0.3016f,
      -1.0355f, -1.0572f, -0.3316f, -0.1235f, -0.3730f, -0.1751f, -0.1921f,
      0.0031f,  -0.6297f, -0.5179f, 0.1082f,  -0.3130f, -0.1120f, -0.5430f,
      -0.1782f, 0.0534f,  -0.1052f, 0.1471f,  -0.7156f, -0.5453f, -0.5437f,
      1.8709f,  1.9696f,  -1.0343f, -0.3150f, -0.8399f, -0.0052f, -0.1123f,
      -0.1059f, 0.6755f,  1.2593f,  -0.2512f, -0.2053f, 0.0835f,  0.3261f,
      -0.0172f, 0.1230f,  -0.3687f, 0.1993f,  0.9390f,  -0.0165f, 0.6856f,
      -0.4372f, -0.4041f, -0.2869f, -0.3871f, -0.3587f, -0.2418f, 0.0518f,
      0.0110f,  -1.4713f, -0.1307f, -0.3246f, -0.5091f, -0.4652f, -0.4288f,
      -0.0763f, -0.1755f, 0.0662f,  -0.3026f, -0.4462f, -0.4123f, -0.2891f,
      -0.2251f, -0.4925f, -0.3820f, -0.1840f, -0.2878f, -0.1973f, -0.1010f,
      -0.1622f, -0.3108f, -0.5292f, -0.1017f, -0.0607f, -0.2426f, -0.6406f,
      -0.3834f, -0.2313f, -0.2433f, -0.1773f, -0.1581f, -0.3295f, -0.3799f,
      -0.4447f, -0.2389f, -0.4231f, -0.1498f, -0.0181f, -0.4429f, -0.3515f,
      0.0425f,  -0.5280f, -0.3462f, -0.3659f, 0.0153f,  -0.1002f, -0.5057f,
      -0.2134f, -0.2859f, -0.1988f, -0.4758f, 0.0967f,  -0.4784f, 0.1868f,
      -0.4387f, -1.3376f, -0.4452f, 0.3837f,  0.1698f,  -0.7076f, -0.4320f,
      0.0382f,  -1.8053f, -0.6589f, 0.1406f,  -0.4340f, 0.0641f,  -0.2558f,
      -0.4496f, -0.5003f, -0.6241f, -0.2217f, -0.8312f, -0.6793f, -0.3563f,
      0.5153f,  -0.7851f, 1.0570f,  0.9702f,  0.5238f,  -0.6932f, -0.4443f,
      0.0407f,  -3.0961f, -0.8461f, 0.0562f,  -0.0642f, 0.2471f,  -0.5911f,
      -0.7715f, -0.1574f, -0.0375f, -0.1951f, -0.3097f, -0.2040f, 0.0128f,
      -0.0918f, -0.0698f, -0.0970f, -0.2946f, -0.1723f, -0.2569f, -0.4382f,
      -0.5174f, -0.2058f, -0.2973f, -0.0858f, -0.2526f, -0.2648f, -0.2339f,
      -0.3474f, 0.0607f,  0.0272f,  -0.3142f, -0.1306f, -0.4938f, -0.1894f,
      -0.0551f, -0.1061f, -0.1613f, -0.1942f, 0.0590f,  -0.2009f, -0.1286f,
      -0.2035f, -0.0393f, -0.0650f, -0.1110f, 0.0123f,  -0.1122f, -0.0246f,
      -0.2042f, 0.0411f,  -0.2771f, -0.0189f, 0.0927f,  0.0286f,  -0.1559f,
      -0.3217f, -0.1039f, 0.1471f,  0.2489f,  0.2085f,  -0.4199f, -0.2404f,
      0.0358f,  -0.7567f, -0.2413f, -0.3437f, -0.2433f, -0.3687f, -0.1194f,
      -0.4289f, -0.1138f, -0.0721f, -0.3461f, -0.0244f, -0.3530f, -0.2842f,
      -0.3823f, -0.1238f, -0.5475f, -0.2688f, -0.0073f, 0.0491f,  -0.4500f,
      0.0201f,  0.0303f,  -0.2160f, -0.4219f, -0.4831f, -0.4593f, -0.2304f,
      -0.2082f, -0.0367f, -0.5226f, -0.0082f, -0.1867f, -0.1812f, -0.2753f,
      2.6650f,  1.9698f,  -2.9425f, 1.2119f,  1.5000f,  0.3356f,  0.3905f,
      -0.2006f, -1.4038f, -1.0917f, 0.1423f,  -0.3528f, 0.0888f,  0.5802f,
      1.0977f,  0.1083f,  -0.0693f, -0.0784f, 0.4247f,  0.4108f,  0.4970f,
      -0.7290f, -0.1659f, -0.0517f, 0.0776f,  -0.0550f, -0.2374f, -0.4245f,
      -0.0165f, -0.6804f, -0.3211f, -0.3101f, -0.1883f, -0.0786f, -0.3971f,
      -0.4130f, -0.0606f, 0.1432f,  -0.0518f, -0.4179f, -0.4949f, -0.3451f,
      -0.7559f, -4.0792f, 1.5526f,  0.2824f,  0.6086f,  -0.2148f, 0.0959f,
      0.0506f,  -5.5176f, -3.9702f, 0.1597f,  -0.1760f, -0.0627f, 0.1657f,
      -1.2996f, -0.2899f, -0.0600f, -0.0531f, -1.5160f, -0.4837f, -1.6961f,
      -0.1134f, -0.1838f, -0.3071f, -0.4215f, -0.4184f, 0.0192f,  -0.2128f,
      -0.3094f, -0.2607f, -0.4855f, -0.1881f, 0.0258f,  -0.5085f, -0.3630f,
      -0.4824f, -0.3762f, -0.3324f, -0.1134f, -0.3350f, 0.0217f,  -0.2803f,
      -0.5669f, -0.5674f, -0.5441f, -0.5965f, -0.3062f, -0.4666f, -0.4079f,
      -0.0065f, -0.7566f, -0.3437f, -0.2474f, -0.2360f, -0.5683f, -0.3853f,
      -0.6670f, -0.4158f, -0.2831f, -0.3327f, -0.7419f, -0.6481f, -0.4004f,
      -0.4025f, -0.6405f, -0.4265f, -0.0167f, 0.3195f,  -0.0822f, -0.4350f,
      -0.0032f, -1.0448f, -0.4407f, 0.0488f,  0.0776f,  -0.3828f, -0.3380f,
      -0.2983f, -0.2220f, -0.4105f, -0.2312f, -0.4166f, -0.3258f, -0.1424f,
      -0.6588f, -0.9433f, 0.3402f,  0.5800f,  0.6368f,  -0.4298f, -0.5743f,
      0.0822f,  -1.0843f, -0.1645f, -0.1990f, 0.0255f,  -0.1039f, -0.3673f,
      0.4367f,  -0.5491f, -0.0932f, -0.0323f, -0.2405f, -0.2922f, -0.4019f,
      -0.4936f, -1.2338f, 0.4681f,  0.7454f,  0.8181f,  -0.3680f, -0.1613f,
      -0.0008f, -1.3326f, -0.0667f, 0.1569f,  -0.0978f, -0.3229f, -0.4222f,
      0.0330f,  0.1064f,  -0.1325f, 0.0121f,  -0.3976f, -0.2254f, -0.3942f,
      -0.4771f, -0.1887f, 0.1020f,  0.3331f,  0.3098f,  -0.1256f, -0.4736f,
      0.0295f,  -0.3919f, -0.0931f, -0.2484f, -0.4629f, -0.2800f, -0.2851f,
      -0.2243f, -0.3958f, -0.3053f, -0.6585f, -0.1159f, -0.2330f, -0.1989f,
      0.2273f,  0.1963f,  0.0283f,  0.0198f,  -0.1298f, -0.0627f, -0.2753f,
      -0.1552f, 0.2734f,  -0.0551f, -0.2927f, -0.3772f, -0.4522f, -0.0786f,
      0.0079f,  0.1664f,  -0.0228f, -0.2908f, -0.1714f, 0.1223f,  -0.0680f,
      -0.5048f, -0.0852f, -0.4653f, -0.5142f, -0.1818f, -0.1659f, 0.0678f,
      -0.1296f, 0.0295f,  -0.3487f, -0.1224f, -0.2690f, -0.3217f, -0.1957f,
      -0.3196f, -0.4530f, -0.1746f, -0.2307f, -0.0504f, -0.0131f, -0.4613f,
      -0.1476f, -0.5596f, -0.3829f, -0.4302f, -0.2910f, -0.2182f, -0.0811f,
      -0.3967f, -0.3912f, -0.0371f, -0.1109f, -0.0793f, -0.2063f, -0.0060f,
      -0.0236f, -0.4098f, -0.0276f, -0.3352f, -0.1888f, -0.2439f, -0.3748f,
      0.0371f,  0.8460f,  -0.5547f, -1.2680f, -1.1623f, -0.1740f, -0.4815f,
      -0.0294f, 4.4764f,  0.3716f,  -0.2826f, -0.0549f, -0.2937f, 0.0632f,
      0.0686f,  -0.4681f, -0.2555f, -0.2427f, -0.2261f, -0.1567f, -0.5199f,
      -0.4079f, -0.0801f, -0.2075f, -0.3956f, -0.0307f, -0.3150f, -0.3490f,
      -0.0379f, 0.3060f,  -0.1775f, -0.1651f, 0.0677f,  -0.1947f, 0.0032f,
      -0.2014f, -0.1575f, -0.1289f, -0.0250f, -0.0762f, -0.2324f, -0.2895f,
      -0.4531f, -0.4601f, -0.1718f, -0.3139f, -0.4350f, 0.0346f,  -0.0891f,
      -0.1581f, 0.2123f,  -0.1074f, 0.0221f,  0.0951f,  0.1161f,  0.0245f,
      -0.0701f, -0.1677f, -0.4170f, -0.2214f, -0.3419f, -0.4873f, -0.0701f,
      -0.0613f, -0.1031f, 0.0141f,  -0.1299f, -0.3953f, -0.2182f, -0.2679f,
      -0.0141f, 0.3392f,  -0.0722f, -0.2390f, 0.1638f,  -0.1596f, -0.1527f,
      -0.3581f, -0.4037f, -0.0736f, 0.0397f,  -0.1288f, -0.1362f, -0.0249f,
      -0.5099f, -0.4040f, -0.1893f, -0.0298f, -0.1332f, -0.1693f, -0.3301f,
      -0.1058f, -0.1414f, -0.5737f, -0.2342f, -0.2560f, -0.3834f, -0.0917f,
      -0.1334f, -0.5077f, -0.3666f, -0.2515f, -0.4824f, -0.4714f, -0.5723f,
      -0.1361f, -0.5244f, -0.2468f, 0.0237f,  -0.1862f, -0.3124f, -0.0183f,
      -0.4662f, -0.4444f, -0.5400f, -0.1730f, -0.0123f, -0.2134f, -0.1024f,
      -0.0172f, -0.4430f, -0.1403f, -0.0751f, -0.2403f, -0.2100f, -0.0678f,
      2.4232f,  1.9825f,  0.1260f,  1.9972f,  2.8061f,  0.3916f,  0.1842f,
      -0.2603f, -1.6092f, -1.6037f, 0.1475f,  0.0516f,  -0.2593f, 0.0359f,
      -0.1802f, 0.0159f,  -0.0529f, -0.0983f, 0.7638f,  0.5529f,  0.9662f,
      -0.4049f, -0.6372f, 0.4907f,  0.7360f,  0.9271f,  -0.6879f, -0.1067f,
      0.0323f,  -1.8447f, 0.2176f,  -0.1047f, -0.0048f, -0.1031f, -0.7931f,
      -0.3059f, -0.4595f, -0.1287f, -0.4031f, 0.1441f,  -0.6651f, 0.2530f,
      -0.4572f, -0.0614f, 0.0345f,  -0.0008f, 0.0333f,  -0.3431f, 0.0538f,
      -0.2691f, 0.2930f,  -0.0820f, -0.0979f, -0.0307f, 0.1713f,  0.0783f,
      -0.4337f, -0.2702f, -0.1677f, -0.1719f, -0.4669f, -0.2847f, -0.4495f,
      -0.3692f, -0.2641f, -0.2833f, -0.1168f, -0.0523f, -0.2368f, -0.4922f,
      -0.3453f, -0.4452f, -0.5212f, 0.0412f,  -0.3310f, -0.2656f, -0.4903f,
      -0.3854f, -0.1009f, -0.1038f, -0.2350f, -0.4430f, -0.5097f, -0.1755f,
      0.0110f,  -0.0712f, -0.0662f, -0.4493f, -0.2111f, -0.3402f, -0.3100f,
      -0.2525f, -0.1856f, -0.2689f, -0.4288f, -0.3912f, -0.0754f, -0.5191f,
      -0.0747f, -0.0626f, -0.4821f, -0.2014f, -0.3124f, -0.4858f, -0.1896f,
      1.0673f,  -0.8529f, 13.7564f, 18.7299f, 19.0062f, -1.1047f, -0.8654f,
      0.1089f,  -1.2958f, -0.7793f, 0.0780f,  -0.1679f, 0.0054f,  -1.2451f,
      -0.1287f, 0.0082f,  -0.2960f, -0.0442f, 2.3817f,  0.4716f,  1.3862f,
      -0.0782f, -0.1871f, -0.2596f, 0.0093f,  0.1451f,  -0.1124f, -0.2315f,
      -0.2677f, -0.1086f, 0.2216f,  0.2928f,  0.0391f,  0.0372f,  -0.2551f,
      0.0552f,  -0.1876f, -0.2361f, -0.1889f, -0.0279f, 0.1204f,  0.2016f,
      -0.5787f, -0.5830f, 0.0530f,  -0.1452f, -0.4899f, -0.2937f, 0.1430f,
      -0.2752f, -0.2320f, -0.1908f, -0.5538f, -0.0858f, -0.1378f, -0.1505f,
      -0.3908f, -0.4732f, -0.3018f, 0.0244f,  -0.2392f, -0.2833f, -0.3997f,
      -0.4495f, -0.2570f, -0.3189f, -0.1534f, -0.1040f, -0.5497f, -0.3524f,
      -0.2053f, 0.2415f,  -0.5027f, 0.0288f,  -0.1904f, -0.2183f, -0.1062f,
      -0.3560f, 0.0165f,  -0.4601f, -0.2144f, -0.0439f, -0.4913f, -0.3160f,
      -0.1641f, 0.1010f,  -0.1044f, -0.4064f, -0.3580f, -0.4015f, 0.1010f,
      -0.1973f, 0.6392f,  -0.5177f, -0.0472f, -0.1526f, 0.1533f,  -0.0819f,
      -0.0252f, -0.0783f, 0.1301f,  0.0158f,  -0.2003f, -0.4700f, -0.2329f,
    };

static const float
    av1_use_flat_gop_nn_biases_layer0[NUM_HIDDEN_NODES_LAYER0] = {
      -1.113218f, 0.f,        -0.268537f, -0.268537f, 0.f,        -0.268534f,
      -0.40681f,  -0.268537f, -0.061835f, -0.614956f, 0.984277f,  -0.280228f,
      -0.354716f, -0.202312f, -0.772829f, -0.464005f, -0.230795f, 0.f,
      -0.124187f, -0.265949f, 0.325168f,  -0.359008f, -2.455546f, -0.229222f,
      -0.692233f, -0.29401f,  -0.632682f, -0.479061f, -0.166094f, 0.077291f,
      -0.235293f, -0.268537f, 0.167899f,  -0.141991f, -0.210089f, -0.177294f,
      -0.325401f, -0.268537f, 0.323627f,  -0.156593f, -0.218451f, -0.230792f,
      -0.268537f, 0.833177f,  0.f,        -0.353177f, -0.260953f, -0.209537f,
    };

static const float
    av1_use_flat_gop_nn_weights_layer1[NUM_HIDDEN_NODES_LAYER0 * NUM_LABELS] = {
      -0.024695f, 0.146668f,  -0.02723f,  0.034577f,  -0.255426f, 0.22402f,
      -0.112595f, -0.131262f, 0.091164f,  -0.045294f, 0.028304f,  -0.051683f,
      0.310497f,  -0.077786f, -0.047873f, -0.057205f, -0.065119f, 0.227417f,
      -0.051126f, -0.137241f, 0.035742f,  -0.058992f, -0.021466f, 0.107947f,
      -0.077183f, -0.04144f,  0.003568f,  -0.027656f, 0.038196f,  0.19684f,
      -0.128401f, 0.149629f,  0.024526f,  0.037376f,  0.090752f,  -0.061666f,
      -0.15743f,  0.057773f,  -0.010582f, 0.120997f,  0.060368f,  0.210028f,
      -0.192244f, -0.064764f, -0.237655f, 0.1852f,    -0.084281f, -0.010434f,
    };

static const float av1_use_flat_gop_nn_biases_layer1[NUM_LABELS] = {
  -0.672434f,
};

static const NN_CONFIG av1_use_flat_gop_nn_config = {
  NUM_FEATURES,
  NUM_LABELS,
  NUM_HIDDEN_LAYERS,
  {
      NUM_HIDDEN_NODES_LAYER0,
  },
  {
      av1_use_flat_gop_nn_weights_layer0,
      av1_use_flat_gop_nn_weights_layer1,
  },
  {
      av1_use_flat_gop_nn_biases_layer0,
      av1_use_flat_gop_nn_biases_layer1,
  },
};

#undef NUM_FEATURES
#undef NUM_HIDDEN_LAYERS
#undef NUM_HIDDEN_NODES_LAYER0
#undef NUM_LABELS

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_USE_FLAT_GOP_MODEL_PARAMS_H_
