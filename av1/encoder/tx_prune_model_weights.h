/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#ifndef AOM_AV1_ENCODER_TX_PRUNE_MODEL_WEIGHTS_H_
#define AOM_AV1_ENCODER_TX_PRUNE_MODEL_WEIGHTS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "av1/encoder/ml.h"

/***************************CONFIG_NN_V2 (New)********************************/
#if CONFIG_NN_V2
// Tx type model for 4x4 block.
static float av1_tx_type_nn_4x4_hor_layer0_weights[32] = {
  -1.64947f, -1.54497f, -1.62832f, -0.17774f, -2.89498f, -0.72498f, 0.72036f,
  0.17996f,  1.20000f,  -0.27654f, 0.77396f,  1.21684f,  -1.75909f, -0.51272f,
  -1.25923f, 0.35005f,  -0.04257f, -0.23389f, -0.41841f, -0.08229f, 0.09503f,
  2.73144f,  -0.16875f, -0.23482f, 0.02194f,  -0.26427f, 0.28049f,  0.21260f,
  1.35792f,  0.27733f,  0.88660f,  -0.68304f,
};

static float av1_tx_type_nn_4x4_hor_layer0_bias[8] = {
  1.38742f, 0.59540f,  -1.37622f, 1.92114f,
  0.00000f, -0.38998f, -0.32726f, -0.15650f,
};

static float av1_tx_type_nn_4x4_hor_layer1_weights[32] = {
  1.65254f,  1.00915f,  -0.89318f, -2.05142f, -0.23235f, 0.96781f,  -0.37145f,
  -0.21056f, 1.13891f,  0.38675f,  0.87739f,  -1.42697f, 0.48015f,  0.61883f,
  -0.03979f, 0.11487f,  0.48042f,  0.45200f,  -0.23242f, 0.75166f,  0.55458f,
  0.39452f,  -0.35285f, 1.59120f,  -1.49221f, -0.48349f, -0.64692f, 1.49297f,
  -0.26782f, -0.65416f, -0.10648f, 0.05568f,
};

static float av1_tx_type_nn_4x4_hor_layer1_bias[4] = {
  4.07177f,
  3.26961f,
  0.58083f,
  1.21199f,
};

static float av1_tx_type_nn_4x4_hor_layer0_out[8] = { 0 };
static float av1_tx_type_nn_4x4_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_4x4_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          4,                                      // num_inputs
          8,                                      // num_outputs
          av1_tx_type_nn_4x4_hor_layer0_weights,  // weights
          av1_tx_type_nn_4x4_hor_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_4x4_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          8,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_4x4_hor_layer1_weights,
          av1_tx_type_nn_4x4_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_4x4_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_4x4_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_4x4_ver_layer0_weights[32] = {
  -0.02032f, 2.61610f,  0.02098f,  -0.30217f, 0.12637f,  0.11017f,  -3.01996f,
  0.35144f,  1.93776f,  -0.20463f, 1.64102f,  -1.41986f, -3.66717f, -0.51655f,
  0.43910f,  0.37778f,  -1.02634f, 0.85337f,  -0.69753f, 1.00206f,  2.11784f,
  1.89427f,  1.92919f,  0.43201f,  -1.67358f, -1.67035f, -1.54623f, 0.16714f,
  -0.06589f, -0.28142f, -0.33118f, 1.72227f,
};

static float av1_tx_type_nn_4x4_ver_layer0_bias[8] = {
  -0.33685f, 0.22025f,  0.28140f, 0.56138f,
  0.93489f,  -1.77048f, 1.34989f, -0.93747f,
};

static float av1_tx_type_nn_4x4_ver_layer1_weights[32] = {
  -1.39506f, -1.06271f, -1.10886f, -1.69719f, 0.19699f,  -2.39850f, -1.26457f,
  0.75328f,  -1.26005f, -0.82738f, -0.12015f, -1.02702f, 1.40828f,  -2.37739f,
  -0.65639f, -0.71992f, -0.90453f, -1.12510f, -2.41362f, -1.16061f, -1.85577f,
  -0.99165f, -1.91366f, 0.16785f,  0.34776f,  0.58154f,  -0.18217f, -0.29257f,
  -0.86315f, -0.53336f, 0.30320f,  -1.32331f,
};

static float av1_tx_type_nn_4x4_ver_layer1_bias[4] = {
  -1.31519f,
  -3.26321f,
  1.71794f,
  -1.90778f,
};

static float av1_tx_type_nn_4x4_ver_layer0_out[8] = { 0 };
static float av1_tx_type_nn_4x4_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_4x4_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          4,                                      // num_inputs
          8,                                      // num_outputs
          av1_tx_type_nn_4x4_ver_layer0_weights,  // weights
          av1_tx_type_nn_4x4_ver_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_4x4_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          8,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_4x4_ver_layer1_weights,
          av1_tx_type_nn_4x4_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_4x4_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_4x4_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Tx type model for 4x8 block.
static float av1_tx_type_nn_4x8_hor_layer0_weights[32] = {
  0.00218f,  -0.41880f, -0.61215f, -0.92588f, 0.54291f,  -0.10898f, 0.70691f,
  0.46819f,  -1.61598f, -0.08834f, -0.96839f, 1.18489f,  -0.45171f, -0.65445f,
  -0.32179f, -0.10399f, 1.04379f,  0.91895f,  0.85589f,  0.08267f,  1.35388f,
  -2.03096f, 0.08168f,  -0.06372f, -0.26732f, -0.48262f, -0.08682f, 2.44071f,
  -1.35896f, -1.17121f, 1.68866f,  0.10357f,
};

static float av1_tx_type_nn_4x8_hor_layer0_bias[8] = {
  2.93391f,  0.66831f, -0.21419f, 0.00000f,
  -0.72878f, 0.15127f, -1.46755f, 0.16658f,
};

static float av1_tx_type_nn_4x8_hor_layer1_weights[32] = {
  -1.52077f, -1.06243f, 0.35319f,  -0.49207f, 0.54524f,  0.44271f, 1.37117f,
  -0.38957f, -1.28889f, -0.57133f, 0.04658f,  0.62278f,  0.37984f, 0.33247f,
  1.65547f,  -0.56806f, -1.38645f, -0.76258f, 0.67926f,  0.08783f, -0.01443f,
  0.34950f,  1.45812f,  -0.51332f, -1.41331f, -0.16453f, 0.05755f, 0.31405f,
  -0.50191f, 0.18219f,  1.83664f,  -0.75276f,
};

static float av1_tx_type_nn_4x8_hor_layer1_bias[4] = {
  -1.17455f,
  -2.26089f,
  -1.79863f,
  -2.26333f,
};

static float av1_tx_type_nn_4x8_hor_layer0_out[8] = { 0 };
static float av1_tx_type_nn_4x8_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_4x8_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          4,                                      // num_inputs
          8,                                      // num_outputs
          av1_tx_type_nn_4x8_hor_layer0_weights,  // weights
          av1_tx_type_nn_4x8_hor_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_4x8_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          8,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_4x8_hor_layer1_weights,
          av1_tx_type_nn_4x8_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_4x8_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_4x8_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_4x8_ver_layer0_weights[128] = {
  -0.00952f, -0.98858f, -0.93181f, 1.39594f,  0.96559f,  0.18162f,  -0.76064f,
  -0.06066f, 0.07907f,  -0.09365f, -0.21313f, -0.02187f, -2.61707f, -2.68702f,
  -0.10982f, 0.18559f,  1.17049f,  1.11387f,  1.12697f,  1.05804f,  1.12764f,
  1.06318f,  1.12052f,  0.17406f,  1.83157f,  0.19362f,  0.46910f,  0.39608f,
  0.33342f,  0.40083f,  0.27645f,  1.06864f,  -4.06645f, -0.38775f, -0.11070f,
  0.03781f,  -0.09141f, 0.06185f,  -0.04852f, 0.20163f,  0.16784f,  0.16641f,
  -0.50941f, -0.61087f, 2.07008f,  -0.82381f, -0.85558f, 0.05528f,  -0.10535f,
  -2.81150f, 0.67038f,  0.43643f,  0.49062f,  -0.04465f, 0.90438f,  0.00977f,
  0.46272f,  1.59751f,  0.95234f,  0.35086f,  0.85624f,  0.73149f,  1.67779f,
  -2.21511f, -1.24746f, -1.09014f, -0.92441f, -1.22591f, -1.06961f, -0.95897f,
  -1.24956f, 0.73797f,  1.23275f,  -0.60064f, -0.07851f, 0.14397f,  0.22110f,
  -0.04422f, 0.14350f,  0.75926f,  0.35032f,  0.48104f,  2.81408f,  0.34662f,
  0.42090f,  0.35521f,  -1.36804f, -0.14974f, -0.47696f, -0.07892f, 0.36910f,
  0.32299f,  0.23916f,  0.06032f,  -0.17844f, -0.17558f, -1.42746f, -0.55828f,
  -1.00418f, -0.64823f, -0.73654f, -0.85197f, -1.50989f, 1.69385f,  -0.04973f,
  -0.09273f, 1.04249f,  0.79235f,  1.13229f,  0.99617f,  0.03851f,  0.56334f,
  0.90795f,  1.08296f,  0.58519f,  1.74765f,  0.63971f,  1.35951f,  0.07803f,
  -0.05127f, 0.26514f,  -0.84629f, -0.66343f, -2.10630f, 0.11017f,  2.18528f,
  -0.21958f, 0.05970f,
};

static float av1_tx_type_nn_4x8_ver_layer0_bias[16] = {
  0.04205f, 0.22260f, -1.03870f, -1.19568f, 0.44283f,  0.01143f,
  0.00235f, 4.26772f, 0.44364f,  -0.33199f, -0.39076f, -0.35129f,
  0.08288f, 0.18195f, -0.79890f, 0.10047f,
};

static float av1_tx_type_nn_4x8_ver_layer1_weights[64] = {
  -0.38193f, -0.12095f, 1.57802f,  0.34932f,  -0.47333f, -0.12304f, -0.01736f,
  -2.52445f, 0.18983f,  -0.64707f, -0.60889f, -0.53750f, 0.91666f,  -0.62823f,
  -0.13377f, -0.43594f, -0.38618f, -0.01328f, 0.97457f,  1.48589f,  -1.03238f,
  -0.33459f, -0.35108f, -2.42417f, 0.60229f,  0.06824f,  -0.75495f, 0.26902f,
  0.65311f,  -0.23887f, -0.44604f, -0.55800f, -0.33842f, 0.04259f,  -0.59589f,
  0.49738f,  -0.62301f, -0.30896f, -0.29602f, -2.57052f, 2.00943f,  -0.66490f,
  -0.76312f, 0.28256f,  1.06311f,  -0.38364f, -0.63508f, -0.57609f, -0.88765f,
  -1.04403f, -0.46531f, 0.34084f,  -1.20498f, -0.68352f, -0.72251f, -2.63242f,
  -0.68736f, -0.37904f, -1.32371f, 0.47288f,  1.51904f,  0.78372f,  -1.01830f,
  -1.01848f,
};

static float av1_tx_type_nn_4x8_ver_layer1_bias[4] = {
  -1.45955f,
  -2.08949f,
  -1.24813f,
  -1.55368f,
};

static float av1_tx_type_nn_4x8_ver_layer0_out[16] = { 0 };
static float av1_tx_type_nn_4x8_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_4x8_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                      // num_inputs
          16,                                     // num_outputs
          av1_tx_type_nn_4x8_ver_layer0_weights,  // weights
          av1_tx_type_nn_4x8_ver_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_4x8_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_4x8_ver_layer1_weights,
          av1_tx_type_nn_4x8_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_4x8_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_4x8_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

/******************************************************************************/

// Tx type model for 8x4 block.
static float av1_tx_type_nn_8x4_hor_layer0_weights[128] = {
  -0.22492f, 0.13341f,  -4.03243f, -0.64015f, 0.02783f,  0.60466f,  -0.13335f,
  0.16828f,  0.12336f,  0.52904f,  1.18455f,  -0.32425f, 0.13052f,  0.93810f,
  -3.71165f, 0.02990f,  -4.63558f, 0.05666f,  0.03524f,  -0.07449f, -0.44006f,
  -0.33215f, -0.33713f, 0.08097f,  0.60873f,  0.29582f,  0.21696f,  -0.78729f,
  -0.16757f, -0.26567f, -0.00720f, -1.11226f, 1.58189f,  1.58463f,  1.48536f,
  1.54374f,  1.60069f,  1.46125f,  1.53932f,  0.05974f,  -1.82192f, 0.47043f,
  0.38090f,  0.20833f,  -0.05637f, 0.05183f,  0.01323f,  -0.25662f, 0.78634f,
  -0.55069f, -0.02975f, -1.29294f, -0.77192f, -2.34299f, -1.28074f, 0.77894f,
  -1.69740f, -1.66032f, -1.44323f, -1.55063f, -1.50845f, -1.23690f, -1.80663f,
  0.75079f,  2.32551f,  0.05878f,  0.80438f,  0.88584f,  0.69153f,  0.89060f,
  0.73660f,  0.87259f,  -0.00745f, -1.30044f, -0.59430f, 2.07270f,  1.03307f,
  -0.84697f, -1.19393f, 0.17549f,  -0.24978f, -3.67234f, 0.20781f,  -0.53946f,
  -0.05068f, 0.88274f,  1.30371f,  0.10288f,  0.07585f,  0.12259f,  -0.30815f,
  0.25437f,  -2.82096f, -2.69482f, 0.02370f,  0.12500f,  -0.21019f, -0.49220f,
  0.03638f,  -0.29795f, 0.28645f,  -0.48432f, -0.38584f, -0.32148f, -0.47197f,
  0.32437f,  0.32528f,  -0.19437f, 0.30383f,  -0.31879f, 0.26359f,  -0.12164f,
  -0.43647f, -0.08288f, -0.33438f, -0.63608f, -0.46647f, -0.46574f, 0.47806f,
  -0.49012f, -1.51234f, -1.13502f, -1.20470f, -1.02913f, -1.09182f, -0.93921f,
  -1.85523f, 0.92532f,
};

static float av1_tx_type_nn_8x4_hor_layer0_bias[16] = {
  0.36631f,  0.02901f,  0.64305f,  1.53074f, -1.40229f, 0.03852f,
  -0.05043f, 0.89632f,  -1.23312f, 0.07036f, 0.17070f,  0.56250f,
  -0.28958f, -0.32869f, -0.01704f, 0.68171f,
};

static float av1_tx_type_nn_8x4_hor_layer1_weights[64] = {
  -0.49441f, -0.31960f, -0.84946f, -0.85800f, -2.37767f, 0.81373f,  -0.73172f,
  -0.69337f, 0.88807f,  -0.49242f, -0.44717f, -0.11436f, 0.09978f,  0.15393f,
  0.17083f,  1.44850f,  -0.20582f, -0.04906f, 0.42990f,  -0.61939f, -1.09692f,
  -1.14885f, -1.36879f, -1.30828f, -0.59558f, -0.30903f, -0.08906f, 0.06953f,
  0.15383f,  -0.04193f, -0.54858f, 1.82676f,  -0.22411f, 0.05264f,  -0.45848f,
  -0.72985f, 0.87553f,  0.04116f,  -1.29774f, -2.63018f, 1.09089f,  -0.36048f,
  -0.16725f, 0.11627f,  0.49918f,  0.07539f,  0.00763f,  0.73706f,  0.87800f,
  0.57049f,  0.60969f,  1.02779f,  1.53339f,  -0.35915f, 0.06410f,  1.44582f,
  0.09698f,  0.71888f,  0.60594f,  0.84103f,  -0.50440f, -0.38825f, 0.15626f,
  -1.10654f,
};

static float av1_tx_type_nn_8x4_hor_layer1_bias[4] = {
  -0.92861f,
  -1.45151f,
  -1.33588f,
  -4.33853f,
};

static float av1_tx_type_nn_8x4_hor_layer0_out[16] = { 0 };
static float av1_tx_type_nn_8x4_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_8x4_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                      // num_inputs
          16,                                     // num_outputs
          av1_tx_type_nn_8x4_hor_layer0_weights,  // weights
          av1_tx_type_nn_8x4_hor_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_8x4_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_8x4_hor_layer1_weights,
          av1_tx_type_nn_8x4_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_8x4_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_8x4_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_8x4_ver_layer0_weights[32] = {
  -1.10946f, 1.86574f,  -1.59343f, 0.27018f, -1.70676f, -0.73982f, -0.19021f,
  -1.94208f, -2.29759f, -1.44402f, 0.28700f, -1.18340f, -1.50158f, -0.44175f,
  -1.36831f, 1.00374f,  2.59312f,  0.50291f, -0.71042f, -0.12238f, -0.15901f,
  -0.22807f, -0.67376f, -0.30215f, 0.54407f, -0.45538f, 1.18262f,  2.28687f,
  1.66212f,  1.70826f,  1.55182f,  0.12230f,
};

static float av1_tx_type_nn_8x4_ver_layer0_bias[8] = {
  0.10943f,  2.09789f, 2.16578f, 0.15766f,
  -0.42461f, 0.00000f, 1.22090f, -1.28717f,
};

static float av1_tx_type_nn_8x4_ver_layer1_weights[32] = {
  1.20426f,  -1.23237f, 2.41053f, -0.72488f, 1.25249f,  0.18018f,  -0.09586f,
  2.17901f,  0.15364f,  1.21535f, -0.38263f, -0.74309f, 0.50551f,  -0.54208f,
  0.59139f,  1.16095f,  0.55919f, -0.60183f, 1.18949f,  1.60787f,  0.54002f,
  -0.10712f, -0.16153f, 0.16207f, -0.32338f, 2.68712f,  -2.83483f, -0.27086f,
  -1.15005f, -0.39311f, 1.51236f, -1.68973f,
};

static float av1_tx_type_nn_8x4_ver_layer1_bias[4] = {
  1.81013f,
  1.10517f,
  2.90059f,
  0.95391f,
};

static float av1_tx_type_nn_8x4_ver_layer0_out[8] = { 0 };
static float av1_tx_type_nn_8x4_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_8x4_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          4,                                      // num_inputs
          8,                                      // num_outputs
          av1_tx_type_nn_8x4_ver_layer0_weights,  // weights
          av1_tx_type_nn_8x4_ver_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_8x4_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          8,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_8x4_ver_layer1_weights,
          av1_tx_type_nn_8x4_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_8x4_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_8x4_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Tx type model for 8x8 block.
static float av1_tx_type_nn_8x8_hor_layer0_weights[128] = {
  -0.85529f, 0.37619f,  0.12754f,  0.08622f,  0.45278f,  0.54929f,  1.60651f,
  -0.62654f, -0.54929f, -0.10131f, -0.17569f, 0.13948f,  0.31695f,  -0.05616f,
  0.20483f,  -0.36448f, 2.27203f,  -0.33087f, 0.47679f,  0.86888f,  0.39370f,
  0.46239f,  0.01113f,  1.50327f,  -1.48226f, -1.69621f, -1.49777f, -1.38885f,
  -1.37753f, -1.22681f, -1.70576f, 0.51329f,  -1.65662f, 1.74197f,  -0.13579f,
  -0.13133f, -0.58396f, -0.55510f, -1.10709f, -2.34975f, 0.22445f,  -0.56491f,
  -0.83432f, 0.13492f,  1.32147f,  2.85285f,  0.13819f,  0.03792f,  -1.30792f,
  0.04155f,  -0.70644f, -0.43430f, -0.16212f, -0.86945f, -1.16976f, 1.68339f,
  0.29540f,  0.01137f,  -0.25335f, -0.16856f, 0.12028f,  0.05207f,  0.39357f,
  -0.01545f, -0.21980f, -1.94091f, -1.01315f, -0.68270f, -0.40590f, -0.67111f,
  2.08283f,  0.19291f,  -4.81426f, -0.65044f, -0.24598f, 0.06371f,  -0.10272f,
  -0.14502f, -0.06821f, 0.45202f,  0.21091f,  -0.80864f, 0.39255f,  1.79189f,
  1.80453f,  1.10484f,  1.17608f,  0.96901f,  -0.35871f, -0.94311f, 0.63147f,
  2.95157f,  0.45917f,  -0.42849f, -0.55643f, -0.06097f, 3.49299f,  -0.50972f,
  0.11075f,  -0.08405f, -0.09274f, -0.22694f, -0.42426f, 0.48632f,  -1.61074f,
  1.82998f,  0.37623f,  -1.20330f, -0.01142f, -1.33307f, -0.27492f, -2.23621f,
  1.38846f,  1.42085f,  1.42568f,  1.36152f,  1.46910f,  1.27473f,  1.34752f,
  0.12753f,  -1.08197f, -1.08280f, -0.79489f, -1.12338f, -1.06795f, -0.87857f,
  -0.99892f, 1.09823f,
};

static float av1_tx_type_nn_8x8_hor_layer0_bias[16] = {
  -0.49232f, -0.29685f, -1.44020f, 1.10940f,  1.16452f, -0.34862f,
  -0.38761f, -0.36243f, 0.21776f,  0.28234f,  2.34269f, -0.04104f,
  -0.26319f, 2.65579f,  -1.30137f, -0.01487f,
};

static float av1_tx_type_nn_8x8_hor_layer1_weights[64] = {
  -0.38058f, -0.41295f, -1.26884f, -0.75560f, -1.57450f, 0.56072f,  -1.42322f,
  -0.29106f, 0.07228f,  0.04391f,  1.61388f,  -0.03055f, 0.81637f,  2.06045f,
  0.27119f,  -0.48328f, -0.45528f, -0.60534f, -1.61209f, -0.78157f, -1.65034f,
  0.60958f,  -1.30523f, 0.25143f,  0.11398f,  0.37860f,  1.54829f,  0.02309f,
  0.67288f,  2.11447f,  0.44845f,  -0.70406f, -0.67897f, -0.38759f, -1.30383f,
  -1.22646f, -1.54571f, 0.60552f,  -1.52565f, 0.11469f,  0.17344f,  0.08622f,
  1.57906f,  -0.00909f, 0.81634f,  2.04909f,  1.26466f,  -1.45741f, -0.75229f,
  0.06200f,  -1.05835f, -0.66257f, -1.73766f, 0.99923f,  -1.87082f, 0.14580f,
  0.49525f,  0.46839f,  1.32203f,  0.33923f,  0.97001f,  2.38584f,  1.58811f,
  0.06161f,
};

static float av1_tx_type_nn_8x8_hor_layer1_bias[4] = {
  1.70385f,
  1.82373f,
  1.78496f,
  1.80826f,
};

static float av1_tx_type_nn_8x8_hor_layer0_out[16] = { 0 };
static float av1_tx_type_nn_8x8_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_8x8_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                      // num_inputs
          16,                                     // num_outputs
          av1_tx_type_nn_8x8_hor_layer0_weights,  // weights
          av1_tx_type_nn_8x8_hor_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_8x8_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_8x8_hor_layer1_weights,
          av1_tx_type_nn_8x8_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_8x8_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_8x8_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_8x8_ver_layer0_weights[128] = {
  -0.67016f, -1.72366f, -1.86576f, -1.50962f, -1.70419f, -1.73964f, -1.84615f,
  2.09681f,  -0.05081f, -0.61030f, 2.02541f,  0.60222f,  0.99936f,  2.02114f,
  -0.53893f, -0.23757f, 0.73566f,  0.25443f,  0.00132f,  -0.74036f, -0.75351f,
  -0.76964f, -1.71007f, -0.15770f, 1.60982f,  2.17638f,  0.90681f,  0.64973f,
  0.85914f,  0.58786f,  -1.46228f, 0.05187f,  1.18804f,  0.30850f,  0.29512f,
  0.40526f,  0.37635f,  0.32311f,  0.37471f,  1.12346f,  3.41856f,  -0.36653f,
  0.42537f,  -0.19240f, 0.00155f,  0.30826f,  -0.02116f, -0.53435f, -0.34829f,
  -0.52466f, -0.11521f, -0.29163f, -2.05689f, -2.87372f, -0.62626f, 0.09585f,
  -0.75257f, 0.10057f,  1.43474f,  0.89450f,  0.75900f,  1.11147f,  1.00558f,
  0.25886f,  2.22095f,  -0.17926f, 0.57161f,  0.39546f,  0.47846f,  0.40452f,
  0.54298f,  0.45814f,  -3.62788f, -3.02374f, 0.03716f,  -0.13937f, -0.09415f,
  -0.12463f, 0.05682f,  0.03672f,  1.20746f,  1.25003f,  1.27071f,  1.31883f,
  1.27473f,  1.34943f,  1.23158f,  0.09039f,  0.19388f,  0.63420f,  2.79612f,
  0.93803f,  -0.11323f, -0.02027f, 0.41286f,  -0.05979f, -3.80705f, -0.52451f,
  -0.77098f, -0.68132f, -0.65559f, -0.60975f, -1.26165f, 0.25582f,  0.05346f,
  0.61403f,  0.32140f,  -2.39831f, -1.42355f, 1.30541f,  1.02361f,  0.12930f,
  -1.61469f, -0.77036f, -0.59144f, 1.27769f,  1.52068f,  0.82137f,  1.83159f,
  -0.66626f, -0.69806f, -1.00564f, -0.85995f, -0.90889f, -0.84412f, -0.85712f,
  -1.29848f, 0.39308f,
};

static float av1_tx_type_nn_8x8_ver_layer0_bias[16] = {
  -0.14868f, -0.48343f, 3.94416f,  -0.78037f, -1.33789f, -0.60611f,
  0.51793f,  0.44030f,  -0.71563f, 0.22561f,  -1.19083f, -0.46149f,
  0.83015f,  0.06024f,  1.17180f,  0.65122f,
};

static float av1_tx_type_nn_8x8_ver_layer1_weights[64] = {
  -1.42711f, -0.21683f, 2.12061f,  0.20489f,  -0.50228f, -0.24770f, 0.23391f,
  1.03470f,  -0.44847f, -0.63225f, -0.21583f, -0.06467f, -0.21892f, -0.07786f,
  1.43322f,  0.00280f,  -1.53057f, -0.18912f, 1.95333f,  0.31151f,  -2.07601f,
  0.06776f,  0.25529f,  0.94800f,  -1.11453f, -0.20594f, -0.13281f, 0.01485f,
  0.17650f,  -0.07955f, 1.43734f,  -0.23193f, -2.06463f, -0.21238f, 2.13707f,
  0.30351f,  0.27594f,  -0.36245f, 0.19539f,  0.91045f,  -0.24068f, -0.37616f,
  0.88792f,  0.02947f,  -0.16903f, -0.04932f, 1.51293f,  -0.95967f, -1.62903f,
  0.05326f,  2.30703f,  0.64445f,  -1.09464f, -0.16623f, 1.00240f,  0.07548f,
  -0.50406f, 0.63854f,  1.02340f,  0.49833f,  0.13671f,  0.26722f,  2.09516f,
  -0.41305f,
};

static float av1_tx_type_nn_8x8_ver_layer1_bias[4] = {
  2.14067f,
  2.76699f,
  2.04233f,
  1.34803f,
};

static float av1_tx_type_nn_8x8_ver_layer0_out[16] = { 0 };
static float av1_tx_type_nn_8x8_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_8x8_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                      // num_inputs
          16,                                     // num_outputs
          av1_tx_type_nn_8x8_ver_layer0_weights,  // weights
          av1_tx_type_nn_8x8_ver_layer0_bias,     // bias
          RELU,                                   // activation
          av1_tx_type_nn_8x8_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_8x8_ver_layer1_weights,
          av1_tx_type_nn_8x8_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_8x8_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                  // num_outputs
  av1_tx_type_nn_8x8_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Tx type model for 8x16 block.
static float av1_tx_type_nn_8x16_hor_layer0_weights[128] = {
  -1.61872f, -1.58520f, -1.41236f, -1.53255f, -1.59794f, -1.25769f, -1.90043f,
  0.73431f,  1.10135f,  0.47054f,  0.43230f,  -0.43009f, -0.09135f, -0.07289f,
  -0.38785f, 1.23775f,  -0.35312f, 0.73789f,  0.88864f,  0.75957f,  0.62579f,
  0.46974f,  0.21851f,  1.63821f,  -2.27289f, -0.68522f, -0.69814f, -0.84368f,
  -0.91320f, -0.63055f, -1.03296f, 0.55778f,  -0.00071f, 1.27539f,  1.60068f,
  1.40975f,  0.97372f,  0.92843f,  1.90853f,  0.12626f,  1.71953f,  1.41978f,
  -0.12234f, -1.27058f, 0.76207f,  0.02495f,  -0.67038f, -0.05255f, 1.72923f,
  1.47630f,  1.47058f,  1.47614f,  1.49354f,  1.66131f,  1.50801f,  0.17145f,
  -2.30947f, -2.10850f, -1.25636f, -0.24900f, 0.72602f,  1.26572f,  0.97865f,
  -0.65466f, 1.31129f,  0.26916f,  0.12139f,  -0.12761f, -0.39143f, -0.28134f,
  0.06584f,  2.24418f,  0.22516f,  0.05011f,  -0.01671f, -0.29476f, -0.40326f,
  0.21138f,  -0.11573f, -0.31154f, -0.36828f, 0.03694f,  -0.07172f, -0.63419f,
  -3.14351f, -1.23125f, 0.65311f,  -0.11406f, 1.97287f,  -0.10422f, 0.83896f,
  0.85033f,  0.49724f,  0.80482f,  0.51454f,  1.06447f,  0.76693f,  0.72599f,
  -0.78573f, -0.53950f, 0.40894f,  0.00086f,  0.10784f,  -0.70498f, 1.16395f,
  1.14597f,  1.13496f,  1.12177f,  1.02100f,  -1.37574f, -2.97144f, 0.33899f,
  0.42013f,  0.86327f,  2.31983f,  2.04008f,  0.95503f,  0.15081f,  0.11530f,
  -0.02574f, -4.77119f, 0.13257f,  -0.01704f, -0.23087f, -0.00825f, 0.07029f,
  -0.28136f, 0.42556f,
};

static float av1_tx_type_nn_8x16_hor_layer0_bias[16] = {
  0.93617f,  -0.24000f, -1.26821f, 0.78780f,  0.13690f, -0.21948f,
  -1.45162f, 0.44584f,  -1.92582f, -0.23169f, 0.56004f, -1.19937f,
  1.81560f,  -1.02643f, -0.81690f, 0.08302f,
};

static float av1_tx_type_nn_8x16_hor_layer1_weights[64] = {
  0.06696f,  -0.11538f, -1.42029f, 0.32965f,  0.81046f,  0.01146f,  1.20945f,
  -0.16899f, 0.53224f,  -0.40232f, 0.01786f,  -0.73242f, 1.29750f,  1.95185f,
  0.70143f,  1.43287f,  0.76220f,  0.79937f,  -1.79011f, -1.15178f, 0.42526f,
  -0.67519f, 0.77267f,  -0.30697f, 2.46004f,  -0.49828f, 0.02875f,  1.09972f,
  1.47662f,  0.61719f,  0.61417f,  -0.12363f, 2.53048f,  0.00418f,  -1.38964f,
  0.88117f,  0.39239f,  -0.19347f, -2.58600f, -0.33715f, 1.09323f,  -0.32127f,
  0.02456f,  -0.19125f, 1.12728f,  0.66502f,  0.34296f,  1.14897f,  0.29967f,
  1.19209f,  0.22108f,  -0.11975f, 1.49776f,  -1.34624f, -2.58478f, -1.34632f,
  1.53207f,  0.45634f,  -1.48476f, 0.17489f,  0.71790f,  -2.12086f, -1.21778f,
  -1.31243f,
};

static float av1_tx_type_nn_8x16_hor_layer1_bias[4] = {
  0.83359f,
  1.06875f,
  1.77645f,
  1.49570f,
};

static float av1_tx_type_nn_8x16_hor_layer0_out[16] = { 0 };
static float av1_tx_type_nn_8x16_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_8x16_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                       // num_inputs
          16,                                      // num_outputs
          av1_tx_type_nn_8x16_hor_layer0_weights,  // weights
          av1_tx_type_nn_8x16_hor_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_8x16_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_8x16_hor_layer1_weights,
          av1_tx_type_nn_8x16_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_8x16_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_8x16_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_8x16_ver_layer0_weights[128] = {
  0.32858f,  -1.28887f, 0.25632f,  -0.05262f, 2.69203f,  -0.07004f, 1.37337f,
  -0.05725f, -0.05659f, 0.05592f,  0.01039f,  -0.29343f, 1.58628f,  -0.30003f,
  -3.43118f, 0.00272f,  1.70928f,  -0.76348f, 0.05889f,  -0.03263f, -0.07724f,
  0.03523f,  -0.19890f, 1.18005f,  -0.03605f, -0.20530f, -4.00733f, 0.10210f,
  -0.05368f, -0.17650f, -0.15317f, 0.06499f,  0.56705f,  1.04341f,  0.62890f,
  0.73451f,  -0.22199f, 0.86659f,  0.78443f,  -0.61664f, -0.50606f, 0.30247f,
  0.14455f,  0.39276f,  0.49203f,  0.65019f,  0.12269f,  1.64080f,  1.68289f,
  1.42694f,  1.60825f,  1.58501f,  1.47252f,  1.62589f,  1.48218f,  0.17726f,
  -0.04884f, 0.35376f,  -0.04796f, 0.32589f,  0.35087f,  0.35258f,  -0.46103f,
  -0.31176f, -0.05203f, 0.07247f,  -0.26756f, 0.22019f,  0.03412f,  0.33773f,
  0.29811f,  -0.11140f, 0.12831f,  -0.44673f, -0.09858f, 0.07889f,  0.15137f,
  0.00347f,  -0.23394f, 0.08886f,  -0.31201f, -0.79912f, -0.51092f, 0.14123f,
  -1.09599f, -4.26020f, -0.68675f, -0.02842f, -1.54538f, -1.28977f, -1.30558f,
  -1.21074f, -1.37142f, -1.14743f, -1.85397f, 0.82985f,  -0.30681f, 0.04494f,
  -0.24023f, -4.18053f, -0.16096f, -0.55492f, -0.27882f, 0.05829f,  -0.41224f,
  -2.52088f, -0.56162f, -1.04547f, -1.70685f, -0.28842f, -1.43673f, -0.01468f,
  -3.20585f, -0.69120f, -0.43931f, -0.46270f, -0.65885f, -0.55884f, -0.75138f,
  0.36381f,  -5.70858f, -0.14548f, -0.15745f, -0.11812f, -0.07605f, -0.07693f,
  -0.12236f, 0.16075f,
};

static float av1_tx_type_nn_8x16_ver_layer0_bias[16] = {
  -0.35385f, 0.30491f,  -0.90011f, 0.42941f,  1.20928f, -0.88331f,
  -1.48818f, -0.34785f, -0.32668f, -0.22695f, 0.89188f, 0.65521f,
  0.57598f,  0.99819f,  0.75175f,  0.17044f,
};

static float av1_tx_type_nn_8x16_ver_layer1_weights[64] = {
  -0.62913f, -0.34304f, 0.42963f,  -0.17440f, -1.44092f, 0.69142f,  -1.36067f,
  0.52211f,  0.44658f,  -0.26501f, -0.41657f, 0.34428f,  -0.34390f, -0.58567f,
  -0.84097f, -1.96311f, -0.37215f, -0.22250f, -1.23811f, -0.07247f, -0.81731f,
  0.58755f,  -1.30559f, 0.39551f,  0.41743f,  -0.09940f, -0.33230f, 0.14458f,
  -0.25139f, -0.54517f, 0.13469f,  -0.38157f, -0.39109f, -0.18205f, 0.06834f,
  -0.08395f, -0.92187f, 0.56724f,  1.44381f,  0.53226f,  -0.22356f, 0.12285f,
  -0.29418f, -1.86749f, -0.22372f, -0.60204f, -0.87746f, -1.16936f, 0.56884f,
  0.62641f,  -0.11823f, 1.00395f,  1.64794f,  -0.64535f, 2.29322f,  -0.23397f,
  0.17251f,  -0.35927f, 0.65631f,  -0.26812f, 0.80128f,  0.85748f,  0.47404f,
  2.20547f,
};

static float av1_tx_type_nn_8x16_ver_layer1_bias[4] = {
  -0.44080f,
  -1.67455f,
  -1.46332f,
  -6.13206f,
};

static float av1_tx_type_nn_8x16_ver_layer0_out[16] = { 0 };
static float av1_tx_type_nn_8x16_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_8x16_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                       // num_inputs
          16,                                      // num_outputs
          av1_tx_type_nn_8x16_ver_layer0_weights,  // weights
          av1_tx_type_nn_8x16_ver_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_8x16_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_8x16_ver_layer1_weights,
          av1_tx_type_nn_8x16_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_8x16_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_8x16_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Tx type model for 16x8 block.
static float av1_tx_type_nn_16x8_hor_layer0_weights[128] = {
  0.02600f,  0.09786f,  -1.05107f, -0.35594f, -0.15658f, 2.99828f,  -0.07106f,
  -0.10101f, -0.14412f, -0.83790f, -0.19434f, 2.28368f,  1.91727f,  -0.00956f,
  -0.90640f, 0.09174f,  1.58895f,  1.38945f,  1.49431f,  1.51381f,  1.44803f,
  1.53544f,  1.44694f,  0.17753f,  1.69735f,  -0.78652f, 0.31092f,  -0.23736f,
  0.02231f,  -0.09884f, -0.00493f, 1.21189f,  -1.94382f, -0.34629f, -0.58309f,
  0.72291f,  -0.30056f, 0.90660f,  -0.57495f, 3.07809f,  0.73644f,  1.43050f,
  1.34356f,  -0.66554f, 0.50102f,  -0.64305f, 0.42044f,  -1.66165f, -0.05733f,
  -2.51402f, -1.01067f, -0.33390f, -0.32986f, -0.92431f, 1.86281f,  -0.07290f,
  -0.26290f, -0.68941f, 1.81156f,  0.66125f,  -2.09974f, 0.17032f,  -0.67461f,
  -0.00876f, -1.50154f, 1.17153f,  1.00377f,  0.33022f,  0.74689f,  0.42878f,
  0.61725f,  -0.83967f, 0.09467f,  -0.39892f, 0.33863f,  0.10656f,  -0.09249f,
  -0.39757f, 0.48481f,  -0.35162f, 1.47014f,  1.67827f,  -1.84051f, 0.16291f,
  -0.50135f, -2.29911f, -0.42217f, -0.13358f, 1.45899f,  -0.14743f, -0.02763f,
  -0.28003f, -0.01364f, 0.21014f,  -0.29026f, -0.20198f, 1.38782f,  0.56731f,
  0.27489f,  0.43227f,  0.41326f,  0.42721f,  0.87720f,  -1.90067f, -5.04951f,
  -0.17638f, -0.58119f, -0.08954f, -0.13692f, -0.12325f, -0.38548f, 0.66462f,
  -1.42377f, -1.21917f, -1.38193f, -1.36539f, -1.39378f, -1.19629f, -1.59812f,
  0.28689f,  0.32394f,  0.52128f,  0.01013f,  -0.28948f, -0.26293f, -0.44331f,
  -0.36570f, -0.50757f,
};

static float av1_tx_type_nn_16x8_hor_layer0_bias[16] = {
  -0.08696f, -0.22110f, -1.43604f, -1.00451f, -1.51029f, 0.63736f,
  0.45260f,  0.16229f,  4.01393f,  -0.21748f, 0.36411f,  -0.08764f,
  -0.12329f, 0.08986f,  1.08117f,  -0.00220f,
};

static float av1_tx_type_nn_16x8_hor_layer1_weights[64] = {
  0.55824f,  -0.14648f, 0.81947f,  -0.45867f, -1.86078f, -0.17291f, 0.34849f,
  0.15153f,  1.75625f,  -0.25760f, 0.72015f,  -0.30059f, -0.57975f, 0.07609f,
  -0.02036f, 0.07912f,  0.57080f,  -0.13792f, 0.74184f,  -0.87669f, -1.87572f,
  -0.27270f, 0.39751f,  0.19652f,  2.03514f,  -0.32944f, 0.76251f,  0.04399f,
  -0.63175f, 0.37420f,  0.08309f,  0.04466f,  0.60255f,  -0.12820f, 1.66065f,
  -0.59496f, -1.94794f, -0.14847f, 0.39424f,  0.16273f,  1.80587f,  0.41197f,
  0.74691f,  -0.21217f, -0.63173f, 0.09510f,  -0.35538f, -0.04407f, 0.92847f,
  0.20141f,  1.68680f,  -0.56528f, -2.26960f, 0.12978f,  0.73748f,  0.42438f,
  2.00673f,  -0.40189f, 0.95423f,  0.23234f,  -0.80953f, 0.65814f,  0.49444f,
  -0.23347f,
};

static float av1_tx_type_nn_16x8_hor_layer1_bias[4] = {
  3.57175f,
  2.42612f,
  3.31259f,
  2.08287f,
};

static float av1_tx_type_nn_16x8_hor_layer0_out[16] = { 0 };
static float av1_tx_type_nn_16x8_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_16x8_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                       // num_inputs
          16,                                      // num_outputs
          av1_tx_type_nn_16x8_hor_layer0_weights,  // weights
          av1_tx_type_nn_16x8_hor_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_16x8_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_16x8_hor_layer1_weights,
          av1_tx_type_nn_16x8_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_16x8_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_16x8_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_16x8_ver_layer0_weights[128] = {
  0.46633f,  1.55328f,  -0.11230f, -0.29571f, 0.18814f,  -1.52430f, -2.34660f,
  0.08644f,  -1.97718f, -1.29140f, -1.12262f, -1.12985f, -1.25911f, -0.96506f,
  -1.57129f, 0.96021f,  1.34192f,  1.28623f,  1.21655f,  1.28758f,  1.25482f,
  1.30195f,  1.19190f,  0.09310f,  0.52072f,  0.91487f,  1.24100f,  1.61236f,
  1.72166f,  2.20750f,  1.62379f,  -1.43936f, 0.50665f,  0.40213f,  0.66502f,
  -1.66699f, -3.07618f, 0.05877f,  0.60987f,  -0.09995f, -0.10916f, 0.48049f,
  0.23812f,  0.39847f,  -0.21682f, -0.63455f, 0.33453f,  -0.67939f, -4.14355f,
  -0.62756f, -0.22502f, -0.17215f, 0.01062f,  0.27049f,  -0.10748f, 0.30945f,
  2.72445f,  -0.89181f, -0.06800f, 0.20595f,  -0.73385f, 0.04071f,  -1.30294f,
  1.83507f,  0.92570f,  0.69609f,  0.76285f,  0.69892f,  0.76409f,  0.63104f,
  0.73397f,  1.09575f,  -0.20129f, -0.24022f, -0.24599f, -0.59107f, -0.88755f,
  -0.68987f, -0.75495f, -1.31002f, -1.30237f, -0.94093f, -2.15678f, -1.49303f,
  -1.17498f, -1.39952f, -0.91270f, -0.05587f, 1.02381f,  -0.75580f, -0.65263f,
  -0.78996f, -0.71075f, -0.71018f, -0.70350f, -1.26196f, 2.34208f,  -0.53611f,
  0.19752f,  -0.16842f, -0.24828f, 0.21857f,  0.08222f,  -2.55894f, -1.75702f,
  0.11394f,  1.03083f,  0.79972f,  -1.54112f, -1.82341f, -0.57597f, -0.02077f,
  -0.39616f, -0.00995f, -0.12809f, 0.01188f,  -0.25117f, 0.09202f,  0.09336f,
  -0.05614f, -0.30039f, 0.25834f,  1.19944f,  1.22533f,  0.92330f,  0.75967f,
  -0.81945f, -0.41647f,
};

static float av1_tx_type_nn_16x8_ver_layer0_bias[16] = {
  0.17841f,  0.67315f,  -1.24450f, 3.13859f,  0.16203f, -0.14992f,
  0.29553f,  -1.15567f, -0.71421f, 1.15977f,  1.14585f, 3.02460f,
  -0.04510f, 0.48000f,  -0.09354f, -0.42422f,
};

static float av1_tx_type_nn_16x8_ver_layer1_weights[64] = {
  0.29912f,  -0.10009f, -1.11478f, 1.76812f,  -0.27719f, 0.52148f,  0.17622f,
  -1.17116f, 0.73397f,  -0.69279f, -0.11080f, 1.53751f,  -1.42003f, 0.14731f,
  0.13592f,  -0.04883f, 0.39186f,  -0.13655f, -0.43994f, 1.82759f,  -0.25601f,
  -0.15018f, 0.51920f,  -1.56070f, 0.31683f,  -0.79367f, -0.02904f, 1.28637f,
  -1.15203f, 0.26627f,  0.42828f,  -0.24258f, 0.38647f,  -0.83352f, 0.32553f,
  2.09522f,  -0.26822f, -0.42191f, 0.32825f,  -1.30748f, 1.50551f,  -0.52669f,
  0.20045f,  1.69318f,  -1.47839f, 0.30802f,  -0.07290f, -0.28106f, 0.68192f,
  -0.15522f, 1.12579f,  2.21921f,  0.09720f,  -0.50265f, 0.83165f,  -1.31721f,
  0.72422f,  -1.24952f, 0.61653f,  2.04117f,  -1.42406f, 0.52568f,  -0.46180f,
  -0.00873f,
};

static float av1_tx_type_nn_16x8_ver_layer1_bias[4] = {
  3.34981f,
  3.74710f,
  1.38339f,
  0.45176f,
};

static float av1_tx_type_nn_16x8_ver_layer0_out[16] = { 0 };
static float av1_tx_type_nn_16x8_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_16x8_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                       // num_inputs
          16,                                      // num_outputs
          av1_tx_type_nn_16x8_ver_layer0_weights,  // weights
          av1_tx_type_nn_16x8_ver_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_16x8_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_16x8_ver_layer1_weights,
          av1_tx_type_nn_16x8_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_16x8_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_16x8_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Tx type model for 16x16 block.
static float av1_tx_type_nn_16x16_layer0_weights[128] = {
  1.26592f,  1.36313f,  1.30956f,  1.29926f,  1.48816f,  1.68851f,  1.32000f,
  0.13321f,  -0.22477f, -0.88906f, -0.19622f, 1.69605f,  1.22180f,  -1.57771f,
  -1.15765f, 0.05710f,  -1.13355f, -0.85486f, -0.99971f, -0.91571f, -1.06031f,
  -0.77952f, -1.15723f, 1.17809f,  1.35602f,  -0.05243f, -0.37596f, 0.26108f,
  0.17611f,  -0.10323f, 0.77279f,  -0.48911f, -0.79308f, 0.55112f,  0.43918f,
  0.27872f,  0.28714f,  0.45830f,  1.05689f,  0.03705f,  -2.49975f, -0.01940f,
  0.05709f,  0.07942f,  -0.13290f, -0.10359f, 0.00143f,  0.37303f,  0.96470f,
  0.53293f,  1.14459f,  0.89185f,  0.43378f,  0.47764f,  0.90924f,  0.15279f,
  -0.15361f, 0.02949f,  0.42240f,  0.68143f,  0.89588f,  0.73754f,  0.10974f,
  1.57755f,  -0.39870f, -0.32914f, 0.35638f,  0.34991f,  -0.00003f, -0.23373f,
  0.29630f,  -0.76699f, -0.01356f, 0.04234f,  0.84253f,  1.92078f,  0.93160f,
  0.71993f,  0.71604f,  0.76455f,  -1.59782f, 0.32332f,  1.11628f,  0.33062f,
  -0.03728f, -0.05710f, 0.80447f,  -0.14719f, 1.34658f,  -0.05718f, 0.64015f,
  0.21926f,  0.41653f,  0.12720f,  0.54092f,  1.39411f,  1.81819f,  -0.24513f,
  0.00955f,  0.38011f,  -0.57787f, -0.41759f, 0.68834f,  -0.31783f, -0.40607f,
  -0.10107f, -0.79374f, 0.75599f,  -0.16282f, -0.14490f, -0.20783f, -0.55019f,
  -0.13793f, -0.22293f, 0.18305f,  0.12445f,  0.56830f,  0.24567f,  0.09278f,
  0.70803f,  0.35803f,  -1.52676f, -0.89624f, 0.77665f,  0.19877f,  0.77175f,
  0.50355f,  0.08592f,
};

static float av1_tx_type_nn_16x16_layer0_bias[16] = {
  -1.31834f, 0.14346f,  -0.10062f, 0.84489f,  0.95617f,  -0.06720f,
  -0.68502f, -0.91442f, -0.31932f, 0.25276f,  -0.15138f, -1.57661f,
  -0.14062f, -0.42120f, 0.94573f,  -0.09287f,
};

static float av1_tx_type_nn_16x16_layer1_weights[64] = {
  -1.80333f, -1.06353f, 0.55139f,  0.74644f,  0.13747f, -0.93018f, -0.10286f,
  0.67133f,  0.24460f,  1.44583f,  0.02173f,  0.26037f, -0.73687f, 0.19566f,
  0.61846f,  -0.58601f, -1.03196f, -0.74415f, 0.30041f, -0.41967f, 1.08740f,
  0.96224f,  -0.59139f, 0.03813f,  0.05403f,  1.33427f, -0.54375f, -1.92181f,
  0.54704f,  0.13608f,  0.22151f,  -0.38076f, 1.18390f, -0.77508f, -1.84283f,
  1.00894f,  0.62318f,  -0.15296f, 1.27600f,  0.22822f, 0.12751f,  0.93910f,
  -0.28502f, 0.53912f,  -0.96889f, 0.10182f,  0.81508f, -0.43028f, 2.67386f,
  0.52204f,  0.49820f,  -0.41711f, 1.05038f,  1.12192f, 0.74349f,  -0.75417f,
  -0.03718f, -0.35769f, 0.89651f,  0.63236f,  0.54215f, -0.07894f, 0.48274f,
  1.08829f,
};

static float av1_tx_type_nn_16x16_layer1_bias[4] = {
  0.81986f,
  1.26865f,
  0.11118f,
  2.48404f,
};

static float av1_tx_type_nn_16x16_layer0_out[16] = { 0 };
static float av1_tx_type_nn_16x16_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_16x16 = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                    // num_inputs
          16,                                   // num_outputs
          av1_tx_type_nn_16x16_layer0_weights,  // weights
          av1_tx_type_nn_16x16_layer0_bias,     // bias
          RELU,                                 // activation
          av1_tx_type_nn_16x16_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_16x16_layer1_weights,
          av1_tx_type_nn_16x16_layer1_bias,
          NONE,
          av1_tx_type_nn_16x16_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                // num_outputs
  av1_tx_type_nn_16x16_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Tx type model for 4x16 block.
static float av1_tx_type_nn_4x16_hor_layer0_weights[32] = {
  0.36539f,  0.25667f,  0.01491f,  -0.21959f, 2.55105f,  0.17615f, 1.79884f,
  1.65936f,  -0.44363f, 0.00706f,  -0.68004f, -0.64360f, 1.75760f, 1.91906f,
  1.47682f,  0.09650f,  -3.59244f, -0.35004f, 0.93295f,  0.25806f, -0.08154f,
  0.79332f,  0.79535f,  1.09467f,  1.57855f,  -0.51359f, 0.90553f, -1.67744f,
  -1.74563f, -0.88830f, -1.77603f, 2.15935f,
};

static float av1_tx_type_nn_4x16_hor_layer0_bias[8] = {
  -0.36435f, -2.22731f, -0.00837f, -1.34546f,
  0.62806f,  -0.20675f, 4.91940f,  -0.56079f,
};

static float av1_tx_type_nn_4x16_hor_layer1_weights[32] = {
  -0.57191f, -1.46418f, 0.67331f,  -1.15027f, 0.46288f,  0.81251f,  2.51768f,
  -0.27147f, 0.00761f,  -2.15214f, -0.69650f, -0.50808f, 0.92832f,  0.45668f,
  2.34201f,  -0.52941f, 0.51008f,  -1.55496f, -0.01371f, -0.12356f, 0.66624f,
  0.88043f,  2.64862f,  -1.28024f, -0.17578f, -1.80034f, -0.32217f, 0.89519f,
  1.28413f,  -0.30326f, 2.45329f,  -0.83335f,
};

static float av1_tx_type_nn_4x16_hor_layer1_bias[4] = {
  2.33198f,
  3.36245f,
  1.62603f,
  2.91056f,
};

static float av1_tx_type_nn_4x16_hor_layer0_out[8] = { 0 };
static float av1_tx_type_nn_4x16_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_4x16_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          4,                                       // num_inputs
          8,                                       // num_outputs
          av1_tx_type_nn_4x16_hor_layer0_weights,  // weights
          av1_tx_type_nn_4x16_hor_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_4x16_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          8,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_4x16_hor_layer1_weights,
          av1_tx_type_nn_4x16_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_4x16_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_4x16_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_4x16_ver_layer0_weights[128] = {
  1.61392f,  1.41239f,  1.47646f,  1.47325f,  1.46110f,  1.49208f,  1.49414f,
  0.12835f,  -0.76986f, 0.07087f,  -0.24572f, -0.93168f, 3.07935f,  -0.18183f,
  -0.09831f, -0.07703f, -0.03222f, -0.25473f, -0.06090f, 2.93713f,  -0.38711f,
  -0.12884f, -0.18329f, -0.06262f, -0.00327f, -0.02930f, -0.01641f, -0.00622f,
  -0.03305f, -4.07069f, -2.76643f, 0.04413f,  -1.03176f, -0.19217f, -0.44980f,
  -2.48615f, -2.58112f, -0.87695f, 0.16187f,  -0.04891f, -0.06854f, 1.08104f,
  0.75245f,  1.49302f,  0.63363f,  1.45715f,  0.92574f,  1.72029f,  0.33326f,
  3.86646f,  0.04422f,  0.41019f,  0.36212f,  0.56600f,  -1.01552f, 0.05128f,
  0.40454f,  -1.05100f, -0.47461f, -1.33168f, -0.46145f, -1.36870f, -0.88838f,
  -1.05358f, -0.18537f, -0.34357f, -0.03698f, 0.68905f,  0.41010f,  0.31223f,
  -0.43382f, -0.74715f, 2.03366f,  -0.30419f, 0.45747f,  0.09526f,  0.31678f,
  0.22915f,  0.21832f,  1.26385f,  -0.06814f, -0.71417f, -1.18947f, 0.03762f,
  0.10936f,  2.97396f,  -0.42638f, -0.03123f, -5.49756f, -0.17029f, -0.11323f,
  0.05173f,  -0.44274f, -0.15738f, 0.11311f,  0.43872f,  0.16837f,  -0.52849f,
  2.90050f,  -0.54735f, -0.29591f, 1.24030f,  0.21696f,  -0.04443f, -1.60877f,
  -1.36365f, -1.27432f, -1.52060f, -1.34397f, -1.13371f, -1.87554f, 0.80123f,
  0.42820f,  -0.14157f, -2.73963f, -0.68040f, -0.35236f, 0.14490f,  2.23477f,
  0.01370f,  -0.20426f, -1.51411f, -0.72293f, 0.64516f,  0.97638f,  0.32616f,
  -0.27975f, -0.01149f,
};

static float av1_tx_type_nn_4x16_ver_layer0_bias[16] = {
  -1.37863f, -0.05763f, -0.07041f, 0.15306f,  0.96026f,  -1.42105f,
  -0.55822f, 1.04845f,  -0.17662f, -1.25345f, -0.11927f, 0.49845f,
  -0.32530f, 0.73483f,  0.08322f,  -0.23890f,
};

static float av1_tx_type_nn_4x16_ver_layer1_weights[64] = {
  0.27194f,  0.50607f,  0.49229f,  -0.48192f, 0.15667f,  -1.38891f, 0.38102f,
  -0.58825f, -0.07337f, -0.52909f, 0.36975f,  0.28710f,  0.34992f,  -0.73630f,
  0.30386f,  -0.58822f, 0.36127f,  0.57950f,  0.55878f,  -0.42796f, 0.19967f,
  -1.45517f, 0.42529f,  -0.54630f, -0.38169f, -0.84899f, 0.41622f,  0.46935f,
  0.39077f,  -0.75448f, 0.31698f,  -0.76187f, 0.97765f,  0.57052f,  0.55825f,
  -0.54273f, 0.20466f,  -1.46347f, 0.41813f,  -0.55019f, -0.19948f, -0.57982f,
  0.41206f,  0.32373f,  0.38537f,  -1.11657f, 0.32887f,  -0.76911f, 1.12259f,
  0.72163f,  0.82603f,  0.37786f,  0.34976f,  -1.86642f, 0.59961f,  -0.16329f,
  -0.36631f, -0.56814f, 0.60410f,  0.53158f,  0.56389f,  -0.70508f, 0.51009f,
  -0.56513f,
};

static float av1_tx_type_nn_4x16_ver_layer1_bias[4] = {
  4.60896f,
  4.53551f,
  4.53124f,
  4.27435f,
};

static float av1_tx_type_nn_4x16_ver_layer0_out[16] = { 0 };
static float av1_tx_type_nn_4x16_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_4x16_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                       // num_inputs
          16,                                      // num_outputs
          av1_tx_type_nn_4x16_ver_layer0_weights,  // weights
          av1_tx_type_nn_4x16_ver_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_4x16_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_4x16_ver_layer1_weights,
          av1_tx_type_nn_4x16_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_4x16_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_4x16_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Tx type model for 16x4 block.
static float av1_tx_type_nn_16x4_hor_layer0_weights[128] = {
  1.45347f,  -0.15743f, 0.44236f,  0.25808f,  0.33944f,  0.38678f,  0.24428f,
  1.67287f,  0.09539f,  -0.42940f, -0.31507f, -0.00154f, -2.98755f, -2.27744f,
  -0.49183f, 0.09333f,  -0.99026f, -0.22157f, 0.53701f,  0.60447f,  0.15686f,
  -0.04646f, 0.26341f,  2.12361f,  0.27090f,  -1.14716f, -0.64146f, -0.91604f,
  -0.75335f, -0.60056f, -1.25084f, 1.68473f,  -3.24075f, -4.03867f, -2.07877f,
  -0.02347f, 0.00333f,  -0.01259f, -0.00465f, 0.02526f,  0.36286f,  -0.10324f,
  2.12780f,  -0.74584f, -1.05052f, 1.78467f,  -0.55065f, -0.03326f, 2.46781f,
  1.18349f,  0.96015f,  1.01696f,  1.10584f,  1.07263f,  1.11531f,  -1.06413f,
  0.32389f,  -1.87360f, -0.14435f, 1.77926f,  1.09966f,  -0.12680f, -0.61386f,
  -0.09724f, -0.33095f, 1.12122f,  1.00791f,  1.52416f,  1.35004f,  1.32657f,
  0.60950f,  -1.13538f, -0.38654f, 0.06473f,  2.10669f,  0.27734f,  -0.38359f,
  -1.91455f, -1.22676f, 0.05786f,  0.97432f,  2.19967f,  0.50457f,  0.78976f,
  0.95183f,  -0.32414f, 0.49437f,  -0.04506f, 0.18993f,  -0.07971f, 0.23889f,
  -0.09872f, -0.66036f, 0.05377f,  2.69638f,  -0.08259f, -0.69210f, -1.08296f,
  -1.96504f, -2.31947f, -0.80161f, -0.80456f, -1.35556f, -0.05323f, -4.42658f,
  -0.30732f, -0.12043f, 0.11126f,  0.10771f,  -0.14956f, -0.02218f, 0.41016f,
  1.16599f,  1.14629f,  1.12881f,  1.18676f,  1.24677f,  1.28695f,  1.11270f,
  0.08233f,  1.75440f,  0.49228f,  -0.34858f, -0.17032f, 0.29288f,  0.47175f,
  0.19055f,  -1.56413f,
};

static float av1_tx_type_nn_16x4_hor_layer0_bias[16] = {
  -1.71227f, 0.47291f, -0.97536f, -0.66216f, 0.11729f,  -0.21451f,
  2.75281f,  0.04318f, 2.03965f,  0.14618f,  -0.70483f, -0.24517f,
  1.14048f,  0.33308f, -1.10886f, 0.41184f,
};

static float av1_tx_type_nn_16x4_hor_layer1_weights[64] = {
  -1.17079f, 0.19096f,  -1.05753f, -0.30803f, -1.21680f, -0.67255f, 1.60115f,
  0.05972f,  1.44759f,  -0.04068f, -0.26331f, 0.31400f,  0.96923f,  0.33443f,
  -0.77215f, -0.91316f, -1.78928f, 0.21483f,  -1.24008f, -0.46190f, -0.12127f,
  -0.62144f, 1.37593f,  0.08373f,  1.56215f,  0.00279f,  -0.14556f, 0.38710f,
  0.96228f,  0.66433f,  -0.51798f, -0.80738f, -0.18539f, 0.19377f,  -1.03090f,
  -1.51044f, -0.59485f, -0.62589f, 1.90742f,  0.09078f,  1.49113f,  0.00205f,
  -0.15918f, 0.40827f,  1.08553f,  0.43431f,  0.33519f,  -1.12669f, -1.10274f,
  0.80004f,  -1.83599f, -0.53134f, 2.00515f,  -0.32670f, 1.37124f,  0.51136f,
  1.62563f,  0.24787f,  0.31757f,  0.81751f,  1.57262f,  0.83214f,  1.04661f,
  -0.43819f,
};

static float av1_tx_type_nn_16x4_hor_layer1_bias[4] = {
  2.32575f,
  2.75703f,
  1.12304f,
  2.15567f,
};

static float av1_tx_type_nn_16x4_hor_layer0_out[16] = { 0 };
static float av1_tx_type_nn_16x4_hor_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_16x4_hor = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          8,                                       // num_inputs
          16,                                      // num_outputs
          av1_tx_type_nn_16x4_hor_layer0_weights,  // weights
          av1_tx_type_nn_16x4_hor_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_16x4_hor_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          16,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_16x4_hor_layer1_weights,
          av1_tx_type_nn_16x4_hor_layer1_bias,
          NONE,
          av1_tx_type_nn_16x4_hor_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_16x4_hor_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};

static float av1_tx_type_nn_16x4_ver_layer0_weights[32] = {
  0.26047f,  0.99930f,  1.16484f,  -0.28196f, -2.67483f, -0.21456f, -0.16854f,
  0.46375f,  1.47951f,  1.13735f,  1.12356f,  0.27385f,  0.50978f,  2.09967f,
  -1.47386f, 0.01950f,  -0.06362f, 0.26014f,  1.04544f,  -0.03099f, 0.07478f,
  -0.39701f, 0.05545f,  2.73633f,  -0.56305f, -0.02208f, -0.44517f, -0.00897f,
  -0.17967f, -0.96622f, 0.42635f,  -1.04784f,
};

static float av1_tx_type_nn_16x4_ver_layer0_bias[8] = {
  -0.52088f, 0.52844f,  -1.03655f, -0.30974f,
  2.59952f,  -1.93604f, 0.00000f,  2.51787f,
};

static float av1_tx_type_nn_16x4_ver_layer1_weights[32] = {
  0.10916f,  -0.21219f, -0.51340f, 0.69161f,  1.45988f,  -1.36942f, -0.40899f,
  1.05136f,  -0.08486f, 0.10008f,  -0.55304f, 0.88012f,  1.61177f,  -1.64507f,
  0.63428f,  1.15130f,  -0.17287f, -0.18592f, -0.01143f, 0.88293f,  1.73326f,
  -1.63624f, 0.09359f,  1.18393f,  0.26531f,  0.22378f,  0.15170f,  1.06965f,
  1.26814f,  -1.93873f, -0.00768f, 1.58309f,
};

static float av1_tx_type_nn_16x4_ver_layer1_bias[4] = {
  2.34713f,
  1.68667f,
  1.25488f,
  1.69812f,
};

static float av1_tx_type_nn_16x4_ver_layer0_out[8] = { 0 };
static float av1_tx_type_nn_16x4_ver_layer1_out[4] = { 0 };

static NN_CONFIG_V2 av1_tx_type_nnconfig_16x4_ver = {
  1,  // num_hidden_layers
  {
      // fc layer setting
      {
          // layer 0
          4,                                       // num_inputs
          8,                                       // num_outputs
          av1_tx_type_nn_16x4_ver_layer0_weights,  // weights
          av1_tx_type_nn_16x4_ver_layer0_bias,     // bias
          RELU,                                    // activation
          av1_tx_type_nn_16x4_ver_layer0_out,      // output
          NULL,
          NULL,
          NULL,
      },
      {
          8,  // num_inputs (!!same as num_outputs of last layer)
          4,
          av1_tx_type_nn_16x4_ver_layer1_weights,
          av1_tx_type_nn_16x4_ver_layer1_bias,
          NONE,
          av1_tx_type_nn_16x4_ver_layer1_out,
          NULL,
          NULL,
          NULL,
      },
  },
  4,                                   // num_outputs
  av1_tx_type_nn_16x4_ver_layer1_out,  // logits (!!same as last layer output)
  SOFTMAX_CROSS_ENTROPY,
};
/******************************************************************************/

// Map tx_size to its corresponding neural net model for tx type prediction.
static NN_CONFIG_V2 *av1_tx_type_nnconfig_map_hor[] = {
  &av1_tx_type_nnconfig_4x4_hor,   // 4x4 transform
  &av1_tx_type_nnconfig_8x8_hor,   // 8x8 transform
  &av1_tx_type_nnconfig_16x16,     // 16x16 transform
  NULL,                            // 32x32 transform
  NULL,                            // 64x64 transform
  &av1_tx_type_nnconfig_4x8_hor,   // 4x8 transform
  &av1_tx_type_nnconfig_8x4_hor,   // 8x4 transform
  &av1_tx_type_nnconfig_8x16_hor,  // 8x16 transform
  &av1_tx_type_nnconfig_16x8_hor,  // 16x8 transform
  NULL,                            // 16x32 transform
  NULL,                            // 32x16 transform
  NULL,                            // 32x64 transform
  NULL,                            // 64x32 transform
  &av1_tx_type_nnconfig_4x16_hor,  // 4x16 transform
  &av1_tx_type_nnconfig_16x4_hor,  // 16x4 transform
  NULL,                            // 8x32 transform
  NULL,                            // 32x8 transform
  NULL,                            // 16x64 transform
  NULL,                            // 64x16 transform
};

static NN_CONFIG_V2 *av1_tx_type_nnconfig_map_ver[] = {
  &av1_tx_type_nnconfig_4x4_ver,   // 4x4 transform
  &av1_tx_type_nnconfig_8x8_ver,   // 8x8 transform
  &av1_tx_type_nnconfig_16x16,     // 16x16 transform
  NULL,                            // 32x32 transform
  NULL,                            // 64x64 transform
  &av1_tx_type_nnconfig_4x8_ver,   // 4x8 transform
  &av1_tx_type_nnconfig_8x4_ver,   // 8x4 transform
  &av1_tx_type_nnconfig_8x16_ver,  // 8x16 transform
  &av1_tx_type_nnconfig_16x8_ver,  // 16x8 transform
  NULL,                            // 16x32 transform
  NULL,                            // 32x16 transform
  NULL,                            // 32x64 transform
  NULL,                            // 64x32 transform
  &av1_tx_type_nnconfig_4x16_ver,  // 4x16 transform
  &av1_tx_type_nnconfig_16x4_ver,  // 16x4 transform
  NULL,                            // 8x32 transform
  NULL,                            // 32x8 transform
  NULL,                            // 16x64 transform
  NULL,                            // 64x16 transform
};
#else
/******************************CONFIG_NN***************************************/
// Tx type model for 4x4 block.
static const float av1_tx_type_nn_weights_4x4_hor_layer0[32] = {
  -1.64947f, -1.54497f, -1.62832f, -0.17774f, -2.89498f, -0.72498f, 0.72036f,
  0.17996f,  1.20000f,  -0.27654f, 0.77396f,  1.21684f,  -1.75909f, -0.51272f,
  -1.25923f, 0.35005f,  -0.04257f, -0.23389f, -0.41841f, -0.08229f, 0.09503f,
  2.73144f,  -0.16875f, -0.23482f, 0.02194f,  -0.26427f, 0.28049f,  0.21260f,
  1.35792f,  0.27733f,  0.88660f,  -0.68304f,
};

static const float av1_tx_type_nn_bias_4x4_hor_layer0[8] = {
  1.38742f, 0.59540f,  -1.37622f, 1.92114f,
  0.00000f, -0.38998f, -0.32726f, -0.15650f,
};

static const float av1_tx_type_nn_weights_4x4_hor_layer1[32] = {
  1.65254f,  1.00915f,  -0.89318f, -2.05142f, -0.23235f, 0.96781f,  -0.37145f,
  -0.21056f, 1.13891f,  0.38675f,  0.87739f,  -1.42697f, 0.48015f,  0.61883f,
  -0.03979f, 0.11487f,  0.48042f,  0.45200f,  -0.23242f, 0.75166f,  0.55458f,
  0.39452f,  -0.35285f, 1.59120f,  -1.49221f, -0.48349f, -0.64692f, 1.49297f,
  -0.26782f, -0.65416f, -0.10648f, 0.05568f,
};

static const float av1_tx_type_nn_bias_4x4_hor_layer1[4] = {
  4.07177f,
  3.26961f,
  0.58083f,
  1.21199f,
};

static const NN_CONFIG av1_tx_type_nnconfig_4x4_hor = {
  4,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_4x4_hor_layer0,
    av1_tx_type_nn_weights_4x4_hor_layer1 },
  { av1_tx_type_nn_bias_4x4_hor_layer0, av1_tx_type_nn_bias_4x4_hor_layer1 }
};

static const float av1_tx_type_nn_weights_4x4_ver_layer0[32] = {
  -0.02032f, 2.61610f,  0.02098f,  -0.30217f, 0.12637f,  0.11017f,  -3.01996f,
  0.35144f,  1.93776f,  -0.20463f, 1.64102f,  -1.41986f, -3.66717f, -0.51655f,
  0.43910f,  0.37778f,  -1.02634f, 0.85337f,  -0.69753f, 1.00206f,  2.11784f,
  1.89427f,  1.92919f,  0.43201f,  -1.67358f, -1.67035f, -1.54623f, 0.16714f,
  -0.06589f, -0.28142f, -0.33118f, 1.72227f,
};

static const float av1_tx_type_nn_bias_4x4_ver_layer0[8] = {
  -0.33685f, 0.22025f,  0.28140f, 0.56138f,
  0.93489f,  -1.77048f, 1.34989f, -0.93747f,
};

static const float av1_tx_type_nn_weights_4x4_ver_layer1[32] = {
  -1.39506f, -1.06271f, -1.10886f, -1.69719f, 0.19699f,  -2.39850f, -1.26457f,
  0.75328f,  -1.26005f, -0.82738f, -0.12015f, -1.02702f, 1.40828f,  -2.37739f,
  -0.65639f, -0.71992f, -0.90453f, -1.12510f, -2.41362f, -1.16061f, -1.85577f,
  -0.99165f, -1.91366f, 0.16785f,  0.34776f,  0.58154f,  -0.18217f, -0.29257f,
  -0.86315f, -0.53336f, 0.30320f,  -1.32331f,
};

static const float av1_tx_type_nn_bias_4x4_ver_layer1[4] = {
  -1.31519f,
  -3.26321f,
  1.71794f,
  -1.90778f,
};

static const NN_CONFIG av1_tx_type_nnconfig_4x4_ver = {
  4,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_4x4_ver_layer0,
    av1_tx_type_nn_weights_4x4_ver_layer1 },
  { av1_tx_type_nn_bias_4x4_ver_layer0, av1_tx_type_nn_bias_4x4_ver_layer1 }
};
/******************************************************************************/

// Tx type model for 4x8 block.
static const float av1_tx_type_nn_weights_4x8_hor_layer0[32] = {
  0.00218f,  -0.41880f, -0.61215f, -0.92588f, 0.54291f,  -0.10898f, 0.70691f,
  0.46819f,  -1.61598f, -0.08834f, -0.96839f, 1.18489f,  -0.45171f, -0.65445f,
  -0.32179f, -0.10399f, 1.04379f,  0.91895f,  0.85589f,  0.08267f,  1.35388f,
  -2.03096f, 0.08168f,  -0.06372f, -0.26732f, -0.48262f, -0.08682f, 2.44071f,
  -1.35896f, -1.17121f, 1.68866f,  0.10357f,
};

static const float av1_tx_type_nn_bias_4x8_hor_layer0[8] = {
  2.93391f,  0.66831f, -0.21419f, 0.00000f,
  -0.72878f, 0.15127f, -1.46755f, 0.16658f,
};

static const float av1_tx_type_nn_weights_4x8_hor_layer1[32] = {
  -1.52077f, -1.06243f, 0.35319f,  -0.49207f, 0.54524f,  0.44271f, 1.37117f,
  -0.38957f, -1.28889f, -0.57133f, 0.04658f,  0.62278f,  0.37984f, 0.33247f,
  1.65547f,  -0.56806f, -1.38645f, -0.76258f, 0.67926f,  0.08783f, -0.01443f,
  0.34950f,  1.45812f,  -0.51332f, -1.41331f, -0.16453f, 0.05755f, 0.31405f,
  -0.50191f, 0.18219f,  1.83664f,  -0.75276f,
};

static const float av1_tx_type_nn_bias_4x8_hor_layer1[4] = {
  -1.17455f,
  -2.26089f,
  -1.79863f,
  -2.26333f,
};

static const NN_CONFIG av1_tx_type_nnconfig_4x8_hor = {
  4,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_4x8_hor_layer0,
    av1_tx_type_nn_weights_4x8_hor_layer1 },
  { av1_tx_type_nn_bias_4x8_hor_layer0, av1_tx_type_nn_bias_4x8_hor_layer1 }
};

static const float av1_tx_type_nn_weights_4x8_ver_layer0[128] = {
  -0.00952f, -0.98858f, -0.93181f, 1.39594f,  0.96559f,  0.18162f,  -0.76064f,
  -0.06066f, 0.07907f,  -0.09365f, -0.21313f, -0.02187f, -2.61707f, -2.68702f,
  -0.10982f, 0.18559f,  1.17049f,  1.11387f,  1.12697f,  1.05804f,  1.12764f,
  1.06318f,  1.12052f,  0.17406f,  1.83157f,  0.19362f,  0.46910f,  0.39608f,
  0.33342f,  0.40083f,  0.27645f,  1.06864f,  -4.06645f, -0.38775f, -0.11070f,
  0.03781f,  -0.09141f, 0.06185f,  -0.04852f, 0.20163f,  0.16784f,  0.16641f,
  -0.50941f, -0.61087f, 2.07008f,  -0.82381f, -0.85558f, 0.05528f,  -0.10535f,
  -2.81150f, 0.67038f,  0.43643f,  0.49062f,  -0.04465f, 0.90438f,  0.00977f,
  0.46272f,  1.59751f,  0.95234f,  0.35086f,  0.85624f,  0.73149f,  1.67779f,
  -2.21511f, -1.24746f, -1.09014f, -0.92441f, -1.22591f, -1.06961f, -0.95897f,
  -1.24956f, 0.73797f,  1.23275f,  -0.60064f, -0.07851f, 0.14397f,  0.22110f,
  -0.04422f, 0.14350f,  0.75926f,  0.35032f,  0.48104f,  2.81408f,  0.34662f,
  0.42090f,  0.35521f,  -1.36804f, -0.14974f, -0.47696f, -0.07892f, 0.36910f,
  0.32299f,  0.23916f,  0.06032f,  -0.17844f, -0.17558f, -1.42746f, -0.55828f,
  -1.00418f, -0.64823f, -0.73654f, -0.85197f, -1.50989f, 1.69385f,  -0.04973f,
  -0.09273f, 1.04249f,  0.79235f,  1.13229f,  0.99617f,  0.03851f,  0.56334f,
  0.90795f,  1.08296f,  0.58519f,  1.74765f,  0.63971f,  1.35951f,  0.07803f,
  -0.05127f, 0.26514f,  -0.84629f, -0.66343f, -2.10630f, 0.11017f,  2.18528f,
  -0.21958f, 0.05970f,
};

static const float av1_tx_type_nn_bias_4x8_ver_layer0[16] = {
  0.04205f, 0.22260f, -1.03870f, -1.19568f, 0.44283f,  0.01143f,
  0.00235f, 4.26772f, 0.44364f,  -0.33199f, -0.39076f, -0.35129f,
  0.08288f, 0.18195f, -0.79890f, 0.10047f,
};

static const float av1_tx_type_nn_weights_4x8_ver_layer1[64] = {
  -0.38193f, -0.12095f, 1.57802f,  0.34932f,  -0.47333f, -0.12304f, -0.01736f,
  -2.52445f, 0.18983f,  -0.64707f, -0.60889f, -0.53750f, 0.91666f,  -0.62823f,
  -0.13377f, -0.43594f, -0.38618f, -0.01328f, 0.97457f,  1.48589f,  -1.03238f,
  -0.33459f, -0.35108f, -2.42417f, 0.60229f,  0.06824f,  -0.75495f, 0.26902f,
  0.65311f,  -0.23887f, -0.44604f, -0.55800f, -0.33842f, 0.04259f,  -0.59589f,
  0.49738f,  -0.62301f, -0.30896f, -0.29602f, -2.57052f, 2.00943f,  -0.66490f,
  -0.76312f, 0.28256f,  1.06311f,  -0.38364f, -0.63508f, -0.57609f, -0.88765f,
  -1.04403f, -0.46531f, 0.34084f,  -1.20498f, -0.68352f, -0.72251f, -2.63242f,
  -0.68736f, -0.37904f, -1.32371f, 0.47288f,  1.51904f,  0.78372f,  -1.01830f,
  -1.01848f,
};

static const float av1_tx_type_nn_bias_4x8_ver_layer1[4] = {
  -1.45955f,
  -2.08949f,
  -1.24813f,
  -1.55368f,
};

static const NN_CONFIG av1_tx_type_nnconfig_4x8_ver = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_4x8_ver_layer0,
    av1_tx_type_nn_weights_4x8_ver_layer1 },
  { av1_tx_type_nn_bias_4x8_ver_layer0, av1_tx_type_nn_bias_4x8_ver_layer1 }
};
/******************************************************************************/

// Tx type model for 8x4 block.
static const float av1_tx_type_nn_weights_8x4_hor_layer0[128] = {
  -0.22492f, 0.13341f,  -4.03243f, -0.64015f, 0.02783f,  0.60466f,  -0.13335f,
  0.16828f,  0.12336f,  0.52904f,  1.18455f,  -0.32425f, 0.13052f,  0.93810f,
  -3.71165f, 0.02990f,  -4.63558f, 0.05666f,  0.03524f,  -0.07449f, -0.44006f,
  -0.33215f, -0.33713f, 0.08097f,  0.60873f,  0.29582f,  0.21696f,  -0.78729f,
  -0.16757f, -0.26567f, -0.00720f, -1.11226f, 1.58189f,  1.58463f,  1.48536f,
  1.54374f,  1.60069f,  1.46125f,  1.53932f,  0.05974f,  -1.82192f, 0.47043f,
  0.38090f,  0.20833f,  -0.05637f, 0.05183f,  0.01323f,  -0.25662f, 0.78634f,
  -0.55069f, -0.02975f, -1.29294f, -0.77192f, -2.34299f, -1.28074f, 0.77894f,
  -1.69740f, -1.66032f, -1.44323f, -1.55063f, -1.50845f, -1.23690f, -1.80663f,
  0.75079f,  2.32551f,  0.05878f,  0.80438f,  0.88584f,  0.69153f,  0.89060f,
  0.73660f,  0.87259f,  -0.00745f, -1.30044f, -0.59430f, 2.07270f,  1.03307f,
  -0.84697f, -1.19393f, 0.17549f,  -0.24978f, -3.67234f, 0.20781f,  -0.53946f,
  -0.05068f, 0.88274f,  1.30371f,  0.10288f,  0.07585f,  0.12259f,  -0.30815f,
  0.25437f,  -2.82096f, -2.69482f, 0.02370f,  0.12500f,  -0.21019f, -0.49220f,
  0.03638f,  -0.29795f, 0.28645f,  -0.48432f, -0.38584f, -0.32148f, -0.47197f,
  0.32437f,  0.32528f,  -0.19437f, 0.30383f,  -0.31879f, 0.26359f,  -0.12164f,
  -0.43647f, -0.08288f, -0.33438f, -0.63608f, -0.46647f, -0.46574f, 0.47806f,
  -0.49012f, -1.51234f, -1.13502f, -1.20470f, -1.02913f, -1.09182f, -0.93921f,
  -1.85523f, 0.92532f,
};

static const float av1_tx_type_nn_bias_8x4_hor_layer0[16] = {
  0.36631f,  0.02901f,  0.64305f,  1.53074f, -1.40229f, 0.03852f,
  -0.05043f, 0.89632f,  -1.23312f, 0.07036f, 0.17070f,  0.56250f,
  -0.28958f, -0.32869f, -0.01704f, 0.68171f,
};

static const float av1_tx_type_nn_weights_8x4_hor_layer1[64] = {
  -0.49441f, -0.31960f, -0.84946f, -0.85800f, -2.37767f, 0.81373f,  -0.73172f,
  -0.69337f, 0.88807f,  -0.49242f, -0.44717f, -0.11436f, 0.09978f,  0.15393f,
  0.17083f,  1.44850f,  -0.20582f, -0.04906f, 0.42990f,  -0.61939f, -1.09692f,
  -1.14885f, -1.36879f, -1.30828f, -0.59558f, -0.30903f, -0.08906f, 0.06953f,
  0.15383f,  -0.04193f, -0.54858f, 1.82676f,  -0.22411f, 0.05264f,  -0.45848f,
  -0.72985f, 0.87553f,  0.04116f,  -1.29774f, -2.63018f, 1.09089f,  -0.36048f,
  -0.16725f, 0.11627f,  0.49918f,  0.07539f,  0.00763f,  0.73706f,  0.87800f,
  0.57049f,  0.60969f,  1.02779f,  1.53339f,  -0.35915f, 0.06410f,  1.44582f,
  0.09698f,  0.71888f,  0.60594f,  0.84103f,  -0.50440f, -0.38825f, 0.15626f,
  -1.10654f,
};

static const float av1_tx_type_nn_bias_8x4_hor_layer1[4] = {
  -0.92861f,
  -1.45151f,
  -1.33588f,
  -4.33853f,
};

static const NN_CONFIG av1_tx_type_nnconfig_8x4_hor = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_8x4_hor_layer0,
    av1_tx_type_nn_weights_8x4_hor_layer1 },
  { av1_tx_type_nn_bias_8x4_hor_layer0, av1_tx_type_nn_bias_8x4_hor_layer1 }
};

static const float av1_tx_type_nn_weights_8x4_ver_layer0[32] = {
  -1.10946f, 1.86574f,  -1.59343f, 0.27018f, -1.70676f, -0.73982f, -0.19021f,
  -1.94208f, -2.29759f, -1.44402f, 0.28700f, -1.18340f, -1.50158f, -0.44175f,
  -1.36831f, 1.00374f,  2.59312f,  0.50291f, -0.71042f, -0.12238f, -0.15901f,
  -0.22807f, -0.67376f, -0.30215f, 0.54407f, -0.45538f, 1.18262f,  2.28687f,
  1.66212f,  1.70826f,  1.55182f,  0.12230f,
};

static const float av1_tx_type_nn_bias_8x4_ver_layer0[8] = {
  0.10943f,  2.09789f, 2.16578f, 0.15766f,
  -0.42461f, 0.00000f, 1.22090f, -1.28717f,
};

static const float av1_tx_type_nn_weights_8x4_ver_layer1[32] = {
  1.20426f,  -1.23237f, 2.41053f, -0.72488f, 1.25249f,  0.18018f,  -0.09586f,
  2.17901f,  0.15364f,  1.21535f, -0.38263f, -0.74309f, 0.50551f,  -0.54208f,
  0.59139f,  1.16095f,  0.55919f, -0.60183f, 1.18949f,  1.60787f,  0.54002f,
  -0.10712f, -0.16153f, 0.16207f, -0.32338f, 2.68712f,  -2.83483f, -0.27086f,
  -1.15005f, -0.39311f, 1.51236f, -1.68973f,
};

static const float av1_tx_type_nn_bias_8x4_ver_layer1[4] = {
  1.81013f,
  1.10517f,
  2.90059f,
  0.95391f,
};

static const NN_CONFIG av1_tx_type_nnconfig_8x4_ver = {
  4,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_8x4_ver_layer0,
    av1_tx_type_nn_weights_8x4_ver_layer1 },
  { av1_tx_type_nn_bias_8x4_ver_layer0, av1_tx_type_nn_bias_8x4_ver_layer1 }
};
/******************************************************************************/

// Tx type model for 8x8 block.
static const float av1_tx_type_nn_weights_8x8_hor_layer0[128] = {
  -0.85529f, 0.37619f,  0.12754f,  0.08622f,  0.45278f,  0.54929f,  1.60651f,
  -0.62654f, -0.54929f, -0.10131f, -0.17569f, 0.13948f,  0.31695f,  -0.05616f,
  0.20483f,  -0.36448f, 2.27203f,  -0.33087f, 0.47679f,  0.86888f,  0.39370f,
  0.46239f,  0.01113f,  1.50327f,  -1.48226f, -1.69621f, -1.49777f, -1.38885f,
  -1.37753f, -1.22681f, -1.70576f, 0.51329f,  -1.65662f, 1.74197f,  -0.13579f,
  -0.13133f, -0.58396f, -0.55510f, -1.10709f, -2.34975f, 0.22445f,  -0.56491f,
  -0.83432f, 0.13492f,  1.32147f,  2.85285f,  0.13819f,  0.03792f,  -1.30792f,
  0.04155f,  -0.70644f, -0.43430f, -0.16212f, -0.86945f, -1.16976f, 1.68339f,
  0.29540f,  0.01137f,  -0.25335f, -0.16856f, 0.12028f,  0.05207f,  0.39357f,
  -0.01545f, -0.21980f, -1.94091f, -1.01315f, -0.68270f, -0.40590f, -0.67111f,
  2.08283f,  0.19291f,  -4.81426f, -0.65044f, -0.24598f, 0.06371f,  -0.10272f,
  -0.14502f, -0.06821f, 0.45202f,  0.21091f,  -0.80864f, 0.39255f,  1.79189f,
  1.80453f,  1.10484f,  1.17608f,  0.96901f,  -0.35871f, -0.94311f, 0.63147f,
  2.95157f,  0.45917f,  -0.42849f, -0.55643f, -0.06097f, 3.49299f,  -0.50972f,
  0.11075f,  -0.08405f, -0.09274f, -0.22694f, -0.42426f, 0.48632f,  -1.61074f,
  1.82998f,  0.37623f,  -1.20330f, -0.01142f, -1.33307f, -0.27492f, -2.23621f,
  1.38846f,  1.42085f,  1.42568f,  1.36152f,  1.46910f,  1.27473f,  1.34752f,
  0.12753f,  -1.08197f, -1.08280f, -0.79489f, -1.12338f, -1.06795f, -0.87857f,
  -0.99892f, 1.09823f,
};

static const float av1_tx_type_nn_bias_8x8_hor_layer0[16] = {
  -0.49232f, -0.29685f, -1.44020f, 1.10940f,  1.16452f, -0.34862f,
  -0.38761f, -0.36243f, 0.21776f,  0.28234f,  2.34269f, -0.04104f,
  -0.26319f, 2.65579f,  -1.30137f, -0.01487f,
};

static const float av1_tx_type_nn_weights_8x8_hor_layer1[64] = {
  -0.38058f, -0.41295f, -1.26884f, -0.75560f, -1.57450f, 0.56072f,  -1.42322f,
  -0.29106f, 0.07228f,  0.04391f,  1.61388f,  -0.03055f, 0.81637f,  2.06045f,
  0.27119f,  -0.48328f, -0.45528f, -0.60534f, -1.61209f, -0.78157f, -1.65034f,
  0.60958f,  -1.30523f, 0.25143f,  0.11398f,  0.37860f,  1.54829f,  0.02309f,
  0.67288f,  2.11447f,  0.44845f,  -0.70406f, -0.67897f, -0.38759f, -1.30383f,
  -1.22646f, -1.54571f, 0.60552f,  -1.52565f, 0.11469f,  0.17344f,  0.08622f,
  1.57906f,  -0.00909f, 0.81634f,  2.04909f,  1.26466f,  -1.45741f, -0.75229f,
  0.06200f,  -1.05835f, -0.66257f, -1.73766f, 0.99923f,  -1.87082f, 0.14580f,
  0.49525f,  0.46839f,  1.32203f,  0.33923f,  0.97001f,  2.38584f,  1.58811f,
  0.06161f,
};

static const float av1_tx_type_nn_bias_8x8_hor_layer1[4] = {
  1.70385f,
  1.82373f,
  1.78496f,
  1.80826f,
};

static const NN_CONFIG av1_tx_type_nnconfig_8x8_hor = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_8x8_hor_layer0,
    av1_tx_type_nn_weights_8x8_hor_layer1 },
  { av1_tx_type_nn_bias_8x8_hor_layer0, av1_tx_type_nn_bias_8x8_hor_layer1 }
};

static const float av1_tx_type_nn_weights_8x8_ver_layer0[128] = {
  -0.67016f, -1.72366f, -1.86576f, -1.50962f, -1.70419f, -1.73964f, -1.84615f,
  2.09681f,  -0.05081f, -0.61030f, 2.02541f,  0.60222f,  0.99936f,  2.02114f,
  -0.53893f, -0.23757f, 0.73566f,  0.25443f,  0.00132f,  -0.74036f, -0.75351f,
  -0.76964f, -1.71007f, -0.15770f, 1.60982f,  2.17638f,  0.90681f,  0.64973f,
  0.85914f,  0.58786f,  -1.46228f, 0.05187f,  1.18804f,  0.30850f,  0.29512f,
  0.40526f,  0.37635f,  0.32311f,  0.37471f,  1.12346f,  3.41856f,  -0.36653f,
  0.42537f,  -0.19240f, 0.00155f,  0.30826f,  -0.02116f, -0.53435f, -0.34829f,
  -0.52466f, -0.11521f, -0.29163f, -2.05689f, -2.87372f, -0.62626f, 0.09585f,
  -0.75257f, 0.10057f,  1.43474f,  0.89450f,  0.75900f,  1.11147f,  1.00558f,
  0.25886f,  2.22095f,  -0.17926f, 0.57161f,  0.39546f,  0.47846f,  0.40452f,
  0.54298f,  0.45814f,  -3.62788f, -3.02374f, 0.03716f,  -0.13937f, -0.09415f,
  -0.12463f, 0.05682f,  0.03672f,  1.20746f,  1.25003f,  1.27071f,  1.31883f,
  1.27473f,  1.34943f,  1.23158f,  0.09039f,  0.19388f,  0.63420f,  2.79612f,
  0.93803f,  -0.11323f, -0.02027f, 0.41286f,  -0.05979f, -3.80705f, -0.52451f,
  -0.77098f, -0.68132f, -0.65559f, -0.60975f, -1.26165f, 0.25582f,  0.05346f,
  0.61403f,  0.32140f,  -2.39831f, -1.42355f, 1.30541f,  1.02361f,  0.12930f,
  -1.61469f, -0.77036f, -0.59144f, 1.27769f,  1.52068f,  0.82137f,  1.83159f,
  -0.66626f, -0.69806f, -1.00564f, -0.85995f, -0.90889f, -0.84412f, -0.85712f,
  -1.29848f, 0.39308f,
};

static const float av1_tx_type_nn_bias_8x8_ver_layer0[16] = {
  -0.14868f, -0.48343f, 3.94416f,  -0.78037f, -1.33789f, -0.60611f,
  0.51793f,  0.44030f,  -0.71563f, 0.22561f,  -1.19083f, -0.46149f,
  0.83015f,  0.06024f,  1.17180f,  0.65122f,
};

static const float av1_tx_type_nn_weights_8x8_ver_layer1[64] = {
  -1.42711f, -0.21683f, 2.12061f,  0.20489f,  -0.50228f, -0.24770f, 0.23391f,
  1.03470f,  -0.44847f, -0.63225f, -0.21583f, -0.06467f, -0.21892f, -0.07786f,
  1.43322f,  0.00280f,  -1.53057f, -0.18912f, 1.95333f,  0.31151f,  -2.07601f,
  0.06776f,  0.25529f,  0.94800f,  -1.11453f, -0.20594f, -0.13281f, 0.01485f,
  0.17650f,  -0.07955f, 1.43734f,  -0.23193f, -2.06463f, -0.21238f, 2.13707f,
  0.30351f,  0.27594f,  -0.36245f, 0.19539f,  0.91045f,  -0.24068f, -0.37616f,
  0.88792f,  0.02947f,  -0.16903f, -0.04932f, 1.51293f,  -0.95967f, -1.62903f,
  0.05326f,  2.30703f,  0.64445f,  -1.09464f, -0.16623f, 1.00240f,  0.07548f,
  -0.50406f, 0.63854f,  1.02340f,  0.49833f,  0.13671f,  0.26722f,  2.09516f,
  -0.41305f,
};

static const float av1_tx_type_nn_bias_8x8_ver_layer1[4] = {
  2.14067f,
  2.76699f,
  2.04233f,
  1.34803f,
};

static const NN_CONFIG av1_tx_type_nnconfig_8x8_ver = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_8x8_ver_layer0,
    av1_tx_type_nn_weights_8x8_ver_layer1 },
  { av1_tx_type_nn_bias_8x8_ver_layer0, av1_tx_type_nn_bias_8x8_ver_layer1 }
};
/******************************************************************************/

// Tx type model for 8x16 block.
static const float av1_tx_type_nn_weights_8x16_hor_layer0[128] = {
  -1.61872f, -1.58520f, -1.41236f, -1.53255f, -1.59794f, -1.25769f, -1.90043f,
  0.73431f,  1.10135f,  0.47054f,  0.43230f,  -0.43009f, -0.09135f, -0.07289f,
  -0.38785f, 1.23775f,  -0.35312f, 0.73789f,  0.88864f,  0.75957f,  0.62579f,
  0.46974f,  0.21851f,  1.63821f,  -2.27289f, -0.68522f, -0.69814f, -0.84368f,
  -0.91320f, -0.63055f, -1.03296f, 0.55778f,  -0.00071f, 1.27539f,  1.60068f,
  1.40975f,  0.97372f,  0.92843f,  1.90853f,  0.12626f,  1.71953f,  1.41978f,
  -0.12234f, -1.27058f, 0.76207f,  0.02495f,  -0.67038f, -0.05255f, 1.72923f,
  1.47630f,  1.47058f,  1.47614f,  1.49354f,  1.66131f,  1.50801f,  0.17145f,
  -2.30947f, -2.10850f, -1.25636f, -0.24900f, 0.72602f,  1.26572f,  0.97865f,
  -0.65466f, 1.31129f,  0.26916f,  0.12139f,  -0.12761f, -0.39143f, -0.28134f,
  0.06584f,  2.24418f,  0.22516f,  0.05011f,  -0.01671f, -0.29476f, -0.40326f,
  0.21138f,  -0.11573f, -0.31154f, -0.36828f, 0.03694f,  -0.07172f, -0.63419f,
  -3.14351f, -1.23125f, 0.65311f,  -0.11406f, 1.97287f,  -0.10422f, 0.83896f,
  0.85033f,  0.49724f,  0.80482f,  0.51454f,  1.06447f,  0.76693f,  0.72599f,
  -0.78573f, -0.53950f, 0.40894f,  0.00086f,  0.10784f,  -0.70498f, 1.16395f,
  1.14597f,  1.13496f,  1.12177f,  1.02100f,  -1.37574f, -2.97144f, 0.33899f,
  0.42013f,  0.86327f,  2.31983f,  2.04008f,  0.95503f,  0.15081f,  0.11530f,
  -0.02574f, -4.77119f, 0.13257f,  -0.01704f, -0.23087f, -0.00825f, 0.07029f,
  -0.28136f, 0.42556f,
};

static const float av1_tx_type_nn_bias_8x16_hor_layer0[16] = {
  0.93617f,  -0.24000f, -1.26821f, 0.78780f,  0.13690f, -0.21948f,
  -1.45162f, 0.44584f,  -1.92582f, -0.23169f, 0.56004f, -1.19937f,
  1.81560f,  -1.02643f, -0.81690f, 0.08302f,
};

static const float av1_tx_type_nn_weights_8x16_hor_layer1[64] = {
  0.06696f,  -0.11538f, -1.42029f, 0.32965f,  0.81046f,  0.01146f,  1.20945f,
  -0.16899f, 0.53224f,  -0.40232f, 0.01786f,  -0.73242f, 1.29750f,  1.95185f,
  0.70143f,  1.43287f,  0.76220f,  0.79937f,  -1.79011f, -1.15178f, 0.42526f,
  -0.67519f, 0.77267f,  -0.30697f, 2.46004f,  -0.49828f, 0.02875f,  1.09972f,
  1.47662f,  0.61719f,  0.61417f,  -0.12363f, 2.53048f,  0.00418f,  -1.38964f,
  0.88117f,  0.39239f,  -0.19347f, -2.58600f, -0.33715f, 1.09323f,  -0.32127f,
  0.02456f,  -0.19125f, 1.12728f,  0.66502f,  0.34296f,  1.14897f,  0.29967f,
  1.19209f,  0.22108f,  -0.11975f, 1.49776f,  -1.34624f, -2.58478f, -1.34632f,
  1.53207f,  0.45634f,  -1.48476f, 0.17489f,  0.71790f,  -2.12086f, -1.21778f,
  -1.31243f,
};

static const float av1_tx_type_nn_bias_8x16_hor_layer1[4] = {
  0.83359f,
  1.06875f,
  1.77645f,
  1.49570f,
};

static const NN_CONFIG av1_tx_type_nnconfig_8x16_hor = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_8x16_hor_layer0,
    av1_tx_type_nn_weights_8x16_hor_layer1 },
  { av1_tx_type_nn_bias_8x16_hor_layer0, av1_tx_type_nn_bias_8x16_hor_layer1 }
};

static const float av1_tx_type_nn_weights_8x16_ver_layer0[128] = {
  0.32858f,  -1.28887f, 0.25632f,  -0.05262f, 2.69203f,  -0.07004f, 1.37337f,
  -0.05725f, -0.05659f, 0.05592f,  0.01039f,  -0.29343f, 1.58628f,  -0.30003f,
  -3.43118f, 0.00272f,  1.70928f,  -0.76348f, 0.05889f,  -0.03263f, -0.07724f,
  0.03523f,  -0.19890f, 1.18005f,  -0.03605f, -0.20530f, -4.00733f, 0.10210f,
  -0.05368f, -0.17650f, -0.15317f, 0.06499f,  0.56705f,  1.04341f,  0.62890f,
  0.73451f,  -0.22199f, 0.86659f,  0.78443f,  -0.61664f, -0.50606f, 0.30247f,
  0.14455f,  0.39276f,  0.49203f,  0.65019f,  0.12269f,  1.64080f,  1.68289f,
  1.42694f,  1.60825f,  1.58501f,  1.47252f,  1.62589f,  1.48218f,  0.17726f,
  -0.04884f, 0.35376f,  -0.04796f, 0.32589f,  0.35087f,  0.35258f,  -0.46103f,
  -0.31176f, -0.05203f, 0.07247f,  -0.26756f, 0.22019f,  0.03412f,  0.33773f,
  0.29811f,  -0.11140f, 0.12831f,  -0.44673f, -0.09858f, 0.07889f,  0.15137f,
  0.00347f,  -0.23394f, 0.08886f,  -0.31201f, -0.79912f, -0.51092f, 0.14123f,
  -1.09599f, -4.26020f, -0.68675f, -0.02842f, -1.54538f, -1.28977f, -1.30558f,
  -1.21074f, -1.37142f, -1.14743f, -1.85397f, 0.82985f,  -0.30681f, 0.04494f,
  -0.24023f, -4.18053f, -0.16096f, -0.55492f, -0.27882f, 0.05829f,  -0.41224f,
  -2.52088f, -0.56162f, -1.04547f, -1.70685f, -0.28842f, -1.43673f, -0.01468f,
  -3.20585f, -0.69120f, -0.43931f, -0.46270f, -0.65885f, -0.55884f, -0.75138f,
  0.36381f,  -5.70858f, -0.14548f, -0.15745f, -0.11812f, -0.07605f, -0.07693f,
  -0.12236f, 0.16075f,
};

static const float av1_tx_type_nn_bias_8x16_ver_layer0[16] = {
  -0.35385f, 0.30491f,  -0.90011f, 0.42941f,  1.20928f, -0.88331f,
  -1.48818f, -0.34785f, -0.32668f, -0.22695f, 0.89188f, 0.65521f,
  0.57598f,  0.99819f,  0.75175f,  0.17044f,
};

static const float av1_tx_type_nn_weights_8x16_ver_layer1[64] = {
  -0.62913f, -0.34304f, 0.42963f,  -0.17440f, -1.44092f, 0.69142f,  -1.36067f,
  0.52211f,  0.44658f,  -0.26501f, -0.41657f, 0.34428f,  -0.34390f, -0.58567f,
  -0.84097f, -1.96311f, -0.37215f, -0.22250f, -1.23811f, -0.07247f, -0.81731f,
  0.58755f,  -1.30559f, 0.39551f,  0.41743f,  -0.09940f, -0.33230f, 0.14458f,
  -0.25139f, -0.54517f, 0.13469f,  -0.38157f, -0.39109f, -0.18205f, 0.06834f,
  -0.08395f, -0.92187f, 0.56724f,  1.44381f,  0.53226f,  -0.22356f, 0.12285f,
  -0.29418f, -1.86749f, -0.22372f, -0.60204f, -0.87746f, -1.16936f, 0.56884f,
  0.62641f,  -0.11823f, 1.00395f,  1.64794f,  -0.64535f, 2.29322f,  -0.23397f,
  0.17251f,  -0.35927f, 0.65631f,  -0.26812f, 0.80128f,  0.85748f,  0.47404f,
  2.20547f,
};

static const float av1_tx_type_nn_bias_8x16_ver_layer1[4] = {
  -0.44080f,
  -1.67455f,
  -1.46332f,
  -6.13206f,
};

static const NN_CONFIG av1_tx_type_nnconfig_8x16_ver = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_8x16_ver_layer0,
    av1_tx_type_nn_weights_8x16_ver_layer1 },
  { av1_tx_type_nn_bias_8x16_ver_layer0, av1_tx_type_nn_bias_8x16_ver_layer1 }
};
/******************************************************************************/

// Tx type model for 16x8 block.
static const float av1_tx_type_nn_weights_16x8_hor_layer0[128] = {
  0.02600f,  0.09786f,  -1.05107f, -0.35594f, -0.15658f, 2.99828f,  -0.07106f,
  -0.10101f, -0.14412f, -0.83790f, -0.19434f, 2.28368f,  1.91727f,  -0.00956f,
  -0.90640f, 0.09174f,  1.58895f,  1.38945f,  1.49431f,  1.51381f,  1.44803f,
  1.53544f,  1.44694f,  0.17753f,  1.69735f,  -0.78652f, 0.31092f,  -0.23736f,
  0.02231f,  -0.09884f, -0.00493f, 1.21189f,  -1.94382f, -0.34629f, -0.58309f,
  0.72291f,  -0.30056f, 0.90660f,  -0.57495f, 3.07809f,  0.73644f,  1.43050f,
  1.34356f,  -0.66554f, 0.50102f,  -0.64305f, 0.42044f,  -1.66165f, -0.05733f,
  -2.51402f, -1.01067f, -0.33390f, -0.32986f, -0.92431f, 1.86281f,  -0.07290f,
  -0.26290f, -0.68941f, 1.81156f,  0.66125f,  -2.09974f, 0.17032f,  -0.67461f,
  -0.00876f, -1.50154f, 1.17153f,  1.00377f,  0.33022f,  0.74689f,  0.42878f,
  0.61725f,  -0.83967f, 0.09467f,  -0.39892f, 0.33863f,  0.10656f,  -0.09249f,
  -0.39757f, 0.48481f,  -0.35162f, 1.47014f,  1.67827f,  -1.84051f, 0.16291f,
  -0.50135f, -2.29911f, -0.42217f, -0.13358f, 1.45899f,  -0.14743f, -0.02763f,
  -0.28003f, -0.01364f, 0.21014f,  -0.29026f, -0.20198f, 1.38782f,  0.56731f,
  0.27489f,  0.43227f,  0.41326f,  0.42721f,  0.87720f,  -1.90067f, -5.04951f,
  -0.17638f, -0.58119f, -0.08954f, -0.13692f, -0.12325f, -0.38548f, 0.66462f,
  -1.42377f, -1.21917f, -1.38193f, -1.36539f, -1.39378f, -1.19629f, -1.59812f,
  0.28689f,  0.32394f,  0.52128f,  0.01013f,  -0.28948f, -0.26293f, -0.44331f,
  -0.36570f, -0.50757f,
};

static const float av1_tx_type_nn_bias_16x8_hor_layer0[16] = {
  -0.08696f, -0.22110f, -1.43604f, -1.00451f, -1.51029f, 0.63736f,
  0.45260f,  0.16229f,  4.01393f,  -0.21748f, 0.36411f,  -0.08764f,
  -0.12329f, 0.08986f,  1.08117f,  -0.00220f,
};

static const float av1_tx_type_nn_weights_16x8_hor_layer1[64] = {
  0.55824f,  -0.14648f, 0.81947f,  -0.45867f, -1.86078f, -0.17291f, 0.34849f,
  0.15153f,  1.75625f,  -0.25760f, 0.72015f,  -0.30059f, -0.57975f, 0.07609f,
  -0.02036f, 0.07912f,  0.57080f,  -0.13792f, 0.74184f,  -0.87669f, -1.87572f,
  -0.27270f, 0.39751f,  0.19652f,  2.03514f,  -0.32944f, 0.76251f,  0.04399f,
  -0.63175f, 0.37420f,  0.08309f,  0.04466f,  0.60255f,  -0.12820f, 1.66065f,
  -0.59496f, -1.94794f, -0.14847f, 0.39424f,  0.16273f,  1.80587f,  0.41197f,
  0.74691f,  -0.21217f, -0.63173f, 0.09510f,  -0.35538f, -0.04407f, 0.92847f,
  0.20141f,  1.68680f,  -0.56528f, -2.26960f, 0.12978f,  0.73748f,  0.42438f,
  2.00673f,  -0.40189f, 0.95423f,  0.23234f,  -0.80953f, 0.65814f,  0.49444f,
  -0.23347f,
};

static const float av1_tx_type_nn_bias_16x8_hor_layer1[4] = {
  3.57175f,
  2.42612f,
  3.31259f,
  2.08287f,
};

static const NN_CONFIG av1_tx_type_nnconfig_16x8_hor = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_16x8_hor_layer0,
    av1_tx_type_nn_weights_16x8_hor_layer1 },
  { av1_tx_type_nn_bias_16x8_hor_layer0, av1_tx_type_nn_bias_16x8_hor_layer1 }
};

static const float av1_tx_type_nn_weights_16x8_ver_layer0[128] = {
  0.46633f,  1.55328f,  -0.11230f, -0.29571f, 0.18814f,  -1.52430f, -2.34660f,
  0.08644f,  -1.97718f, -1.29140f, -1.12262f, -1.12985f, -1.25911f, -0.96506f,
  -1.57129f, 0.96021f,  1.34192f,  1.28623f,  1.21655f,  1.28758f,  1.25482f,
  1.30195f,  1.19190f,  0.09310f,  0.52072f,  0.91487f,  1.24100f,  1.61236f,
  1.72166f,  2.20750f,  1.62379f,  -1.43936f, 0.50665f,  0.40213f,  0.66502f,
  -1.66699f, -3.07618f, 0.05877f,  0.60987f,  -0.09995f, -0.10916f, 0.48049f,
  0.23812f,  0.39847f,  -0.21682f, -0.63455f, 0.33453f,  -0.67939f, -4.14355f,
  -0.62756f, -0.22502f, -0.17215f, 0.01062f,  0.27049f,  -0.10748f, 0.30945f,
  2.72445f,  -0.89181f, -0.06800f, 0.20595f,  -0.73385f, 0.04071f,  -1.30294f,
  1.83507f,  0.92570f,  0.69609f,  0.76285f,  0.69892f,  0.76409f,  0.63104f,
  0.73397f,  1.09575f,  -0.20129f, -0.24022f, -0.24599f, -0.59107f, -0.88755f,
  -0.68987f, -0.75495f, -1.31002f, -1.30237f, -0.94093f, -2.15678f, -1.49303f,
  -1.17498f, -1.39952f, -0.91270f, -0.05587f, 1.02381f,  -0.75580f, -0.65263f,
  -0.78996f, -0.71075f, -0.71018f, -0.70350f, -1.26196f, 2.34208f,  -0.53611f,
  0.19752f,  -0.16842f, -0.24828f, 0.21857f,  0.08222f,  -2.55894f, -1.75702f,
  0.11394f,  1.03083f,  0.79972f,  -1.54112f, -1.82341f, -0.57597f, -0.02077f,
  -0.39616f, -0.00995f, -0.12809f, 0.01188f,  -0.25117f, 0.09202f,  0.09336f,
  -0.05614f, -0.30039f, 0.25834f,  1.19944f,  1.22533f,  0.92330f,  0.75967f,
  -0.81945f, -0.41647f,
};

static const float av1_tx_type_nn_bias_16x8_ver_layer0[16] = {
  0.17841f,  0.67315f,  -1.24450f, 3.13859f,  0.16203f, -0.14992f,
  0.29553f,  -1.15567f, -0.71421f, 1.15977f,  1.14585f, 3.02460f,
  -0.04510f, 0.48000f,  -0.09354f, -0.42422f,
};

static const float av1_tx_type_nn_weights_16x8_ver_layer1[64] = {
  0.29912f,  -0.10009f, -1.11478f, 1.76812f,  -0.27719f, 0.52148f,  0.17622f,
  -1.17116f, 0.73397f,  -0.69279f, -0.11080f, 1.53751f,  -1.42003f, 0.14731f,
  0.13592f,  -0.04883f, 0.39186f,  -0.13655f, -0.43994f, 1.82759f,  -0.25601f,
  -0.15018f, 0.51920f,  -1.56070f, 0.31683f,  -0.79367f, -0.02904f, 1.28637f,
  -1.15203f, 0.26627f,  0.42828f,  -0.24258f, 0.38647f,  -0.83352f, 0.32553f,
  2.09522f,  -0.26822f, -0.42191f, 0.32825f,  -1.30748f, 1.50551f,  -0.52669f,
  0.20045f,  1.69318f,  -1.47839f, 0.30802f,  -0.07290f, -0.28106f, 0.68192f,
  -0.15522f, 1.12579f,  2.21921f,  0.09720f,  -0.50265f, 0.83165f,  -1.31721f,
  0.72422f,  -1.24952f, 0.61653f,  2.04117f,  -1.42406f, 0.52568f,  -0.46180f,
  -0.00873f,
};

static const float av1_tx_type_nn_bias_16x8_ver_layer1[4] = {
  3.34981f,
  3.74710f,
  1.38339f,
  0.45176f,
};

static const NN_CONFIG av1_tx_type_nnconfig_16x8_ver = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_16x8_ver_layer0,
    av1_tx_type_nn_weights_16x8_ver_layer1 },
  { av1_tx_type_nn_bias_16x8_ver_layer0, av1_tx_type_nn_bias_16x8_ver_layer1 }
};
/******************************************************************************/

// Tx type model for 16x16 block.
static const float av1_tx_type_nn_weights_16x16_layer0[128] = {
  1.26592f,  1.36313f,  1.30956f,  1.29926f,  1.48816f,  1.68851f,  1.32000f,
  0.13321f,  -0.22477f, -0.88906f, -0.19622f, 1.69605f,  1.22180f,  -1.57771f,
  -1.15765f, 0.05710f,  -1.13355f, -0.85486f, -0.99971f, -0.91571f, -1.06031f,
  -0.77952f, -1.15723f, 1.17809f,  1.35602f,  -0.05243f, -0.37596f, 0.26108f,
  0.17611f,  -0.10323f, 0.77279f,  -0.48911f, -0.79308f, 0.55112f,  0.43918f,
  0.27872f,  0.28714f,  0.45830f,  1.05689f,  0.03705f,  -2.49975f, -0.01940f,
  0.05709f,  0.07942f,  -0.13290f, -0.10359f, 0.00143f,  0.37303f,  0.96470f,
  0.53293f,  1.14459f,  0.89185f,  0.43378f,  0.47764f,  0.90924f,  0.15279f,
  -0.15361f, 0.02949f,  0.42240f,  0.68143f,  0.89588f,  0.73754f,  0.10974f,
  1.57755f,  -0.39870f, -0.32914f, 0.35638f,  0.34991f,  -0.00003f, -0.23373f,
  0.29630f,  -0.76699f, -0.01356f, 0.04234f,  0.84253f,  1.92078f,  0.93160f,
  0.71993f,  0.71604f,  0.76455f,  -1.59782f, 0.32332f,  1.11628f,  0.33062f,
  -0.03728f, -0.05710f, 0.80447f,  -0.14719f, 1.34658f,  -0.05718f, 0.64015f,
  0.21926f,  0.41653f,  0.12720f,  0.54092f,  1.39411f,  1.81819f,  -0.24513f,
  0.00955f,  0.38011f,  -0.57787f, -0.41759f, 0.68834f,  -0.31783f, -0.40607f,
  -0.10107f, -0.79374f, 0.75599f,  -0.16282f, -0.14490f, -0.20783f, -0.55019f,
  -0.13793f, -0.22293f, 0.18305f,  0.12445f,  0.56830f,  0.24567f,  0.09278f,
  0.70803f,  0.35803f,  -1.52676f, -0.89624f, 0.77665f,  0.19877f,  0.77175f,
  0.50355f,  0.08592f,
};

static const float av1_tx_type_nn_bias_16x16_layer0[16] = {
  -1.31834f, 0.14346f,  -0.10062f, 0.84489f,  0.95617f,  -0.06720f,
  -0.68502f, -0.91442f, -0.31932f, 0.25276f,  -0.15138f, -1.57661f,
  -0.14062f, -0.42120f, 0.94573f,  -0.09287f,
};

static const float av1_tx_type_nn_weights_16x16_layer1[64] = {
  -1.80333f, -1.06353f, 0.55139f,  0.74644f,  0.13747f, -0.93018f, -0.10286f,
  0.67133f,  0.24460f,  1.44583f,  0.02173f,  0.26037f, -0.73687f, 0.19566f,
  0.61846f,  -0.58601f, -1.03196f, -0.74415f, 0.30041f, -0.41967f, 1.08740f,
  0.96224f,  -0.59139f, 0.03813f,  0.05403f,  1.33427f, -0.54375f, -1.92181f,
  0.54704f,  0.13608f,  0.22151f,  -0.38076f, 1.18390f, -0.77508f, -1.84283f,
  1.00894f,  0.62318f,  -0.15296f, 1.27600f,  0.22822f, 0.12751f,  0.93910f,
  -0.28502f, 0.53912f,  -0.96889f, 0.10182f,  0.81508f, -0.43028f, 2.67386f,
  0.52204f,  0.49820f,  -0.41711f, 1.05038f,  1.12192f, 0.74349f,  -0.75417f,
  -0.03718f, -0.35769f, 0.89651f,  0.63236f,  0.54215f, -0.07894f, 0.48274f,
  1.08829f,
};

static const float av1_tx_type_nn_bias_16x16_layer1[4] = {
  0.81986f,
  1.26865f,
  0.11118f,
  2.48404f,
};

static const NN_CONFIG av1_tx_type_nnconfig_16x16 = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  {
      av1_tx_type_nn_weights_16x16_layer0,
      av1_tx_type_nn_weights_16x16_layer1,
  },
  {
      av1_tx_type_nn_bias_16x16_layer0,
      av1_tx_type_nn_bias_16x16_layer1,
  },
};
/******************************************************************************/

// Tx type model for 4x16 block.
static const float av1_tx_type_nn_weights_4x16_hor_layer0[32] = {
  0.36539f,  0.25667f,  0.01491f,  -0.21959f, 2.55105f,  0.17615f, 1.79884f,
  1.65936f,  -0.44363f, 0.00706f,  -0.68004f, -0.64360f, 1.75760f, 1.91906f,
  1.47682f,  0.09650f,  -3.59244f, -0.35004f, 0.93295f,  0.25806f, -0.08154f,
  0.79332f,  0.79535f,  1.09467f,  1.57855f,  -0.51359f, 0.90553f, -1.67744f,
  -1.74563f, -0.88830f, -1.77603f, 2.15935f,
};

static const float av1_tx_type_nn_bias_4x16_hor_layer0[8] = {
  -0.36435f, -2.22731f, -0.00837f, -1.34546f,
  0.62806f,  -0.20675f, 4.91940f,  -0.56079f,
};

static const float av1_tx_type_nn_weights_4x16_hor_layer1[32] = {
  -0.57191f, -1.46418f, 0.67331f,  -1.15027f, 0.46288f,  0.81251f,  2.51768f,
  -0.27147f, 0.00761f,  -2.15214f, -0.69650f, -0.50808f, 0.92832f,  0.45668f,
  2.34201f,  -0.52941f, 0.51008f,  -1.55496f, -0.01371f, -0.12356f, 0.66624f,
  0.88043f,  2.64862f,  -1.28024f, -0.17578f, -1.80034f, -0.32217f, 0.89519f,
  1.28413f,  -0.30326f, 2.45329f,  -0.83335f,
};

static const float av1_tx_type_nn_bias_4x16_hor_layer1[4] = {
  2.33198f,
  3.36245f,
  1.62603f,
  2.91056f,
};

static const NN_CONFIG av1_tx_type_nnconfig_4x16_hor = {
  4,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_4x16_hor_layer0,
    av1_tx_type_nn_weights_4x16_hor_layer1 },
  { av1_tx_type_nn_bias_4x16_hor_layer0, av1_tx_type_nn_bias_4x16_hor_layer1 }
};

static const float av1_tx_type_nn_weights_4x16_ver_layer0[128] = {
  1.61392f,  1.41239f,  1.47646f,  1.47325f,  1.46110f,  1.49208f,  1.49414f,
  0.12835f,  -0.76986f, 0.07087f,  -0.24572f, -0.93168f, 3.07935f,  -0.18183f,
  -0.09831f, -0.07703f, -0.03222f, -0.25473f, -0.06090f, 2.93713f,  -0.38711f,
  -0.12884f, -0.18329f, -0.06262f, -0.00327f, -0.02930f, -0.01641f, -0.00622f,
  -0.03305f, -4.07069f, -2.76643f, 0.04413f,  -1.03176f, -0.19217f, -0.44980f,
  -2.48615f, -2.58112f, -0.87695f, 0.16187f,  -0.04891f, -0.06854f, 1.08104f,
  0.75245f,  1.49302f,  0.63363f,  1.45715f,  0.92574f,  1.72029f,  0.33326f,
  3.86646f,  0.04422f,  0.41019f,  0.36212f,  0.56600f,  -1.01552f, 0.05128f,
  0.40454f,  -1.05100f, -0.47461f, -1.33168f, -0.46145f, -1.36870f, -0.88838f,
  -1.05358f, -0.18537f, -0.34357f, -0.03698f, 0.68905f,  0.41010f,  0.31223f,
  -0.43382f, -0.74715f, 2.03366f,  -0.30419f, 0.45747f,  0.09526f,  0.31678f,
  0.22915f,  0.21832f,  1.26385f,  -0.06814f, -0.71417f, -1.18947f, 0.03762f,
  0.10936f,  2.97396f,  -0.42638f, -0.03123f, -5.49756f, -0.17029f, -0.11323f,
  0.05173f,  -0.44274f, -0.15738f, 0.11311f,  0.43872f,  0.16837f,  -0.52849f,
  2.90050f,  -0.54735f, -0.29591f, 1.24030f,  0.21696f,  -0.04443f, -1.60877f,
  -1.36365f, -1.27432f, -1.52060f, -1.34397f, -1.13371f, -1.87554f, 0.80123f,
  0.42820f,  -0.14157f, -2.73963f, -0.68040f, -0.35236f, 0.14490f,  2.23477f,
  0.01370f,  -0.20426f, -1.51411f, -0.72293f, 0.64516f,  0.97638f,  0.32616f,
  -0.27975f, -0.01149f,
};

static const float av1_tx_type_nn_bias_4x16_ver_layer0[16] = {
  -1.37863f, -0.05763f, -0.07041f, 0.15306f,  0.96026f,  -1.42105f,
  -0.55822f, 1.04845f,  -0.17662f, -1.25345f, -0.11927f, 0.49845f,
  -0.32530f, 0.73483f,  0.08322f,  -0.23890f,
};

static const float av1_tx_type_nn_weights_4x16_ver_layer1[64] = {
  0.27194f,  0.50607f,  0.49229f,  -0.48192f, 0.15667f,  -1.38891f, 0.38102f,
  -0.58825f, -0.07337f, -0.52909f, 0.36975f,  0.28710f,  0.34992f,  -0.73630f,
  0.30386f,  -0.58822f, 0.36127f,  0.57950f,  0.55878f,  -0.42796f, 0.19967f,
  -1.45517f, 0.42529f,  -0.54630f, -0.38169f, -0.84899f, 0.41622f,  0.46935f,
  0.39077f,  -0.75448f, 0.31698f,  -0.76187f, 0.97765f,  0.57052f,  0.55825f,
  -0.54273f, 0.20466f,  -1.46347f, 0.41813f,  -0.55019f, -0.19948f, -0.57982f,
  0.41206f,  0.32373f,  0.38537f,  -1.11657f, 0.32887f,  -0.76911f, 1.12259f,
  0.72163f,  0.82603f,  0.37786f,  0.34976f,  -1.86642f, 0.59961f,  -0.16329f,
  -0.36631f, -0.56814f, 0.60410f,  0.53158f,  0.56389f,  -0.70508f, 0.51009f,
  -0.56513f,
};

static const float av1_tx_type_nn_bias_4x16_ver_layer1[4] = {
  4.60896f,
  4.53551f,
  4.53124f,
  4.27435f,
};

static const NN_CONFIG av1_tx_type_nnconfig_4x16_ver = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_4x16_ver_layer0,
    av1_tx_type_nn_weights_4x16_ver_layer1 },
  { av1_tx_type_nn_bias_4x16_ver_layer0, av1_tx_type_nn_bias_4x16_ver_layer1 }
};
/******************************************************************************/

// Tx type model for 16x4 block.
static const float av1_tx_type_nn_weights_16x4_hor_layer0[128] = {
  1.45347f,  -0.15743f, 0.44236f,  0.25808f,  0.33944f,  0.38678f,  0.24428f,
  1.67287f,  0.09539f,  -0.42940f, -0.31507f, -0.00154f, -2.98755f, -2.27744f,
  -0.49183f, 0.09333f,  -0.99026f, -0.22157f, 0.53701f,  0.60447f,  0.15686f,
  -0.04646f, 0.26341f,  2.12361f,  0.27090f,  -1.14716f, -0.64146f, -0.91604f,
  -0.75335f, -0.60056f, -1.25084f, 1.68473f,  -3.24075f, -4.03867f, -2.07877f,
  -0.02347f, 0.00333f,  -0.01259f, -0.00465f, 0.02526f,  0.36286f,  -0.10324f,
  2.12780f,  -0.74584f, -1.05052f, 1.78467f,  -0.55065f, -0.03326f, 2.46781f,
  1.18349f,  0.96015f,  1.01696f,  1.10584f,  1.07263f,  1.11531f,  -1.06413f,
  0.32389f,  -1.87360f, -0.14435f, 1.77926f,  1.09966f,  -0.12680f, -0.61386f,
  -0.09724f, -0.33095f, 1.12122f,  1.00791f,  1.52416f,  1.35004f,  1.32657f,
  0.60950f,  -1.13538f, -0.38654f, 0.06473f,  2.10669f,  0.27734f,  -0.38359f,
  -1.91455f, -1.22676f, 0.05786f,  0.97432f,  2.19967f,  0.50457f,  0.78976f,
  0.95183f,  -0.32414f, 0.49437f,  -0.04506f, 0.18993f,  -0.07971f, 0.23889f,
  -0.09872f, -0.66036f, 0.05377f,  2.69638f,  -0.08259f, -0.69210f, -1.08296f,
  -1.96504f, -2.31947f, -0.80161f, -0.80456f, -1.35556f, -0.05323f, -4.42658f,
  -0.30732f, -0.12043f, 0.11126f,  0.10771f,  -0.14956f, -0.02218f, 0.41016f,
  1.16599f,  1.14629f,  1.12881f,  1.18676f,  1.24677f,  1.28695f,  1.11270f,
  0.08233f,  1.75440f,  0.49228f,  -0.34858f, -0.17032f, 0.29288f,  0.47175f,
  0.19055f,  -1.56413f,
};

static const float av1_tx_type_nn_bias_16x4_hor_layer0[16] = {
  -1.71227f, 0.47291f, -0.97536f, -0.66216f, 0.11729f,  -0.21451f,
  2.75281f,  0.04318f, 2.03965f,  0.14618f,  -0.70483f, -0.24517f,
  1.14048f,  0.33308f, -1.10886f, 0.41184f,
};

static const float av1_tx_type_nn_weights_16x4_hor_layer1[64] = {
  -1.17079f, 0.19096f,  -1.05753f, -0.30803f, -1.21680f, -0.67255f, 1.60115f,
  0.05972f,  1.44759f,  -0.04068f, -0.26331f, 0.31400f,  0.96923f,  0.33443f,
  -0.77215f, -0.91316f, -1.78928f, 0.21483f,  -1.24008f, -0.46190f, -0.12127f,
  -0.62144f, 1.37593f,  0.08373f,  1.56215f,  0.00279f,  -0.14556f, 0.38710f,
  0.96228f,  0.66433f,  -0.51798f, -0.80738f, -0.18539f, 0.19377f,  -1.03090f,
  -1.51044f, -0.59485f, -0.62589f, 1.90742f,  0.09078f,  1.49113f,  0.00205f,
  -0.15918f, 0.40827f,  1.08553f,  0.43431f,  0.33519f,  -1.12669f, -1.10274f,
  0.80004f,  -1.83599f, -0.53134f, 2.00515f,  -0.32670f, 1.37124f,  0.51136f,
  1.62563f,  0.24787f,  0.31757f,  0.81751f,  1.57262f,  0.83214f,  1.04661f,
  -0.43819f,
};

static const float av1_tx_type_nn_bias_16x4_hor_layer1[4] = {
  2.32575f,
  2.75703f,
  1.12304f,
  2.15567f,
};

static const NN_CONFIG av1_tx_type_nnconfig_16x4_hor = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_16x4_hor_layer0,
    av1_tx_type_nn_weights_16x4_hor_layer1 },
  { av1_tx_type_nn_bias_16x4_hor_layer0, av1_tx_type_nn_bias_16x4_hor_layer1 }
};

static const float av1_tx_type_nn_weights_16x4_ver_layer0[32] = {
  0.26047f,  0.99930f,  1.16484f,  -0.28196f, -2.67483f, -0.21456f, -0.16854f,
  0.46375f,  1.47951f,  1.13735f,  1.12356f,  0.27385f,  0.50978f,  2.09967f,
  -1.47386f, 0.01950f,  -0.06362f, 0.26014f,  1.04544f,  -0.03099f, 0.07478f,
  -0.39701f, 0.05545f,  2.73633f,  -0.56305f, -0.02208f, -0.44517f, -0.00897f,
  -0.17967f, -0.96622f, 0.42635f,  -1.04784f,
};

static const float av1_tx_type_nn_bias_16x4_ver_layer0[8] = {
  -0.52088f, 0.52844f,  -1.03655f, -0.30974f,
  2.59952f,  -1.93604f, 0.00000f,  2.51787f,
};

static const float av1_tx_type_nn_weights_16x4_ver_layer1[32] = {
  0.10916f,  -0.21219f, -0.51340f, 0.69161f,  1.45988f,  -1.36942f, -0.40899f,
  1.05136f,  -0.08486f, 0.10008f,  -0.55304f, 0.88012f,  1.61177f,  -1.64507f,
  0.63428f,  1.15130f,  -0.17287f, -0.18592f, -0.01143f, 0.88293f,  1.73326f,
  -1.63624f, 0.09359f,  1.18393f,  0.26531f,  0.22378f,  0.15170f,  1.06965f,
  1.26814f,  -1.93873f, -0.00768f, 1.58309f,
};

static const float av1_tx_type_nn_bias_16x4_ver_layer1[4] = {
  2.34713f,
  1.68667f,
  1.25488f,
  1.69812f,
};

static const NN_CONFIG av1_tx_type_nnconfig_16x4_ver = {
  4,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  { av1_tx_type_nn_weights_16x4_ver_layer0,
    av1_tx_type_nn_weights_16x4_ver_layer1 },
  { av1_tx_type_nn_bias_16x4_ver_layer0, av1_tx_type_nn_bias_16x4_ver_layer1 }
};
/******************************************************************************/

// Map tx_size to its corresponding neural net model for tx type prediction.
static const NN_CONFIG *av1_tx_type_nnconfig_map_hor[] = {
  &av1_tx_type_nnconfig_4x4_hor,   // 4x4 transform
  &av1_tx_type_nnconfig_8x8_hor,   // 8x8 transform
  &av1_tx_type_nnconfig_16x16,     // 16x16 transform
  NULL,                            // 32x32 transform
  NULL,                            // 64x64 transform
  &av1_tx_type_nnconfig_4x8_hor,   // 4x8 transform
  &av1_tx_type_nnconfig_8x4_hor,   // 8x4 transform
  &av1_tx_type_nnconfig_8x16_hor,  // 8x16 transform
  &av1_tx_type_nnconfig_16x8_hor,  // 16x8 transform
  NULL,                            // 16x32 transform
  NULL,                            // 32x16 transform
  NULL,                            // 32x64 transform
  NULL,                            // 64x32 transform
  &av1_tx_type_nnconfig_4x16_hor,  // 4x16 transform
  &av1_tx_type_nnconfig_16x4_hor,  // 16x4 transform
  NULL,                            // 8x32 transform
  NULL,                            // 32x8 transform
  NULL,                            // 16x64 transform
  NULL,                            // 64x16 transform
};

static const NN_CONFIG *av1_tx_type_nnconfig_map_ver[] = {
  &av1_tx_type_nnconfig_4x4_ver,   // 4x4 transform
  &av1_tx_type_nnconfig_8x8_ver,   // 8x8 transform
  &av1_tx_type_nnconfig_16x16,     // 16x16 transform
  NULL,                            // 32x32 transform
  NULL,                            // 64x64 transform
  &av1_tx_type_nnconfig_4x8_ver,   // 4x8 transform
  &av1_tx_type_nnconfig_8x4_ver,   // 8x4 transform
  &av1_tx_type_nnconfig_8x16_ver,  // 8x16 transform
  &av1_tx_type_nnconfig_16x8_ver,  // 16x8 transform
  NULL,                            // 16x32 transform
  NULL,                            // 32x16 transform
  NULL,                            // 32x64 transform
  NULL,                            // 64x32 transform
  &av1_tx_type_nnconfig_4x16_ver,  // 4x16 transform
  &av1_tx_type_nnconfig_16x4_ver,  // 16x4 transform
  NULL,                            // 8x32 transform
  NULL,                            // 32x8 transform
  NULL,                            // 16x64 transform
  NULL,                            // 64x16 transform
};
#endif  // CONFIG_NN_V2

#if CONFIG_NEW_TX_PARTITION
// Tx split model for 8x8 block.
static const float av1_tx_split_nn_weights_8x8_layer0[144] = {
  0.373085f,   0.155979f,  0.828105f,  -0.144805f, 0.026844f,  0.211814f,
  -0.100931f,  0.193598f,  0.204902f,  -0.354949f, -0.203362f, 0.345391f,
  -10.402082f, -6.193234f, -6.349980f, -6.604737f, -1.971066f, -3.937962f,
  -6.708403f,  -0.955257f, -8.276703f, -3.821987f, -6.805996f, -7.041824f,
  0.192220f,   0.115800f,  0.310568f,  -0.495795f, -0.025454f, -0.166161f,
  0.011903f,   0.140433f,  -0.022533f, 0.140990f,  -0.010818f, 0.485021f,
  -8.254085f,  -5.543548f, 1.294869f,  1.076928f,  -0.578802f, -4.089819f,
  0.826976f,   -0.488747f, 0.779672f,  1.717556f,  1.433417f,  1.237000f,
  -0.070989f,  -0.158314f, 0.203960f,  -0.396725f, 0.005808f,  -0.151146f,
  -0.015481f,  0.086705f,  -0.022262f, 0.087614f,  -0.002145f, 0.347355f,
  -7.800653f,  -6.111475f, 1.315977f,  1.159693f,  -0.819767f, -4.415745f,
  0.740404f,   -0.520667f, 1.052980f,  1.258964f,  1.335589f,  1.246341f,
  -0.339928f,  -0.095897f, 0.272153f,  -0.517880f, 0.007502f,  -0.011646f,
  -0.005847f,  0.107345f,  -0.111115f, 0.081656f,  0.015800f,  0.449531f,
  -10.274256f, -5.813243f, 1.299438f,  1.053578f,  1.063611f,  -3.159878f,
  0.845408f,   -0.440582f, 1.124792f,  -0.402570f, 1.430974f,  1.274865f,
  -0.101416f,  -0.105538f, 0.157494f,  -0.265000f, -0.034112f, 0.150214f,
  -0.008629f,  0.033917f,  0.029176f,  -0.027895f, -0.012789f, 0.360946f,
  -9.055856f,  -5.420022f, 1.305300f,  1.037221f,  0.824649f,  -3.390842f,
  0.695224f,   -0.517856f, 1.169544f,  -0.166765f, 1.173215f,  1.267628f,
  -11.278430f, -7.035507f, 0.388088f,  -1.037886f, 0.886147f,  -3.447013f,
  5.775808f,   -0.237123f, 0.373430f,  1.993753f,  2.239124f,  0.528587f,
  -9.795724f,  -6.032860f, 0.952106f,  0.929082f,  0.289468f,  -2.989757f,
  0.983533f,   -0.629013f, 1.351717f,  0.388473f,  1.637240f,  1.576859f,
};

static const float av1_tx_split_nn_bias_8x8_layer0[12] = {
  -5.002550f, -5.289120f, -2.497319f, 6.197292f, 3.130690f, -2.540099f,
  7.496601f,  10.268342f, 12.360687f, 1.044796f, 3.463893f, 1.409374f,
};

static const float av1_tx_split_nn_weights_8x8_layer1[48] = {
  -0.450853f, -0.919031f, -0.896239f, 0.967031f,  0.323851f, 0.005813f,
  0.922491f,  0.567087f,  0.314083f,  0.761493f,  0.572327f, 0.781546f,
  -0.083096f, -0.417982f, -0.300885f, -0.357716f, 0.501658f, 0.512451f,
  0.555781f,  0.296859f,  0.461813f,  -0.146335f, 0.812908f, -0.874717f,
  0.246836f,  -0.000235f, 0.110456f,  0.055466f,  0.140368f, -0.297199f,
  -0.229248f, -0.169994f, 0.173133f,  0.577863f,  0.616053f, 0.715688f,
  0.143336f,  0.129578f,  0.219849f,  -0.014311f, 0.452200f, 0.169101f,
  0.344979f,  0.290693f,  0.964947f,  0.468193f,  0.706292f, 0.487736f,
};

static const float av1_tx_split_nn_bias_8x8_layer1[4] = {
  0.773731f,
  -1.035591f,
  -1.272196f,
  -0.052573f,
};

static const NN_CONFIG av1_tx_split_nnconfig_8x8 = {
  12,  // num_inputs
  4,   // num_outputs
  1,   // num_hidden_layers
  {
      12,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_8x8_layer0,
      av1_tx_split_nn_weights_8x8_layer1,
  },
  {
      av1_tx_split_nn_bias_8x8_layer0,
      av1_tx_split_nn_bias_8x8_layer1,
  }
};
/******************************************************************************/

// Tx split model for 16x16 block.
static const float av1_tx_split_nn_weights_16x16_layer0[288] = {
  0.474789f,   0.467633f,   -0.205558f, 0.162160f,  -0.193350f,  -0.533412f,
  -0.892076f,  0.138487f,   -0.134297f, 0.509919f,  0.258666f,   0.709633f,
  0.005570f,   -0.218537f,  -0.360002f, -0.890066f, 0.100759f,   -0.118849f,
  -0.087320f,  0.233528f,   0.469519f,  -0.052407f, -0.062030f,  0.127983f,
  -14.521076f, -14.824409f, -5.925343f, -4.550285f, -0.104264f,  -6.221783f,
  -7.466872f,  -2.770705f,  -1.125299f, -0.434030f, -5.577441f,  -4.941262f,
  -2.424983f,  -7.316950f,  -6.936694f, -6.859045f, -0.522272f,  -7.948709f,
  -2.732031f,  -1.760950f,  -1.452434f, -2.811649f, -16.385056f, -11.277051f,
  0.103894f,   -0.112287f,  0.138792f,  -0.289157f, 0.023979f,   -0.075042f,
  -0.748017f,  0.176022f,   0.065406f,  -0.092085f, 0.363097f,   0.247442f,
  -0.067787f,  -0.187864f,  -0.010650f, -0.690471f, -0.223198f,  -0.014131f,
  -0.104747f,  -0.103919f,  -0.310102f, -0.061472f, 0.024974f,   0.028510f,
  1.922951f,   -13.038629f, -5.184891f, -3.958634f, -0.024691f,  -5.770776f,
  1.295616f,   1.642872f,   -1.140596f, -0.507206f, 1.080762f,   0.969907f,
  -0.934794f,  -7.020148f,  -5.956906f, 1.264692f,  -0.852806f,  -6.789222f,
  -1.081474f,  -1.818094f,  -1.109554f, -1.112096f, 3.338646f,   2.900530f,
  0.163126f,   -0.201565f,  -0.112479f, -0.113811f, 0.035561f,   -0.118354f,
  -0.673341f,  -0.120831f,  -0.070242f, -0.106012f, 0.299334f,   0.208943f,
  -0.072576f,  0.130355f,   -0.073502f, -0.614139f, -0.052736f,  0.030779f,
  0.076121f,   0.270168f,   0.134819f,  -0.078132f, 0.038411f,   -0.140863f,
  1.698634f,   -13.686110f, -5.239293f, -3.900518f, -0.046580f,  -5.229640f,
  1.161890f,   -1.000117f,  -1.395689f, -0.400401f, 0.729593f,   0.666682f,
  -0.563972f,  -7.389325f,  -6.652141f, 1.141915f,  -0.435921f,  -6.439124f,
  1.439778f,   -1.455298f,  -2.015669f, -0.765921f, 3.111624f,   0.392600f,
  0.041909f,   -0.138870f,  0.163211f,  0.078836f,  0.072037f,   0.208750f,
  -0.707087f,  -0.172304f,  -0.235477f, -0.222708f, 0.412507f,   0.266403f,
  0.081290f,   -0.006198f,  0.130307f,  -0.666511f, -0.011587f,  0.165661f,
  -0.060216f,  -0.308181f,  -0.287677f, 0.123838f,  0.063158f,   0.026804f,
  2.000005f,   -14.237525f, -5.196146f, -4.998881f, 0.001263f,   -5.445941f,
  1.398517f,   -1.249557f,  -1.540626f, 0.080212f,  1.165236f,   1.037267f,
  1.730966f,   -7.011312f,  -5.846158f, 1.350454f,  -0.830604f,  -6.558525f,
  -0.431735f,  -1.262654f,  -1.503832f, 1.679058f,  3.055064f,   3.364932f,
  0.308833f,   -0.282628f,  0.104158f,  0.144368f,  0.060847f,   0.250332f,
  -0.616914f,  0.286310f,   0.331518f,  -0.089389f, 0.352213f,   0.255191f,
  -0.002531f,  0.070201f,   0.092433f,  -0.572650f, 0.178407f,   -0.232221f,
  0.100095f,   -0.135690f,  -0.008503f, 0.101192f,  -0.053127f,  -0.128340f,
  2.747343f,   -14.358504f, -5.874633f, -3.861737f, -0.091565f,  -5.548666f,
  1.450527f,   1.853165f,   -1.246696f, -0.347517f, 1.060727f,   0.977281f,
  1.152183f,   -6.411179f,  -6.136017f, 1.419789f,  -0.841874f,  -7.082089f,
  1.185764f,   -1.920747f,  -1.614745f, 1.166723f,  3.292876f,   0.760933f,
  6.757953f,   -14.438561f, -4.015242f, -4.311859f, -0.266637f,  -6.324111f,
  2.004012f,   -0.314831f,  -1.721621f, -0.326944f, 4.190153f,   3.693672f,
  -0.110710f,  -6.836761f,  -7.726433f, 1.782313f,  -0.430224f,  -7.611656f,
  0.216015f,   -1.827168f,  -2.919386f, 0.271241f,  0.409721f,   3.852640f,
  1.833062f,   -14.412664f, -6.073769f, -4.773946f, -0.189923f,  -5.963335f,
  1.380601f,   -0.012708f,  -1.379821f, 0.238998f,  0.916703f,   0.807589f,
  0.660815f,   -7.945954f,  -6.416434f, 1.083890f,  -0.158921f,  -7.656726f,
  0.316826f,   -1.522116f,  -1.464910f, 0.916378f,  2.944309f,   2.163005f,
};

static const float av1_tx_split_nn_bias_16x16_layer0[24] = {
  6.268684f,  -13.619514f, -3.627919f, -5.156554f, -1.786631f, -6.526987f,
  0.846223f,  5.237819f,   -0.570121f, -0.171747f, 0.633480f,  -2.481526f,
  -5.331006f, -5.822598f,  -5.546602f, -2.247579f, -0.156891f, -6.534738f,
  4.514485f,  -1.323309f,  -1.071433f, 2.873798f,  4.589424f,  3.816363f,
};

static const float av1_tx_split_nn_weights_16x16_layer1[96] = {
  0.446465f,  -0.123193f, 0.304816f,  0.116611f,  -0.397557f, -1.344930f,
  0.930132f,  -0.304186f, 1.021219f,  -2.574242f, 1.119615f,  -0.464554f,
  -0.070013f, -0.921667f, -2.464425f, 0.107023f,  0.210489f,  0.190961f,
  0.182165f,  0.287232f,  0.370201f,  -1.057579f, -0.155697f, 0.026754f,
  0.895846f,  0.291774f,  0.282412f,  0.262008f,  -0.306217f, 0.011065f,
  -0.181248f, -0.242073f, 0.592525f,  -0.183235f, -0.830109f, -0.498763f,
  -0.255777f, 0.081580f,  -0.265661f, 0.177407f,  0.601665f,  0.244513f,
  0.225809f,  0.201387f,  -0.231325f, 0.150312f,  0.179371f,  0.177837f,
  0.102297f,  -0.136649f, -0.168782f, 0.025326f,  -1.016812f, -2.239410f,
  0.351617f,  -0.162177f, 0.130038f,  -1.007639f, 0.754786f,  -0.496034f,
  0.141694f,  0.719664f,  0.772544f,  0.783673f,  -0.192176f, -0.389227f,
  -0.161577f, -0.263991f, -0.276891f, -0.411949f, -0.395276f, 0.691036f,
  0.207945f,  0.402686f,  0.119843f,  0.460084f,  0.153306f,  -0.464401f,
  -0.033482f, -0.137280f, 0.479768f,  0.151004f,  0.369237f,  0.287854f,
  0.041650f,  0.291733f,  0.335202f,  0.007366f,  -0.055260f, -0.452030f,
  0.283905f,  0.123112f,  0.016658f,  0.068217f,  -0.097666f, 0.194833f,
};

static const float av1_tx_split_nn_bias_16x16_layer1[4] = {
  0.703861f,
  -0.402871f,
  -0.776935f,
  -0.159249f,
};

static const NN_CONFIG av1_tx_split_nnconfig_16x16 = {
  12,  // num_inputs
  4,   // num_outputs
  1,   // num_hidden_layers
  {
      24,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_16x16_layer0,
      av1_tx_split_nn_weights_16x16_layer1,
  },
  {
      av1_tx_split_nn_bias_16x16_layer0,
      av1_tx_split_nn_bias_16x16_layer1,
  }
};
/******************************************************************************/

// Tx split model for 32x32 block.
static const float av1_tx_split_nn_weights_32x32_layer0[384] = {
  0.046958f,  -0.487182f,  -0.207276f,  0.122975f,  -0.232357f,  0.423061f,
  0.083695f,  0.209004f,   0.145371f,   0.204790f,  0.060497f,   -0.149358f,
  0.365585f,  -0.089112f,  -0.073255f,  0.030706f,  -1.362476f,  0.054072f,
  -0.173057f, 0.291562f,   -0.099945f,  0.168347f,  0.620501f,   -0.213687f,
  -0.149155f, -0.026749f,  -0.098648f,  0.066940f,  1.149279f,   0.127861f,
  -1.176607f, 0.003148f,   -5.366087f,  -7.283329f, -11.267402f, -0.408448f,
  -9.570044f, -1.536999f,  -0.402654f,  -1.978521f, -0.716269f,  -5.488766f,
  -0.894987f, -6.339402f,  -15.706795f, -0.034319f, -0.551972f,  -2.990029f,
  -5.850733f, -0.458903f,  -0.504074f,  -8.328808f, -1.831571f,  -1.024393f,
  -7.112581f, -0.499243f,  -1.144540f,  -2.283289f, -7.257950f,  -0.226295f,
  -7.514639f, -4.882783f,  -5.486747f,  -0.430405f, 0.077035f,   -2.401762f,
  0.051418f,  -0.077755f,  0.096103f,   0.305131f,  0.253048f,   -0.120158f,
  -0.187295f, -0.067741f,  0.181189f,   0.031300f,  -0.145218f,  0.121743f,
  -0.133812f, 0.302333f,   -0.990706f,  0.013882f,  0.281146f,   1.914602f,
  -0.019104f, -0.249083f,  0.829565f,   -0.175785f, 0.132313f,   -0.112619f,
  -0.345249f, -0.212430f,  0.763366f,   -0.040021f, -0.953698f,  0.211920f,
  -6.773754f, 1.813483f,   1.850887f,   -0.241908f, 0.777898f,   1.923581f,
  -0.722870f, 1.304593f,   -0.989449f,  -4.830653f, 0.684986f,   -6.561400f,
  2.868989f,  0.058735f,   0.032861f,   -2.974526f, 0.940006f,   -1.050485f,
  0.582973f,  2.005292f,   -0.625027f,  -0.869050f, 1.204904f,   0.003256f,
  -0.828437f, -2.222440f,  -5.388565f,  -0.574854f, 1.161660f,   -3.657036f,
  0.995444f,  -0.390634f,  0.086816f,   -2.120298f, -0.007238f,  0.039954f,
  0.149144f,  0.771780f,   -0.277857f,  -0.340927f, -0.163937f,  -0.241664f,
  0.222880f,  0.187714f,   -0.122869f,  -0.154828f, 0.119205f,   -0.031351f,
  -0.916409f, -0.164136f,  0.281230f,   1.632281f,  0.072912f,   -0.028147f,
  0.721958f,  0.059104f,   0.179564f,   0.184564f,  -0.149165f,  0.089537f,
  0.578704f,  0.104578f,   -0.878127f,  0.040060f,  -7.906764f,  1.696033f,
  1.897944f,  -0.620451f,  0.899170f,   -3.258062f, -0.658620f,  -1.274688f,
  -0.595886f, -5.746620f,  -0.045377f,  -6.150619f, 2.662146f,   -0.149803f,
  -0.606079f, -2.589373f,  1.088823f,   -0.605775f, -0.052084f,  2.262103f,
  1.045070f,  -0.602276f,  1.475711f,   0.121007f,  -0.549108f,  -2.341201f,
  -6.369531f, -0.467140f,  1.523135f,   -4.514831f, 1.063686f,   -0.600889f,
  -0.030633f, 2.324924f,   0.006825f,   0.019342f,  0.001229f,   -0.079619f,
  0.061515f,  -0.214943f,  0.158505f,   0.350269f,  0.171238f,   -0.115931f,
  -0.199798f, 0.060098f,   0.142894f,   -0.075136f, -0.844648f,  0.021943f,
  0.217210f,  -1.943381f,  0.083168f,   0.160790f,  0.744566f,   0.243001f,
  0.189233f,  -0.059563f,  -0.085992f,  0.116181f,  0.657602f,   -0.035946f,
  -0.825389f, -0.216224f,  -7.106647f,  0.398453f,  2.003256f,   -0.611031f,
  1.015970f,  0.242447f,   -0.708766f,  -1.068971f, 0.633790f,   -6.257261f,
  0.429580f,  -5.340114f,  2.736088f,   -0.298433f, -0.004105f,  -2.270438f,
  0.994900f,  -0.937148f,  0.369547f,   0.543087f,  -0.674773f,  -0.873675f,
  1.270663f,  -0.202287f,  1.022378f,   -2.501356f, -5.619052f,  -0.355971f,
  1.254564f,  -5.113996f,  0.988437f,   -0.291573f, 0.032199f,   2.221923f,
  -0.037061f, -0.116490f,  0.062776f,   0.214843f,  -0.141882f,  0.118676f,
  -0.093076f, -0.351724f,  0.133107f,   -0.028494f, -0.030131f,  0.062847f,
  -0.060066f, -0.206935f,  -0.792724f,  0.079483f,  0.207122f,   -2.070919f,
  0.009768f,  -0.063885f,  0.656649f,   0.082179f,  0.176691f,   -0.029725f,
  0.387008f,  -0.054883f,  0.578538f,   -0.302890f, -0.771945f,  -0.044829f,
  -7.954852f, 0.691611f,   1.982518f,   0.053139f,  1.039066f,   -2.520956f,
  -0.688823f, 1.398113f,   0.585718f,   -5.623590f, -0.287870f,  -5.661140f,
  3.004946f,  -0.266782f,  0.034117f,   -1.801512f, 1.110053f,   -0.742834f,
  -0.299888f, 0.747327f,   1.072419f,   -1.223879f, 1.498942f,   -0.040088f,
  0.614769f,  -2.027053f,  -6.822720f,  -0.086497f, 1.561462f,   -4.991680f,
  1.024338f,  -0.125531f,  -10.264043f, -4.154160f, 6.217939f,   -0.850219f,
  8.768229f,  -10.248763f, -1.587616f,  0.143881f,  -0.037559f,  -5.744375f,
  2.609962f,  -6.611407f,  -1.469895f,  -0.163888f, -0.269790f,  -2.814628f,
  -2.000585f, -1.688349f,  2.355441f,   -2.240026f, 0.163853f,   -2.334688f,
  -1.834162f, -0.365103f,  0.527970f,   -3.017580f, -6.819779f,  -0.330589f,
  -2.063322f, -4.371924f,  -1.584512f,  -0.205044f, -8.084334f,  1.429348f,
  1.933336f,  -0.030412f,  0.988867f,   -1.443481f, -0.495259f,  0.205299f,
  0.042660f,  -5.434615f,  0.274066f,   -5.954418f, 3.039747f,   -0.246382f,
  0.038041f,  -2.720037f,  1.391626f,   -0.377117f, 0.089482f,   1.267496f,
  0.324743f,  -0.826808f,  1.479658f,   -0.222381f, -0.064690f,  -1.920969f,
  -6.289336f, -0.201474f,  1.657285f,   -4.738920f, 1.219278f,   -0.298556f,
};

static const float av1_tx_split_nn_bias_32x32_layer0[32] = {
  -5.231327f, 2.679321f,  3.204740f,  -0.244696f, 4.470131f,  7.003172f,
  0.035981f,  2.466602f,  -5.151036f, -6.529060f, -0.601036f, -6.743008f,
  4.306645f,  -0.191898f, -0.221784f, -2.031725f, 0.638000f,  -0.471934f,
  -4.955877f, 2.894309f,  1.367930f,  -0.564191f, -1.569413f, -0.183420f,
  1.607205f,  -1.802805f, -7.581415f, -0.213269f, 1.058535f,  -7.780657f,
  -2.460422f, -0.069723f,
};

static const float av1_tx_split_nn_weights_32x32_layer1[128] = {
  -0.669995f, -2.482290f, -0.186782f, 0.702236f,  0.199582f,  0.141325f,
  0.406655f,  0.062232f,  0.261268f,  0.005268f,  0.115091f,  0.127218f,
  0.261082f,  0.266604f,  0.294823f,  0.142072f,  0.259178f,  0.044700f,
  0.195831f,  0.122425f,  0.812225f,  0.302140f,  0.027128f,  0.361781f,
  0.043124f,  -0.198312f, -0.196001f, -0.007293f, 0.009421f,  0.092909f,
  -0.050655f, -0.135316f, 0.631881f,  0.531723f,  0.321812f,  0.754984f,
  -0.155297f, 0.067084f,  0.406300f,  0.945714f,  0.480478f,  0.184205f,
  0.147482f,  0.045271f,  0.103841f,  -0.605359f, -0.016932f, 0.065983f,
  -0.589512f, -0.081462f, 0.043673f,  0.341414f,  0.032772f,  -0.077084f,
  -0.046222f, -0.020094f, 0.140097f,  0.345794f,  0.062080f,  0.384671f,
  0.249130f,  0.569187f,  0.145603f,  0.557647f,  0.909975f,  0.224928f,
  0.341536f,  0.058189f,  0.591609f,  -0.132234f, 0.232678f,  -0.006831f,
  0.047217f,  0.335522f,  0.347247f,  0.427795f,  0.309486f,  0.237179f,
  0.512534f,  0.092909f,  0.121879f,  0.168937f,  -0.355713f, 0.172236f,
  0.951090f,  -0.285442f, -0.231386f, -0.437254f, 0.046657f,  0.629805f,
  0.471497f,  0.749959f,  0.202508f,  -0.128414f, 0.184357f,  0.027133f,
  0.262237f,  0.416224f,  0.514781f,  0.123746f,  -0.519419f, 0.171223f,
  -0.234954f, 0.531523f,  1.003354f,  -0.035347f, 0.639394f,  1.073954f,
  -0.069867f, 0.032503f,  0.109259f,  -0.003018f, 1.093342f,  0.509677f,
  0.717268f,  0.391205f,  0.826398f,  0.018090f,  0.713313f,  0.542313f,
  -0.273572f, 0.444055f,  0.371339f,  0.564918f,  0.052463f,  0.126713f,
  0.042319f,  0.130111f,
};

static const float av1_tx_split_nn_bias_32x32_layer1[4] = {
  -0.073753f,
  0.879884f,
  -0.920125f,
  0.866336f,
};

static const NN_CONFIG av1_tx_split_nnconfig_32x32 = {
  12,  // num_inputs
  4,   // num_outputs
  1,   // num_hidden_layers
  {
      32,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_32x32_layer0,
      av1_tx_split_nn_weights_32x32_layer1,
  },
  {
      av1_tx_split_nn_bias_32x32_layer0,
      av1_tx_split_nn_bias_32x32_layer1,
  }
};
/******************************************************************************/

// Tx split model for 64x64 block.
static const float av1_tx_split_nn_weights_64x64_layer0[384] = {
  -0.163357f,  0.139774f,  -0.185753f, -0.194900f, 0.113803f,  -0.049728f,
  -1.141747f,  -0.247310f, 0.191688f,  0.000684f,  -0.110437f, 0.265803f,
  -0.333200f,  0.122450f,  0.460385f,  0.134246f,  -0.157015f, 0.171543f,
  -0.068857f,  -0.142538f, 0.055898f,  0.074063f,  0.103949f,  0.115054f,
  -0.131891f,  -0.351742f, -1.175047f, 0.101650f,  0.138633f,  0.175611f,
  0.052515f,   0.144388f,  -3.675124f, -0.173294f, -2.783707f, -0.402233f,
  -0.852360f,  -0.691844f, -2.876175f, -7.312618f, -2.273328f, -0.932840f,
  -2.776503f,  -0.305702f, -7.159744f, -1.192084f, -3.924724f, -7.901151f,
  -0.365000f,  -0.501630f, -0.270283f, -0.346485f, -3.149643f, -0.458924f,
  -3.204340f,  -1.775877f, -1.829706f, -3.567985f, -2.256177f, -3.290000f,
  -25.757950f, -5.931861f, -0.544950f, -2.090314f, -0.010694f, -0.042869f,
  0.062737f,   0.063353f,  0.212052f,  -0.042426f, -2.768570f, -0.088384f,
  -0.213944f,  1.947254f,  0.110610f,  0.960840f,  0.136319f,  -1.979805f,
  -0.499830f,  0.167963f,  -0.315831f, 0.110407f,  0.236026f,  0.082720f,
  -0.025561f,  -0.035533f, 0.200685f,  0.249893f,  0.105281f,  0.235776f,
  3.164895f,   0.054719f,  0.052670f,  -0.226794f, -0.059907f, -0.040422f,
  -3.424864f,  -0.230189f, -3.187440f, -0.034457f, -0.277574f, -0.484354f,
  0.422989f,   1.338834f,  -1.201274f, 0.200741f,  1.782817f,  -0.809361f,
  -6.339804f,  0.392111f,  -4.208724f, -0.401390f, -0.455925f, -0.491104f,
  -0.273227f,  -0.712468f, -3.568536f, 0.021360f,  -3.159549f, -1.776143f,
  0.525809f,   -2.435971f, 0.498909f,  -3.757555f, 2.398833f,  -5.503863f,
  -0.002143f,  -1.633448f, -0.386582f, -0.086436f, 0.112283f,  -0.001255f,
  0.040870f,   -0.051164f, -2.440881f, -0.178699f, 0.204503f,  -1.659598f,
  0.057098f,   0.578466f,  0.186497f,  1.923075f,  -0.019453f, 0.007772f,
  0.374330f,   -0.053986f, 0.287643f,  0.024397f,  -0.093080f, -0.098158f,
  -0.017092f,  0.621203f,  -0.020861f, 0.044307f,  3.185825f,  0.068676f,
  0.264453f,   0.278450f,  -0.341807f, 0.044793f,  -2.966478f, -0.278219f,
  -2.671680f,  0.020423f,  -0.501904f, -0.544847f, 0.562525f,  1.247239f,
  -1.505224f,  0.215754f,  -1.084059f, -0.244475f, -7.489781f, 0.277892f,
  -3.473467f,  0.970773f,  -0.128657f, -0.307661f, -0.122801f, 0.697115f,
  -3.287625f,  -0.267082f, -3.281390f, -1.710432f, -0.435975f, -1.948191f,
  0.399222f,   -2.928223f, 2.477332f,  -4.849609f, -0.251135f, -2.201354f,
  0.702988f,   -0.057887f, -0.129951f, 0.016725f,  -0.070176f, 0.047533f,
  3.023566f,   0.182436f,  -0.132877f, 2.105023f,  0.203903f,  0.489572f,
  0.150025f,   -2.008847f, 0.222033f,  -0.052571f, 0.278572f,  -0.205633f,
  0.069151f,   0.057010f,  0.424874f,  0.074297f,  -0.027829f, -0.434513f,
  0.054639f,   -0.096778f, -2.965554f, -0.039963f, -0.027391f, -0.067237f,
  0.143189f,   -0.108002f, -4.015142f, 0.293854f,  -2.769948f, -0.039893f,
  -0.372629f,  0.717550f,  0.300946f,  1.220136f,  -2.434674f, 0.220532f,
  -1.060258f,  0.034633f,  -5.501379f, 0.340665f,  -3.379550f, 3.481403f,
  -0.373476f,  -0.472257f, -0.099358f, -0.706129f, -2.764891f, -0.366821f,
  -2.794182f,  -1.452025f, 0.235963f,  -3.211427f, 0.130441f,  -3.309384f,
  3.610559f,   -5.374982f, -0.225922f, -2.132611f, -0.073099f, -0.067738f,
  0.029017f,   0.105513f,  -0.313987f, 0.002827f,  2.680857f,  -0.149454f,
  -0.141401f,  -1.825205f, 0.099800f,  0.351381f,  -0.402679f, 1.928909f,
  0.060733f,   -0.112215f, -0.183725f, -0.010875f, 0.226246f,  0.062391f,
  -0.249618f,  -0.015117f, 0.037222f,  -0.565675f, 0.000148f,  0.029752f,
  -2.745430f,  -0.192095f, 0.498102f,  -0.067898f, 0.204089f,  -0.016899f,
  -1.969419f,  0.273978f,  -2.700315f, 0.005030f,  -0.712079f, 0.733197f,
  0.629894f,   1.242437f,  -2.691414f, 0.078747f,  1.452238f,  0.181960f,
  -6.409108f,  0.272419f,  -3.843341f, 0.092978f,  -0.517390f, -0.180997f,
  0.153923f,   0.642429f,  -3.701921f, -0.269350f, -3.458511f, -1.644308f,
  -0.297359f,  -3.979359f, 0.505582f,  -2.235056f, 2.745480f,  -4.808592f,
  -0.333131f,  -2.977683f, -4.243013f, 0.128521f,  -2.588990f, -0.311894f,
  -1.437891f,  0.246628f,  -7.355105f, -7.660661f, -1.676779f, -7.265692f,
  0.725816f,   -2.792214f, -7.387144f, -6.766887f, -4.250912f, 2.266179f,
  -0.767393f,  -0.600641f, -1.860558f, -0.292079f, -4.011251f, -0.219491f,
  -4.496087f,  -1.992320f, -0.261304f, -3.972974f, -8.741853f, -4.056727f,
  -0.745925f,  -6.455836f, -0.113011f, -1.849653f, -3.427984f, 0.209662f,
  -2.739849f,  0.020829f,  -0.203196f, 0.134941f,  0.379660f,  0.913978f,
  -1.675223f,  0.145357f,  0.460668f,  0.066695f,  -6.882690f, -0.110051f,
  -3.997168f,  1.699738f,  -0.802448f, -0.317460f, -0.163015f, -0.043201f,
  -3.741879f,  -0.221445f, -3.429426f, -1.949018f, 0.201020f,  -2.820048f,
  -0.013069f,  -2.876819f, 3.244849f,  -5.533144f, -0.375890f, -2.000652f,
};

static const float av1_tx_split_nn_bias_64x64_layer0[32] = {
  -4.419834f, 2.915720f,  -3.658958f, -0.508265f, -0.394101f, 0.146479f,
  2.445936f,  4.507841f,  -2.565729f, 2.504941f,  2.205232f,  3.472952f,
  -8.748951f, 2.534985f,  -5.137657f, 2.025067f,  -0.336747f, -0.216172f,
  4.534413f,  0.796407f,  -2.508404f, -0.002234f, -3.955281f, -0.468072f,
  -0.375857f, -3.551570f, 2.445183f,  -3.959559f, 8.609996f,  -5.366344f,
  -0.245499f, -2.324293f,
};

static const float av1_tx_split_nn_weights_64x64_layer1[128] = {
  -0.020038f, -0.010603f, 0.130804f,  -0.185654f, 0.156962f,  -0.040028f,
  -0.177954f, 0.160444f,  0.594891f,  -0.570108f, -1.109697f, 0.916323f,
  -0.237840f, 0.076464f,  -0.207161f, -0.043028f, 0.002027f,  0.088178f,
  0.408456f,  -0.107124f, 0.060678f,  0.256821f,  0.400221f,  -0.003960f,
  0.205242f,  0.079538f,  0.292750f,  -0.027633f, 0.129499f,  -0.486416f,
  -0.022472f, -0.136361f, 0.034178f,  0.813827f,  0.124522f,  0.538672f,
  0.142630f,  0.035994f,  -0.044722f, 0.269185f,  0.259839f,  0.376664f,
  0.055069f,  0.212727f,  0.236017f,  -0.119546f, 0.173908f,  0.012940f,
  0.487967f,  -0.441302f, -0.243817f, -0.869261f, 0.022772f,  -0.071352f,
  -0.246331f, 0.150381f,  0.419036f,  -1.283917f, -1.346872f, 0.948908f,
  0.122684f,  0.228288f,  0.223141f,  -0.062880f, 0.267273f,  -0.469009f,
  0.300714f,  -0.035537f, -0.151092f, 0.033243f,  -0.083954f, 0.062204f,
  -0.328241f, 0.136157f,  0.044101f,  -0.099044f, 0.301310f,  0.435132f,
  0.023226f,  0.440868f,  -0.820726f, -0.540511f, -0.515328f, -0.352225f,
  0.298817f,  0.343317f,  0.227181f,  0.505699f,  -0.146168f, -0.813934f,
  -0.418463f, 0.329126f,  -0.243686f, -0.274607f, 1.141390f,  -0.519253f,
  0.056584f,  -0.405958f, -0.146720f, 0.367860f,  0.521038f,  0.778372f,
  0.393421f,  0.163980f,  0.014830f,  -0.126190f, 0.146492f,  -0.137432f,
  0.959669f,  -0.241196f, -0.377633f, 0.711132f,  0.972224f,  0.152722f,
  -0.232369f, 0.491887f,  0.260622f,  0.161239f,  -2.094605f, 1.354902f,
  -0.110255f, 0.354291f,  0.061279f,  -0.292223f, 0.595051f,  -1.166008f,
  -0.028361f, 0.537691f,
};

static const float av1_tx_split_nn_bias_64x64_layer1[4] = {
  1.008097f,
  -0.498867f,
  -0.557468f,
  -0.659801f,
};

static const NN_CONFIG av1_tx_split_nnconfig_64x64 = {
  12,  // num_inputs
  4,   // num_outputs
  1,   // num_hidden_layers
  {
      32,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_64x64_layer0,
      av1_tx_split_nn_weights_64x64_layer1,
  },
  {
      av1_tx_split_nn_bias_64x64_layer0,
      av1_tx_split_nn_bias_64x64_layer1,
  }
};
/******************************************************************************/

// Tx split model for 8x16 block.
static const float av1_tx_split_nn_weights_8x16_layer0[512] = {
  0.008706f,   0.148967f,   -0.160689f,  -0.282403f,  -0.110820f,  0.029307f,
  -0.048682f,  0.163936f,   0.112060f,   0.115985f,   0.186502f,   0.259759f,
  0.325057f,   -1.053798f,  0.027529f,   0.173346f,   -0.247865f,  -0.050586f,
  -0.182748f,  0.056384f,   0.203333f,   0.021658f,   0.015164f,   -0.087442f,
  -0.124108f,  -0.037750f,  -0.205561f,  0.233535f,   -0.055001f,  0.121733f,
  0.062396f,   -0.068495f,  0.112846f,   -0.029663f,  0.065050f,   0.094911f,
  -0.033090f,  -0.181993f,  -0.081860f,  0.027762f,   0.449803f,   -0.153997f,
  0.177085f,   0.176908f,   0.497651f,   -0.301789f,  -0.082769f,  -0.083364f,
  0.220406f,   0.437270f,   0.140604f,   0.191820f,   0.070028f,   -0.021215f,
  -0.025093f,  0.134911f,   0.028828f,   -0.087983f,  -0.899327f,  -0.090872f,
  0.338996f,   0.163529f,   0.085760f,   -0.045752f,  -0.556936f,  -3.024941f,
  -1.001161f,  -2.623173f,  -0.268849f,  -3.732467f,  -7.156731f,  -3.927835f,
  -17.371922f, -7.581237f,  -4.620238f,  -5.752151f,  -9.585605f,  -6.089787f,
  -3.459236f,  -6.519960f,  -0.303135f,  -11.514290f, -3.801391f,  -5.544197f,
  -11.065317f, -5.950223f,  -5.986601f,  -0.506990f,  -5.574658f,  -8.249856f,
  -4.864985f,  0.062282f,   -5.237353f,  -0.801951f,  -9.312742f,  -4.992184f,
  -4.200308f,  -6.315866f,  -0.364066f,  -7.575521f,  -7.695835f,  -6.233013f,
  -5.464857f,  -1.756686f,  -6.676375f,  -1.464777f,  -11.523097f, -3.605280f,
  -11.380379f, -0.144299f,  -5.191412f,  -3.281153f,  0.026833f,   -4.925627f,
  -5.647656f,  -1.249171f,  -0.229417f,  -2.148773f,  -4.976895f,  -1.549322f,
  -6.933633f,  -7.438485f,  -5.362781f,  -0.423710f,  -7.270756f,  -3.450208f,
  -0.460404f,  -7.933499f,  -0.060796f,  -0.381301f,  0.188915f,   -0.482814f,
  0.218080f,   0.052056f,   -0.258212f,  -0.076196f,  -0.125439f,  -0.054571f,
  -0.102671f,  -0.165090f,  -0.158683f,  -0.850112f,  0.012470f,   0.161666f,
  0.180518f,   0.199895f,   0.016670f,   -0.028792f,  0.013969f,   -0.026456f,
  -0.038689f,  0.183116f,   -0.083196f,  -0.055132f,  0.022083f,   -0.094103f,
  0.000137f,   -0.073457f,  0.001824f,   -0.100206f,  0.289873f,   0.025924f,
  -0.001475f,  -0.229439f,  -0.227257f,  -0.246289f,  0.087291f,   0.061299f,
  0.485200f,   0.190082f,   0.154234f,   -0.060103f,  -0.015066f,  0.121548f,
  -0.231034f,  0.127247f,   -0.106495f,  0.136039f,   -0.093293f,  0.015647f,
  -0.070919f,  -0.044288f,  0.009939f,   0.120722f,   0.121121f,   0.086777f,
  -0.709010f,  0.063164f,   0.584601f,   -0.083182f,  -0.201007f,  -0.241184f,
  -0.252414f,  -2.070982f,  -0.926645f,  0.632045f,   0.149715f,   -2.850108f,
  -6.773580f,  -3.519566f,  4.440470f,   -6.963968f,  -4.038210f,  -6.239212f,
  3.184038f,   1.641755f,   -0.237420f,  3.321357f,   -0.460185f,  -10.510607f,
  -2.835936f,  -5.615609f,  -9.581537f,  1.393264f,   -4.983533f,  -0.450568f,
  -5.318336f,  -7.759728f,  -4.929632f,  -0.071730f,  -4.965301f,  -0.466271f,
  -8.984825f,  -0.390878f,  -4.111952f,  1.482303f,   -0.464770f,  -6.834233f,
  1.589730f,   -6.264070f,  -4.841766f,  -1.670637f,  1.606385f,   -1.118513f,
  -10.789104f, -3.820304f,  0.772173f,   -0.195932f,  -5.323112f,  -2.928015f,
  0.010789f,   1.154397f,   -3.841392f,  -1.508927f,  -0.461057f,  -2.023488f,
  -5.492938f,  -1.409095f,  -6.966518f,  2.128327f,   1.562568f,   -0.161474f,
  1.768794f,   -3.530464f,  -0.204127f,  -7.026234f,  0.051494f,   0.262418f,
  -0.029063f,  -0.506412f,  -0.110378f,  -0.177869f,  0.100419f,   -0.074692f,
  -0.007550f,  -0.071437f,  -0.246025f,  -0.109797f,  -0.167092f,  -0.820678f,
  0.051861f,   -0.201440f,  0.065781f,   0.030903f,   0.213251f,   0.054118f,
  0.008978f,   -0.008579f,  -0.099213f,  -0.108190f,  -0.004098f,  -0.141620f,
  0.044393f,   -0.138910f,  -0.074729f,  -0.045122f,  0.141622f,   0.128570f,
  -0.519756f,  0.012805f,   -0.069690f,  0.146102f,   -0.217476f,  0.165560f,
  -0.034860f,  -0.080407f,  0.515260f,   -0.036135f,  0.198025f,   -0.124580f,
  0.839902f,   0.172398f,   0.332561f,   -0.024468f,  -0.103379f,  0.251121f,
  -0.017841f,  -0.208236f,  0.000859f,   0.009402f,   0.030748f,   -0.268966f,
  -0.054319f,  0.077892f,   -0.673224f,  0.022201f,   0.555678f,   -0.134388f,
  0.107994f,   0.044791f,   -0.447695f,  -3.612972f,  -0.672773f,  0.788562f,
  -0.277318f,  -4.497281f,  -6.096453f,  -4.766270f,  4.416097f,   -7.697475f,
  -5.842667f,  -5.007094f,  2.028800f,   1.792036f,   0.994261f,   0.087397f,
  -0.098392f,  -12.231838f, -4.620152f,  -5.372140f,  -16.848415f, 1.263519f,
  -5.587913f,  -0.336783f,  -6.070452f,  -7.513999f,  -4.324697f,  -0.156012f,
  -5.057558f,  -0.801907f,  -10.554936f, 2.854021f,   -3.001579f,  1.924380f,
  -0.125017f,  -7.950968f,  1.545191f,   -6.280745f,  -5.276745f,  -1.422817f,
  1.856837f,   -1.601185f,  -11.460361f, -3.345600f,  -1.855278f,  -0.429886f,
  -4.809439f,  -3.441611f,  0.060694f,   0.285455f,   -0.795384f,  -0.922761f,
  -0.525281f,  -2.205379f,  -4.833350f,  -1.348174f,  -6.360427f,  2.359873f,
  1.674077f,   -0.462879f,  2.063807f,   -3.040541f,  -0.074346f,  -7.930112f,
  -0.235809f,  -3.288008f,  -0.585351f,  -0.538012f,  -0.085064f,  -2.395069f,
  -5.216609f,  -3.394021f,  0.702762f,   -6.006907f,  -4.094669f,  -6.953358f,
  0.406964f,   -0.233930f,  -0.237646f,  1.026158f,   -0.350515f,  -12.901628f,
  -3.659032f,  -5.382937f,  -13.432230f, 4.848146f,   -6.728113f,  -0.285737f,
  -4.503033f,  -8.002105f,  -4.534365f,  -0.722114f,  -4.677269f,  -0.385441f,
  -7.742633f,  0.738723f,   -4.134079f,  0.620666f,   -0.013993f,  -6.200027f,
  1.414102f,   -6.552051f,  -4.462392f,  -1.210085f,  -0.038331f,  -0.656697f,
  -9.814882f,  -2.113951f,  -3.028183f,  -0.575831f,  -5.606905f,  -3.073105f,
  0.012709f,   -0.839956f,  -3.780185f,  -1.795141f,  -0.373978f,  -2.740093f,
  -5.322299f,  -1.246722f,  -6.466084f,  1.420467f,   0.036744f,   -0.849337f,
  0.054509f,   -2.865515f,  -0.499508f,  -7.278360f,  -0.442639f,  -2.685421f,
  -0.965477f,  0.767120f,   -0.428495f,  -3.705269f,  -6.913683f,  -3.962320f,
  4.396061f,   -7.145083f,  -5.163342f,  -6.011438f,  2.415459f,   2.027718f,
  0.691211f,   2.138062f,   -0.108030f,  -11.065655f, -3.602587f,  -5.371963f,
  -12.110271f, 1.158038f,   -5.980221f,  -0.061210f,  -5.617925f,  -8.064033f,
  -4.866569f,  0.061719f,   -5.109336f,  -0.715507f,  -9.526828f,  1.498451f,
  -3.692022f,  1.557368f,   -0.242436f,  -7.464099f,  1.435612f,   -6.222495f,
  -5.568766f,  -1.405574f,  1.690963f,   -1.527159f,  -11.328235f, -3.713979f,
  -2.254317f,  -0.440525f,  -5.185599f,  -3.587081f,  -0.243675f,  0.877766f,
  -2.328087f,  -1.126118f,  -0.010644f,  -1.847149f,  -5.138073f,  -1.307305f,
  -6.529606f,  2.345231f,   1.672828f,   -0.361974f,  1.869782f,   -3.808848f,
  -0.074687f,  -7.965084f,
};

static const float av1_tx_split_nn_bias_8x16_layer0[64] = {
  -0.222468f,  -2.452683f, -0.657045f, 3.035878f,  -0.155896f, -3.676363f,
  -5.939752f,  -3.987022f, 5.339483f,  -5.605974f, -3.459944f, -5.636736f,
  6.505204f,   1.255344f,  6.474634f,  2.681355f,  -0.210658f, -3.587264f,
  -4.336063f,  -4.520172f, -5.686054f, 4.422442f,  -5.667303f, -0.269528f,
  -6.381737f,  -7.783754f, -4.224124f, -0.312825f, -4.652617f, -0.978672f,
  -11.272622f, 3.655785f,  -4.799411f, 6.245343f,  -0.255997f, -6.708319f,
  6.440729f,   -8.199196f, -6.363768f, -1.922003f, 0.271982f,  -0.753267f,
  -13.484363f, -1.560596f, 11.751105f, -0.299819f, -5.019947f, -3.862802f,
  -2.549896f,  6.316492f,  -9.447174f, -1.105249f, -0.092660f, -2.201824f,
  -5.874821f,  -0.856331f, -4.235859f, 3.928864f,  -1.328228f, -0.122227f,
  3.796253f,   -3.168672f, -0.173636f, -7.891180f,
};

static const float av1_tx_split_nn_weights_8x16_layer1[256] = {
  0.232165f,  -0.148270f, 0.076505f,  -0.026096f, 0.220465f,  -0.008030f,
  -1.404613f, 0.961791f,  -0.084607f, -0.005044f, -0.026219f, 0.899865f,
  0.086278f,  0.127242f,  0.394722f,  0.413462f,  0.120820f,  -0.114591f,
  0.006122f,  0.093655f,  -0.086652f, -0.049754f, 0.143003f,  -0.105466f,
  -0.436279f, -1.326538f, 1.101524f,  -1.584017f, 0.575400f,  0.107406f,
  -1.195371f, 0.666809f,  -0.177412f, -1.309339f, 0.762426f,  0.659489f,
  -0.268611f, -1.190394f, -0.415255f, -1.605964f, 1.398609f,  -0.100569f,
  -1.316340f, 0.762530f,  0.188156f,  -0.505773f, 1.286807f,  -1.028432f,
  -0.369644f, -0.241235f, 0.171357f,  0.229580f,  0.952196f,  0.441636f,
  0.194012f,  0.117609f,  -0.196259f, -0.063454f, 0.065243f,  0.274509f,
  -0.026203f, 0.311430f,  0.095200f,  0.107787f,  0.059703f,  0.486103f,
  -0.019000f, 0.133062f,  0.510605f,  -3.373556f, -1.316510f, 0.387027f,
  0.397732f,  -1.103108f, 0.294491f,  0.513056f,  -0.315377f, -1.337730f,
  0.293980f,  -0.134504f, -0.297537f, -1.090894f, -0.464240f, -1.217318f,
  0.228701f,  -0.164921f, 0.074804f,  0.080670f,  0.205897f,  -1.550563f,
  -1.087492f, -0.231459f, 0.028634f,  -0.082608f, 0.250944f,  -0.027628f,
  0.345004f,  -1.078753f, 0.572918f,  -0.273691f, 0.027645f,  -1.107030f,
  0.402571f,  -0.040128f, 0.422076f,  -0.991618f, -0.353464f, -0.075326f,
  -0.172541f, -0.163905f, -0.142526f, -0.143768f, 0.817721f,  -2.660967f,
  -0.311349f, -0.060757f, 0.436450f,  -0.147997f, -0.398959f, 0.032816f,
  0.057609f,  -0.106641f, -0.734719f, 0.903315f,  -0.268058f, 0.007365f,
  -0.105462f, -0.082394f, 0.751295f,  -0.385769f, -0.966343f, 0.327253f,
  0.302656f,  -0.022628f, 0.184348f,  -0.221701f, -0.065639f, -0.069888f,
  -0.215460f, 0.106734f,  -0.193433f, -0.830946f, -0.082972f, 0.177517f,
  0.489272f,  0.474720f,  0.089199f,  -0.040948f, -0.087076f, -1.159800f,
  -0.803069f, -0.838416f, -0.278506f, -0.772316f, 0.251069f,  -0.356508f,
  -0.474882f, -0.042714f, 0.290818f,  0.272565f,  0.773841f,  0.986032f,
  1.022310f,  1.235305f,  -0.029465f, -0.339171f, 0.329815f,  -0.318613f,
  -0.284057f, -2.450262f, -0.484857f, -1.857594f, 0.132151f,  -0.408417f,
  -0.690755f, 0.502435f,  5.685473f,  5.666665f,  5.278402f,  5.515622f,
  0.221568f,  0.529491f,  0.415177f,  0.281052f,  0.699595f,  -2.021808f,
  -0.053581f, -0.411313f, 0.387025f,  -1.076157f, -0.530352f, -0.081383f,
  0.051579f,  -0.077253f, 0.075905f,  0.065855f,  0.326726f,  0.364698f,
  -0.058359f, 0.009426f,  -0.230792f, -0.779588f, -1.277078f, 1.493782f,
  0.555069f,  -0.394652f, -0.197219f, 0.272887f,  0.126782f,  0.012021f,
  0.368348f,  0.162288f,  0.094465f,  -0.247114f, 0.075685f,  0.078707f,
  -0.202324f, -1.028201f, -0.700566f, 1.037483f,  0.116701f,  -1.263533f,
  -0.117863f, 0.156398f,  -0.338131f, -1.608200f, -0.708837f, -0.197160f,
  0.125323f,  -0.154637f, -0.198303f, 0.226893f,  -0.151028f, 0.251845f,
  0.475694f,  0.497341f,  0.264677f,  -0.162636f, -0.084407f, -0.239908f,
  0.838968f,  0.459218f,  0.601890f,  0.432977f,  0.369339f,  -0.807639f,
  -0.107838f, 0.261443f,  0.031554f,  0.313928f,  -0.021427f, 0.036705f,
  1.153886f,  -0.675493f, -0.051942f, -0.618454f,
};

static const float av1_tx_split_nn_bias_8x16_layer1[4] = {
  1.059091f,
  -1.432711f,
  -0.467326f,
  -0.230041f,
};

static const NN_CONFIG av1_tx_split_nnconfig_8x16 = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      64,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_8x16_layer0,
      av1_tx_split_nn_weights_8x16_layer1,
  },
  {
      av1_tx_split_nn_bias_8x16_layer0,
      av1_tx_split_nn_bias_8x16_layer1,
  }
};
/******************************************************************************/

// Tx split model for 16x32 block.
static const float av1_tx_split_nn_weights_16x32_layer0[256] = {
  0.214440f,   -0.086757f,  -0.003167f,  -0.837302f,  -0.160980f,  0.113661f,
  -0.089368f,  -0.642192f,  -0.040712f,  -0.142937f,  0.131576f,   -0.911850f,
  0.053469f,   -0.202959f,  -0.029837f,  0.341125f,   -0.179237f,  0.090060f,
  0.054456f,   -0.278953f,  -0.040720f,  -0.181924f,  0.409384f,   -0.038371f,
  0.034102f,   -0.678194f,  0.482541f,   0.284239f,   -1.300912f,  0.023720f,
  -0.156172f,  -0.055062f,  -2.624224f,  -5.430269f,  -11.504310f, -0.615325f,
  -0.494802f,  -8.710828f,  -0.523554f,  0.181713f,   -3.521369f,  0.003528f,
  -0.162640f,  -1.447468f,  -1.823844f,  -4.117291f,  -14.521627f, -0.898752f,
  -7.939065f,  -12.516768f, -0.228721f,  0.018982f,   -5.583223f,  -0.030730f,
  -7.984490f,  -0.121411f,  -1.990650f,  -0.129253f,  -3.391106f,  -4.998734f,
  -1.796882f,  -12.288423f, -6.603404f,  -8.433517f,  -0.338900f,  0.065749f,
  0.129163f,   -0.755507f,  0.152921f,   -0.155527f,  0.295235f,   -0.923349f,
  -0.071625f,  0.289075f,   -0.279990f,  -0.727448f,  0.122631f,   -0.260299f,
  0.047692f,   -0.128472f,  0.167796f,   -0.183864f,  -0.012991f,  0.014154f,
  0.079660f,   -0.009997f,  -0.213097f,  -0.112143f,  -0.071335f,  -0.834672f,
  0.320335f,   -0.136739f,  -1.195615f,  -0.222185f,  0.321312f,   -0.263210f,
  -1.827886f,  -5.239021f,  -10.545290f, -0.151341f,  -0.045088f,  0.217829f,
  -0.446797f,  0.051533f,   -2.628632f,  -0.384310f,  -0.384198f,  0.386875f,
  -1.008449f,  -4.170134f,  4.430185f,   -1.273449f,  -7.640859f,  2.246397f,
  -0.049796f,  0.093279f,   -4.370115f,  -0.192468f,  1.973601f,   0.003403f,
  -1.412509f,  0.239565f,   0.763412f,   1.093936f,   0.487039f,   -13.722408f,
  -7.155999f,  -8.410864f,  0.113642f,   -0.095919f,  -0.188196f,  -0.708701f,
  0.012178f,   0.011392f,   -0.209721f,  -0.884075f,  0.164024f,   -0.149444f,
  0.143020f,   -0.692341f,  -0.227012f,  0.453073f,   0.069857f,   -0.200682f,
  0.018247f,   -0.021139f,  -0.041317f,  0.261913f,   -0.228155f,  0.192613f,
  -0.063541f,  0.148114f,   0.029469f,   -0.800970f,  0.213047f,   -0.085196f,
  -1.143013f,  0.222139f,   -0.061324f,  0.044507f,   -2.515277f,  -5.660876f,
  -11.760623f, 0.130194f,   -0.436996f,  4.622432f,   -0.457507f,  -0.034249f,
  -3.263473f,  -0.498804f,  0.007853f,   0.399219f,   -1.800737f,  -3.899369f,
  4.417291f,   -1.009584f,  -7.713755f,  2.741141f,   -0.723780f,  0.123886f,
  -4.994302f,  -0.535376f,  2.064732f,   -0.488256f,  -1.818281f,  0.106451f,
  0.756957f,   1.499472f,   0.657610f,   -12.732122f, -5.874076f,  -8.561155f,
  -3.045424f,  -4.372559f,  -9.836555f,  -0.673533f,  -0.334720f,  1.690431f,
  -0.393732f,  0.545913f,   -3.332750f,  -0.476151f,  -0.295179f,  -0.057421f,
  -1.456660f,  -4.061445f,  0.112553f,   -1.181640f,  -6.661728f,  -2.127332f,
  -1.072065f,  -0.280956f,  -6.551425f,  -0.536725f,  -0.718327f,  -0.268650f,
  -0.972042f,  0.688322f,   2.303338f,   0.492784f,   -0.531351f,  -10.831738f,
  -6.536975f,  -5.913453f,  -1.930707f,  -5.425492f,  -11.583106f, 0.432496f,
  -0.542585f,  2.729111f,   -0.578627f,  -0.171262f,  -2.897873f,  0.074428f,
  -0.228792f,  0.535162f,   -1.177916f,  -4.157263f,  4.467278f,   -0.636676f,
  -7.621882f,  2.500491f,   -0.535092f,  -0.407224f,  -4.500089f,  -0.029855f,
  1.710763f,   -0.575392f,  -1.627628f,  -0.125988f,  1.149914f,   1.352979f,
  0.421749f,   -13.069753f, -6.013796f,  -8.318614f,
};

static const float av1_tx_split_nn_bias_16x32_layer0[32] = {
  -2.154176f, -5.448051f, -13.668990f, 2.965132f,  -0.285108f, 1.942034f,
  -0.158425f, 0.190044f,  -3.071860f,  -0.172429f, -0.201269f, -1.695773f,
  -1.661639f, -4.500017f, 3.270531f,   -0.867051f, -7.085031f, 4.411665f,
  -0.330444f, -0.272143f, -5.817105f,  -0.176634f, 4.029921f,  -0.232953f,
  -1.734610f, -2.046185f, 1.263547f,   3.684499f,  -0.027515f, -16.457491f,
  -7.033929f, -7.827878f,
};

static const float av1_tx_split_nn_weights_16x32_layer1[128] = {
  0.013862f,  -0.046420f, 0.585373f,  -0.747509f, 0.364218f,  -1.634125f,
  -0.624408f, -1.578624f, -1.332164f, -1.899552f, 0.210923f,  0.052844f,
  -0.374907f, 0.364624f,  0.264401f,  0.376575f,  0.248798f,  -0.005459f,
  0.353117f,  0.150672f,  0.112044f,  0.481664f,  0.340796f,  0.348110f,
  0.119798f,  -0.149402f, -0.206716f, 0.317158f,  0.668739f,  -0.041776f,
  0.058571f,  0.126437f,  -0.086351f, -0.591861f, 0.175761f,  0.086054f,
  0.271187f,  -0.212081f, 0.202385f,  0.153980f,  0.470226f,  0.155868f,
  0.214667f,  0.305256f,  0.008323f,  0.406445f,  0.338437f,  0.612904f,
  -0.332063f, -0.619698f, 0.172363f,  0.779727f,  -0.224778f, -2.044589f,
  -0.327455f, -0.101434f, -0.503589f, -0.559802f, 0.479385f,  0.234342f,
  0.094556f,  0.538528f,  -0.036429f, -0.849440f, 0.750812f,  -0.797631f,
  0.693951f,  -0.039293f, 0.420527f,  -1.041519f, 0.114456f,  0.327767f,
  0.146245f,  -0.373977f, -0.041741f, -0.488945f, -0.310631f, 0.166438f,
  -0.157407f, 0.115747f,  0.787631f,  -0.492723f, 1.489448f,  -0.025978f,
  0.012459f,  -0.060184f, -0.263492f, -0.196933f, 0.406167f,  0.519842f,
  -0.307797f, -0.068126f, 0.048131f,  0.174070f,  0.202685f,  0.123606f,
  0.181112f,  0.455417f,  0.180017f,  -0.421376f, 0.007059f,  0.626616f,
  0.561902f,  0.521469f,  0.050960f,  -0.266677f, -0.018453f, -0.066509f,
  0.341632f,  0.013076f,  -0.126784f, -0.087816f, 0.808517f,  0.026144f,
  0.174054f,  -0.065822f, -0.309264f, -2.057102f, -0.448292f, 0.650055f,
  -0.470839f, -1.822060f, 0.071965f,  -0.494685f, 0.692688f,  -1.563420f,
  -1.176936f, 1.605713f,
};

static const float av1_tx_split_nn_bias_16x32_layer1[4] = {
  0.762402f,
  -0.419528f,
  -0.448386f,
  0.013578f,
};

static const NN_CONFIG av1_tx_split_nnconfig_16x32 = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      32,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_16x32_layer0,
      av1_tx_split_nn_weights_16x32_layer1,
  },
  {
      av1_tx_split_nn_bias_16x32_layer0,
      av1_tx_split_nn_bias_16x32_layer1,
  }
};
/******************************************************************************/

// Tx split model for 16x64 block.
static const float av1_tx_split_nn_weights_16x64_layer0[128] = {
  1.321162f,  -0.338026f, -0.315452f, 0.071746f,  -0.262095f, 0.208378f,
  1.011826f,  0.480933f,  0.147428f,  1.344027f,  -0.006655f, -0.385775f,
  1.196962f,  -0.263285f, -1.698326f, -1.889782f, -0.580532f, -1.040935f,
  -1.944421f, -2.488641f, -4.197682f, -0.447540f, 0.002644f,  0.189506f,
  -1.189043f, -0.668555f, -0.896686f, -1.832895f, -0.589928f, -1.398583f,
  -0.696757f, -0.494821f, 1.416331f,  0.173597f,  0.285980f,  -0.175968f,
  -0.356637f, -0.160898f, 0.787003f,  1.105643f,  -0.202014f, 1.597886f,
  -0.088827f, 0.161532f,  1.026822f,  0.318126f,  -1.982353f, -1.735532f,
  -0.077079f, -0.489411f, -2.055496f, 0.564962f,  -3.903480f, -0.761782f,
  0.584135f,  0.326250f,  -2.112016f, -0.199651f, -1.124749f, -0.440976f,
  -0.073860f, 1.119996f,  0.300064f,  0.252159f,  1.291686f,  0.122855f,
  -0.051772f, 0.016534f,  -0.190905f, -0.085955f, 0.511160f,  0.844423f,
  0.036928f,  1.444543f,  0.100446f,  0.125971f,  0.947409f,  -0.435303f,
  -1.644089f, -1.453313f, 0.054772f,  -0.473445f, -1.125447f, -4.443506f,
  -3.477941f, -0.738945f, -0.639857f, -0.787817f, -1.907924f, 0.064659f,
  -0.695584f, 0.982832f,  0.200189f,  -2.051821f, -0.084639f, -0.129354f,
  -3.759842f, -1.399974f, -1.875461f, 2.812289f,  -3.991897f, -0.823110f,
  2.736391f,  2.622819f,  -1.444650f, -4.365030f, -0.531664f, -0.407399f,
  -3.438725f, -0.951106f, -2.993652f, -2.810758f, 0.314805f,  -0.232404f,
  -1.833882f, -1.259001f, -4.116950f, -0.766911f, -0.395862f, -0.092190f,
  -2.228919f, 0.386273f,  -0.736178f, 0.974000f,  0.046369f,  0.682809f,
  -0.229035f, -0.255198f,
};

static const float av1_tx_split_nn_bias_16x64_layer0[16] = {
  -2.354578f, -0.965667f, -3.659526f, 4.387677f,  -5.532654f, -0.080437f,
  0.771596f,  -2.054621f, -2.355837f, 1.287621f,  -0.659201f, 2.648803f,
  3.998829f,  2.295854f,  0.953344f,  -2.764555f,
};

static const float av1_tx_split_nn_weights_16x64_layer1[32] = {
  -0.386260f, 0.227140f,  0.541364f,  0.387506f,  -0.712658f, 0.037970f,
  0.062652f,  -0.261678f, 0.881087f,  -0.534471f, 0.354547f,  0.260619f,
  0.380408f,  -0.053347f, -0.174436f, 0.251525f,  -0.319863f, 0.524512f,
  0.434955f,  -0.489995f, 0.148547f,  -0.150256f, 0.070519f,  0.399902f,
  -0.089852f, 0.369853f,  0.215425f,  0.073713f,  0.359998f,  -0.188487f,
  -0.308656f, 0.236498f,
};

static const float av1_tx_split_nn_bias_16x64_layer1[2] = {
  -0.007776f,
  0.007776f,
};

static const NN_CONFIG av1_tx_split_nnconfig_16x64 = {
  8,  // num_inputs
  2,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_16x64_layer0,
      av1_tx_split_nn_weights_16x64_layer1,
  },
  {
      av1_tx_split_nn_bias_16x64_layer0,
      av1_tx_split_nn_bias_16x64_layer1,
  }
};
/******************************************************************************/

// Tx split model for 32x64 block.
static const float av1_tx_split_nn_weights_32x64_layer0[256] = {
  0.174930f,  -0.788353f, 0.238239f,  0.375000f,  -0.075364f, -0.306148f,
  -0.413254f, 0.313947f,  -1.296242f, -0.379396f, -0.495781f, -0.561032f,
  -0.093978f, -0.257821f, -0.320403f, 0.253673f,  -0.022910f, -0.150037f,
  0.222465f,  -0.135986f, 0.048318f,  -0.586822f, 0.180605f,  0.814833f,
  0.146110f,  0.157469f,  -0.512512f, 0.046844f,  0.111553f,  -0.074888f,
  -0.038607f, 0.341655f,  -6.126758f, -0.620331f, -2.807292f, 0.176229f,
  -3.513526f, -4.282414f, -0.670014f, -1.254760f, -0.809137f, -5.133852f,
  -1.764843f, -0.240554f, -2.790462f, 0.026693f,  -6.161190f, -2.661687f,
  -6.751094f, -0.295490f, -9.323830f, -6.462364f, 0.044558f,  -0.549099f,
  -1.434814f, 0.053973f,  -9.933718f, -0.441322f, -1.149451f, -4.816069f,
  -3.706382f, -4.250051f, -4.632912f, -1.239063f, -0.077506f, -0.755410f,
  -0.121965f, 1.039215f,  -0.309046f, 0.350590f,  -0.057961f, -0.087833f,
  -1.031554f, 0.336188f,  0.071003f,  0.144688f,  0.107221f,  0.194534f,
  0.059731f,  -0.169388f, 0.070854f,  -0.140780f, 0.092409f,  -0.487194f,
  -0.011882f, -1.772330f, -0.226842f, 0.890017f,  -0.043476f, 0.077830f,
  -0.465901f, 0.080351f,  -0.244958f, 0.535032f,  0.038930f,  0.020432f,
  -6.350341f, 0.239342f,  -1.830402f, 0.109359f,  -3.880939f, -4.519018f,
  -1.010566f, -0.637861f, -0.127488f, -5.297055f, -2.538575f, 0.222304f,
  -3.126220f, -0.363607f, -0.041214f, -1.785928f, -6.018940f, -0.140073f,
  1.782313f,  -7.018731f, -1.063239f, 0.237971f,  -1.032591f, 0.032885f,
  3.340030f,  -0.112060f, 0.007971f,  -4.540715f, -3.714948f, -2.425507f,
  -4.572936f, -0.113117f, -0.278739f, -0.629884f, 0.033443f,  0.893968f,
  0.230269f,  -0.067762f, 0.532983f,  -0.263347f, -1.122392f, 0.035942f,
  0.537825f,  -1.744690f, 0.072737f,  0.062134f,  0.270978f,  -0.092621f,
  -0.030301f, 0.287399f,  -0.059082f, 0.063325f,  -0.052166f, 0.246753f,
  0.341046f,  0.795280f,  -0.067325f, -0.236837f, -0.589188f, -0.037198f,
  0.073297f,  -0.274649f, 0.040522f,  0.146410f,  -5.615050f, 0.018904f,
  -2.865930f, -0.164234f, -3.414918f, -3.739706f, -1.064124f, -0.874834f,
  0.114757f,  -5.763709f, -0.795600f, 0.209524f,  -2.321763f, -0.473452f,
  3.304655f,  -3.252949f, -6.267320f, -0.518845f, 2.280299f,  -6.132434f,
  0.176914f,  0.029708f,  -1.753716f, -0.266310f, 3.240977f,  -0.016390f,
  0.398219f,  -4.141730f, -3.198847f, -2.592462f, -3.607648f, 0.275719f,
  -3.894774f, 2.442656f,  -3.266171f, -0.484086f, -3.376189f, -3.561981f,
  0.170081f,  -0.887124f, -3.618005f, -4.362345f, -2.615719f, 0.124505f,
  -1.993958f, -0.309628f, 1.076189f,  -4.557148f, -5.345412f, -0.738785f,
  -1.933232f, -6.284731f, -0.587846f, 0.370912f,  -2.992463f, -0.685548f,
  -1.137433f, -0.659132f, -2.195347f, -3.913321f, -1.818603f, -5.296817f,
  -5.574424f, -3.967305f, -6.394349f, 0.213090f,  -2.188700f, -0.412520f,
  -3.406144f, -3.870236f, -1.413042f, -0.855080f, 0.433317f,  -5.982457f,
  -1.667145f, -0.220363f, -2.816697f, -0.230983f, 2.063738f,  -2.856512f,
  -6.052085f, -0.285812f, 1.514536f,  -6.714015f, -0.499445f, 0.197907f,
  -1.438813f, -0.186260f, 3.355147f,  0.005836f,  0.441239f,  -4.375771f,
  -3.316825f, -2.355774f, -4.233965f, -0.073076f,
};

static const float av1_tx_split_nn_bias_32x64_layer0[32] = {
  -7.337619f, -0.415367f, -2.395953f, -1.417987f, -3.782653f, -4.145950f,
  -1.193590f, -0.792001f, 1.980830f,  -6.101995f, -5.003466f, -2.239284f,
  -2.836128f, -0.225015f, 1.415658f,  -3.403192f, -5.845839f, -0.275306f,
  3.617564f,  -7.216553f, -1.777272f, -2.407655f, -2.711518f, 1.608490f,
  0.458590f,  0.134730f,  3.396708f,  -4.788477f, -3.199826f, -6.199792f,
  -3.778460f, 3.516962f,
};

static const float av1_tx_split_nn_weights_32x64_layer1[128] = {
  0.129247f,  -0.488246f, 0.878115f,  -0.701747f, 0.350697f,  0.013306f,
  0.003358f,  -0.227000f, 0.033832f,  -1.351854f, 0.276866f,  0.773895f,
  -0.088802f, -0.015423f, 0.068671f,  0.350273f,  0.013196f,  -1.118868f,
  -0.403437f, -0.135241f, 0.097027f,  0.068326f,  0.524979f,  0.010971f,
  0.411202f,  -0.302438f, -0.096315f, -0.073003f, -0.282602f, 0.005891f,
  -0.402003f, 0.150988f,  0.551788f,  -0.077146f, 0.071414f,  -0.134174f,
  0.342006f,  -0.223276f, -0.051588f, 0.986920f,  -0.137083f, 0.216190f,
  -0.127833f, -0.339068f, -0.340588f, 0.035618f,  0.029663f,  0.211554f,
  0.269838f,  -0.761807f, 0.397209f,  -0.356177f, 0.191740f,  -0.029469f,
  0.216488f,  0.181930f,  -0.341281f, -0.002890f, -0.121464f, -0.149672f,
  0.082972f,  -0.542916f, 0.802330f,  -0.659609f, 0.664927f,  -1.333394f,
  0.233435f,  -1.154135f, 0.012775f,  0.052679f,  0.333048f,  0.129154f,
  0.139791f,  -0.109790f, -0.705446f, 0.262807f,  -0.010843f, -1.169575f,
  -1.080046f, -1.243374f, 0.062239f,  0.136611f,  0.173737f,  0.032992f,
  -0.106557f, 0.288162f,  0.291480f,  0.461210f,  0.486279f,  0.588383f,
  -0.202295f, 0.485822f,  0.504003f,  0.373567f,  0.289653f,  0.041858f,
  -0.217813f, -1.007038f, 0.732666f,  0.378798f,  -0.082937f, -0.335160f,
  0.061636f,  0.113871f,  -0.597042f, 0.157087f,  -0.044386f, 0.084775f,
  -0.742806f, 0.709178f,  0.022134f,  -0.250765f, -0.212210f, -0.095264f,
  -0.299416f, -1.590267f, 0.311821f,  -0.536606f, 0.716837f,  0.054289f,
  -0.307013f, -0.600937f, -0.281585f, -0.336099f, 0.177900f,  -0.512005f,
  -0.136673f, -0.099282f,
};

static const float av1_tx_split_nn_bias_32x64_layer1[4] = {
  0.677458f,
  -0.833130f,
  -0.812087f,
  0.541961f,
};

static const NN_CONFIG av1_tx_split_nnconfig_32x64 = {
  8,  // num_inputs
  4,  // num_outputs
  1,  // num_hidden_layers
  {
      32,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_32x64_layer0,
      av1_tx_split_nn_weights_32x64_layer1,
  },
  {
      av1_tx_split_nn_bias_32x64_layer0,
      av1_tx_split_nn_bias_32x64_layer1,
  }
};
/******************************************************************************/

// Tx split model for 4x16 block.
static const float av1_tx_split_nn_weights_4x16_layer0[128] = {
  0.931474f,  -0.183700f, -0.249171f, 0.047625f,  0.727252f,  -0.219069f,
  -0.058618f, -0.415904f, -0.109755f, 0.279453f,  -0.350876f, -0.140681f,
  0.353107f,  -0.389719f, -0.097842f, -0.168431f, -2.902663f, -0.466705f,
  -0.101990f, -0.109210f, -3.369733f, -3.643304f, -2.841429f, -3.203382f,
  1.703362f,  -3.205956f, -4.582482f, -0.427450f, -0.573612f, -2.745853f,
  -7.784524f, -0.055847f, 0.719537f,  0.036562f,  0.118095f,  -0.046482f,
  0.870231f,  -1.159974f, 0.188009f,  -1.043338f, -0.013348f, -0.111578f,
  0.172611f,  -0.238768f, -0.306337f, 0.111649f,  0.039801f,  0.129077f,
  0.625977f,  0.231716f,  -1.016711f, -0.926862f, 0.384949f,  0.747370f,
  -3.097634f, 0.469422f,  -0.792179f, -2.968342f, 0.790372f,  0.081649f,
  -0.265527f, -0.527705f, -6.181936f, 0.463195f,  0.311903f,  0.145510f,
  0.107812f,  -0.001030f, 0.457367f,  -0.797839f, -0.085985f, -0.672411f,
  0.058164f,  -0.222004f, 0.216847f,  0.380492f,  -0.047522f, 0.121026f,
  0.130142f,  0.063936f,  1.136972f,  -0.510322f, 0.753285f,  -0.554743f,
  0.991644f,  1.269222f,  -3.498582f, 0.937744f,  -0.601882f, -3.515439f,
  0.165552f,  -0.519889f, 0.083702f,  1.465118f,  -6.376130f, -0.905325f,
  -1.391479f, -0.128387f, -0.208748f, -0.571494f, -1.186599f, -1.073797f,
  -2.341276f, -1.148740f, -1.499803f, -3.464051f, 3.592625f,  -0.504882f,
  -0.484961f, -0.854744f, -7.010800f, 0.062896f,  0.221322f,  -0.107482f,
  -0.247326f, -0.686206f, 0.944529f,  0.612449f,  -3.594441f, 1.026726f,
  -0.627217f, -3.868834f, 0.788894f,  -0.649344f, 0.025808f,  0.980366f,
  -6.749431f, -0.120214f,
};

static const float av1_tx_split_nn_bias_4x16_layer0[16] = {
  -1.996939f, -0.200853f, 5.378592f,   -0.061417f, 2.076407f, 1.752848f,
  -3.205579f, -2.337602f, -14.747408f, -2.516311f, 6.179139f, -0.004017f,
  -0.206164f, 5.325444f,  -3.875729f,  3.309283f,
};

static const float av1_tx_split_nn_weights_4x16_layer1[32] = {
  -0.023662f, 0.343919f,  -0.404740f, -0.501902f, 0.127567f,  0.341077f,
  -0.330534f, -0.454893f, 0.385408f,  -0.042397f, 0.209227f,  -0.243938f,
  -0.261379f, 0.085089f,  0.057944f,  0.471134f,  -0.492052f, -0.442237f,
  0.426032f,  -0.664274f, 0.377848f,  0.140093f,  -0.360874f, -0.024475f,
  0.020209f,  0.374680f,  -0.234019f, -0.047818f, 0.013466f,  0.411923f,
  -0.427383f, -0.313637f,
};

static const float av1_tx_split_nn_bias_4x16_layer1[2] = {
  0.676961f,
  -0.676960f,
};

static const NN_CONFIG av1_tx_split_nnconfig_4x16 = {
  8,  // num_inputs
  2,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_4x16_layer0,
      av1_tx_split_nn_weights_4x16_layer1,
  },
  {
      av1_tx_split_nn_bias_4x16_layer0,
      av1_tx_split_nn_bias_4x16_layer1,
  }
};
/******************************************************************************/

// Tx split model for 4x8 block.
static const float av1_tx_split_nn_weights_4x8_layer0[128] = {
  -0.059867f, -0.858259f, 0.010204f,  0.080800f,  0.353625f,  -0.467501f,
  -0.279892f, -0.080181f, -0.142841f, 0.224800f,  0.063069f,  0.466821f,
  -0.721102f, 0.336990f,  0.415052f,  0.375842f,  -0.742670f, -0.257965f,
  -4.056455f, -0.417200f, -1.612972f, -2.814216f, -0.474070f, -0.124376f,
  -0.618001f, -4.292533f, -2.270582f, -0.616103f, -0.001015f, 0.351215f,
  -0.703607f, 0.066139f,  -0.017563f, -1.048936f, 0.213210f,  0.023938f,
  -0.137495f, 0.404873f,  0.268698f,  -0.253776f, 0.060792f,  -0.329440f,
  -0.050264f, -0.162900f, -1.068018f, 1.123592f,  0.138147f,  1.124433f,
  -0.603211f, 0.265268f,  -4.072471f, 0.229473f,  0.820926f,  -3.152894f,
  -0.410831f, 0.106688f,  -1.115363f, -3.071087f, -2.814048f, 0.355872f,
  0.353884f,  0.501626f,  -0.913179f, 0.617763f,  0.075130f,  0.227053f,
  -0.292389f, 0.396195f,  -0.219637f, 0.055775f,  0.013158f,  0.334188f,
  0.125975f,  0.080001f,  -0.026318f, -0.695502f, 0.129763f,  0.047652f,
  -0.554123f, 0.026672f,  -0.524236f, -0.040346f, -4.153587f, -0.977733f,
  0.472076f,  -2.216423f, -0.339250f, -0.685022f, 0.912858f,  -4.646110f,
  -2.836090f, -1.040271f, 0.049709f,  -0.207233f, 0.525127f,  -0.040133f,
  0.184338f,  0.097427f,  -3.661171f, -0.121583f, -1.107226f, -2.643712f,
  -0.511058f, -0.725856f, 0.115350f,  -3.295951f, -1.655626f, -0.038201f,
  0.129429f,  -0.097117f, -0.632148f, -0.021324f, -0.515180f, 0.347707f,
  -3.953322f, -0.849508f, 0.214540f,  -2.952381f, -0.618445f, -0.226599f,
  0.076723f,  -3.484516f, -2.820108f, -0.528298f, 0.034826f,  -0.021566f,
  -0.225556f, -0.112556f,
};

static const float av1_tx_split_nn_bias_4x8_layer0[16] = {
  -0.027956f, -0.189276f, -3.693787f, 5.900157f,  -0.579183f, -4.119867f,
  -0.099637f, -0.050242f, 3.485042f,  -0.202202f, -2.376010f, 5.041715f,
  -6.080580f, -6.096891f, -0.807981f, -0.497873f,
};

static const float av1_tx_split_nn_weights_4x8_layer1[32] = {
  -0.422349f, -0.152340f, -0.156641f, -0.566306f, 0.871320f, -1.148644f,
  0.018000f,  -0.138313f, -0.237801f, -0.231125f, 0.229006f, -0.231495f,
  -0.204347f, 0.182788f,  -0.295450f, -0.294503f, 0.283787f, 0.424576f,
  -0.552496f, -0.049572f, 0.270025f,  0.326040f,  0.103340f, -0.084435f,
  -0.317648f, 0.082481f,  -0.222554f, 0.208414f,  0.335259f, 0.076776f,
  0.222721f,  -0.197558f,
};

static const float av1_tx_split_nn_bias_4x8_layer1[2] = {
  0.066815f,
  -0.066828f,
};

static const NN_CONFIG av1_tx_split_nnconfig_4x8 = {
  8,  // num_inputs
  2,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_4x8_layer0,
      av1_tx_split_nn_weights_4x8_layer1,
  },
  {
      av1_tx_split_nn_bias_4x8_layer0,
      av1_tx_split_nn_bias_4x8_layer1,
  }
};
/******************************************************************************/

// Tx split model for 8x32 block.
static const float av1_tx_split_nn_weights_8x32_layer0[192] = {
  0.726351f,  -1.658811f, 0.451881f,   0.047012f,  0.881003f,  -0.115941f,
  0.577393f,  0.213052f,  0.477786f,   -0.176176f, 0.123633f,  0.027620f,
  0.232188f,  1.129627f,  -0.251854f,  0.339140f,  -0.168740f, 0.198508f,
  -0.244085f, -0.694087f, 0.020348f,   -1.866305f, 1.182810f,  -0.374377f,
  -3.587806f, -5.236939f, -13.162245f, -1.853044f, -1.449030f, -0.508827f,
  -2.694105f, -0.724095f, -0.363693f,  -0.060313f, -2.414339f, -6.216967f,
  -7.506750f, -4.741603f, 0.098463f,   -6.214391f, -2.618696f, -8.985423f,
  -0.263702f, -1.571406f, -0.435366f,  -5.779884f, -4.939770f, -0.223961f,
  1.781999f,  -1.419670f, -0.113626f,  -0.358763f, 0.620567f,  -0.612241f,
  1.470790f,  -0.098228f, -0.244735f,  0.209020f,  -0.131194f, 0.048502f,
  -0.214294f, 1.152014f,  0.138622f,   -0.221118f, -0.050420f, -0.079560f,
  -0.075202f, -0.888932f, -0.136324f,  -1.739783f, 1.273290f,  -0.518412f,
  0.307027f,  1.544233f,  3.797998f,   -3.109860f, -0.323017f, 0.386683f,
  0.455765f,  0.547071f,  0.219539f,   -0.446468f, -2.189370f, -5.875957f,
  4.499691f,  1.450096f,  -0.182391f,  -5.920831f, -2.412574f, -8.012991f,
  -0.534316f, -0.165597f, -0.218836f,  1.306716f,  1.297656f,  0.404674f,
  -0.987783f, -1.320403f, -0.095882f,  0.203498f,  0.594365f,  -0.357080f,
  -0.913550f, -0.097239f, -0.232853f,  -0.040541f, -0.065848f, -0.054634f,
  -0.035527f, 1.388648f,  0.101964f,   0.025207f,  0.066174f,  -0.334730f,
  0.333474f,  -0.883873f, 0.111731f,   -1.681239f, 1.614299f,  -0.213852f,
  1.721783f,  1.774522f,  4.027859f,   -1.960438f, 0.873898f,  -0.587409f,
  0.838624f,  0.193182f,  0.025481f,   -0.347464f, -2.027041f, -7.283419f,
  -1.797248f, 1.427584f,  -0.420054f,  -5.902254f, -2.872956f, -8.985358f,
  -0.393064f, 0.753716f,  -0.046870f,  1.917338f,  1.535414f,  -0.508275f,
  -0.882495f, -1.698544f, 6.886456f,   -1.060716f, -1.744253f, 2.098560f,
  -1.434038f, -0.146568f, -0.415379f,  -0.381683f, -2.009703f, -5.462253f,
  -1.431488f, -1.643857f, -0.076830f,  -4.822495f, -3.613893f, -9.371571f,
  -0.611498f, -1.913816f, -0.791422f,  -2.478602f, -1.963700f, 1.348821f,
  0.847475f,  1.494549f,  3.375612f,   -2.422148f, 0.567734f,  -0.161562f,
  0.875462f,  0.532491f,  -0.490413f,  -0.185097f, -2.205541f, -6.538527f,
  2.576536f,  1.417056f,  -0.278492f,  -5.849622f, -2.914950f, -9.264110f,
  -0.503514f, 0.543363f,  0.061321f,   1.912107f,  1.460162f,  -0.246501f,
};

static const float av1_tx_split_nn_bias_8x32_layer0[24] = {
  1.377979f,  -2.252306f, 2.304556f,  -4.408864f, 3.430554f,  1.441219f,
  -0.574199f, -2.395206f, -0.137233f, -0.116980f, -1.910153f, -6.181450f,
  2.490632f,  -2.057728f, -0.141788f, -4.671548f, -2.915712f, -9.710190f,
  -0.249739f, 4.023502f,  -0.039308f, 0.767978f,  0.951357f,  -0.857707f,
};

static const float av1_tx_split_nn_weights_8x32_layer1[48] = {
  -0.154758f, -0.369174f, 0.122262f,  0.739003f,  0.018516f,  -0.230132f,
  -0.137314f, -0.513737f, 0.099792f,  0.348816f,  0.213400f,  -0.009192f,
  -0.399647f, -0.220232f, 0.441242f,  0.301853f,  -0.113552f, -0.171579f,
  -0.029307f, 0.237964f,  0.276337f,  0.390282f,  -0.546902f, 0.227962f,
  -0.205271f, -0.407810f, -0.332103f, 0.181622f,  -0.128722f, -0.170411f,
  0.125837f,  -0.480047f, 1.037550f,  -0.368642f, 0.402998f,  0.245873f,
  0.275265f,  -0.045513f, -0.216206f, 0.110246f,  -0.189956f, -0.161064f,
  0.069623f,  -0.611949f, 0.407681f,  -0.206631f, -0.355554f, -0.137990f,
};

static const float av1_tx_split_nn_bias_8x32_layer1[2] = {
  -0.134962f,
  0.134961f,
};

static const NN_CONFIG av1_tx_split_nnconfig_8x32 = {
  8,  // num_inputs
  2,  // num_outputs
  1,  // num_hidden_layers
  {
      24,
  },  // hidden nodes
  {
      av1_tx_split_nn_weights_8x32_layer0,
      av1_tx_split_nn_weights_8x32_layer1,
  },
  {
      av1_tx_split_nn_bias_8x32_layer0,
      av1_tx_split_nn_bias_8x32_layer1,
  }
};
/******************************************************************************/
#else   // CONFIG_NEW_TX_PARTITION
// Tx split model for 4x8 block.
static const float av1_tx_split_nn_weights_4x8_layer0[8 * 16] = {
  0.068650f,  -0.732073f, -0.040361f, 0.322550f,  -0.021123f, 0.212518f,
  -0.350546f, 0.435987f,  -0.111756f, -0.401568f, 0.069548f,  -0.313000f,
  0.073918f,  -0.373805f, -0.775810f, -0.124753f, 0.181094f,  -0.602641f,
  -0.026219f, -0.350112f, 0.020599f,  -0.311752f, -0.476482f, -0.669465f,
  -0.310921f, 0.348869f,  -0.115984f, 0.154250f,  0.200485f,  -0.016689f,
  0.020392f,  0.413810f,  0.634064f,  -0.627530f, 0.399178f,  -0.012284f,
  0.472030f,  0.091087f,  -0.706100f, -0.447944f, -0.274226f, 0.445656f,
  0.309339f,  0.505522f,  0.038496f,  -0.152809f, 0.408684f,  -0.068151f,
  0.271612f,  0.353233f,  -0.150365f, 0.075212f,  -0.035096f, 0.346615f,
  0.124382f,  0.477072f,  0.216288f,  0.070548f,  -0.106362f, 0.681613f,
  -0.145502f, -0.218631f, -0.099248f, -0.001983f, -0.196819f, -0.969045f,
  0.063009f,  -0.123053f, 0.104875f,  -0.137581f, -0.282933f, -0.003624f,
  -0.315659f, -0.333523f, -0.503000f, -0.100063f, -0.536711f, -0.059978f,
  -0.670248f, -0.353762f, 0.181109f,  0.289715f,  -0.071206f, 0.261141f,
  0.052796f,  -0.114554f, -0.139214f, -0.261380f, 0.075984f,  -0.647925f,
  -0.099528f, -0.677814f, 0.015712f,  -0.389385f, -0.095622f, -0.165117f,
  -0.109454f, -0.175240f, -0.393914f, 0.212330f,  0.037822f,  0.248280f,
  0.180197f,  0.110493f,  -0.525727f, -0.092329f, -0.524029f, -0.407364f,
  -0.542373f, -0.435626f, -0.912194f, 0.062794f,  0.160433f,  0.741485f,
  -0.103659f, -0.119327f, -0.055275f, 0.334358f,  0.014713f,  0.046327f,
  0.831114f,  -0.576682f, 0.354369f,  -0.082088f, 0.452331f,  0.039730f,
  -0.792429f, -0.385862f,
};

static const float av1_tx_split_nn_bias_4x8_layer0[16] = {
  0.238621f,  2.186830f,  1.383035f,  -0.867139f, 1.257119f, -0.351571f,
  -0.240650f, -0.971692f, 2.744843f,  1.116991f,  0.139062f, -0.165332f,
  0.262171f,  -1.598153f, -1.427340f, -1.602306f,
};

static const float av1_tx_split_nn_weights_4x8_layer1[16] = {
  -0.367134f, 1.373058f, -0.897039f, -0.326819f, -0.734030f, -0.290413f,
  -0.501249f, 0.505321f, -0.537692f, -0.767893f, 0.268697f,  0.278987f,
  0.085082f,  0.614986f, 0.847904f,  0.637578f,
};

static const float av1_tx_split_nn_bias_4x8_layer1[1] = {
  0.20586078f,
};

static const NN_CONFIG av1_tx_split_nnconfig_4x8 = {
  8,  // num_inputs
  1,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_4x8_layer0,
      av1_tx_split_nn_weights_4x8_layer1,
  },
  {
      av1_tx_split_nn_bias_4x8_layer0,
      av1_tx_split_nn_bias_4x8_layer1,
  },
};
/******************************************************************************/

// Tx split model for 8x8 block.
static const float av1_tx_split_nn_weights_8x8_layer0[144] = {
  0.177983f,  -0.938386f, -0.074460f, -0.221843f, -0.073182f, -0.295155f,
  -0.098202f, -0.279510f, 0.001054f,  -0.119319f, -1.835282f, -0.581507f,
  -1.222222f, -1.049006f, -0.807508f, -0.454252f, -0.774879f, -0.180607f,
  -0.886976f, -0.231971f, -0.824677f, -0.351872f, -1.323819f, 0.235378f,
  0.015331f,  -0.341818f, 0.145549f,  -0.348362f, 0.147647f,  -0.323400f,
  0.047558f,  -0.553025f, -0.295485f, -0.330368f, -0.530605f, -0.407516f,
  0.447740f,  0.782381f,  -0.179164f, -0.584675f, -0.052645f, 0.038656f,
  -0.096783f, 0.038342f,  -0.170762f, -0.405844f, -0.552665f, -0.509866f,
  0.757204f,  -1.296465f, 0.631015f,  0.009265f,  0.646192f,  0.044523f,
  0.653161f,  0.033820f,  0.849639f,  -0.068555f, -1.036085f, -0.511652f,
  0.104693f,  -1.458690f, 0.286051f,  -0.089800f, 0.381564f,  -0.302640f,
  0.304465f,  -0.268706f, 0.432603f,  -0.117914f, -2.070031f, -0.565696f,
  -0.073027f, -1.783570f, -0.318144f, -0.320990f, -0.343966f, -0.140996f,
  -0.322977f, -0.232147f, -0.373210f, -0.158266f, -1.922305f, -0.634373f,
  0.101894f,  -0.221847f, 0.018412f,  -0.423887f, -0.266684f, -0.444930f,
  -0.196237f, 0.106638f,  -0.065834f, -0.538401f, -0.280772f, -0.620348f,
  1.089957f,  -0.799928f, 0.504112f,  -0.165763f, 0.578741f,  -0.172653f,
  0.547316f,  -0.143484f, 0.717220f,  -0.297190f, -1.237854f, -0.074819f,
  -0.977304f, -0.484092f, -0.646427f, -0.451443f, -0.612126f, -0.224475f,
  -0.731608f, -0.257077f, -0.665857f, -0.346742f, -1.216372f, 0.227267f,
  0.231249f,  -1.693073f, -0.035899f, 0.380845f,  -0.058476f, 0.409405f,
  -0.066679f, 0.406731f,  -0.068501f, 0.396748f,  0.639462f,  0.150834f,
  -0.418659f, -1.421931f, 0.101889f,  0.083573f,  0.129746f,  0.134460f,
  0.081185f,  0.127420f,  0.083664f,  0.051096f,  1.361688f,  0.386093f,
};

static const float av1_tx_split_nn_bias_8x8_layer0[12] = {
  4.280443f, 2.218902f, -0.256953f, 3.161431f,  2.082548f, 2.506052f,
  2.563224f, 1.421976f, -1.627813f, -1.436085f, 2.297265f, 1.500469f,
};

static const float av1_tx_split_nn_weights_8x8_layer1[12] = {
  1.178833f,  -0.428527f, -0.078737f, 0.381434f, -0.466895f, -0.901745f,
  -0.766968f, -0.356663f, 0.450146f,  0.509370f, -0.356604f, -0.443506f,
};

static const float av1_tx_split_nn_bias_8x8_layer1[1] = {
  -0.156294f,
};

static const NN_CONFIG av1_tx_split_nnconfig_8x8 = {
  12,  // num_inputs
  1,   // num_outputs
  1,   // num_hidden_layers
  {
      12,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_8x8_layer0,
      av1_tx_split_nn_weights_8x8_layer1,
  },
  {
      av1_tx_split_nn_bias_8x8_layer0,
      av1_tx_split_nn_bias_8x8_layer1,
  },
};
/******************************************************************************/

// Tx split model for 8x16 block.
static const float av1_tx_split_nn_weights_8x16_layer0[8 * 64] = {
  0.374660f,  0.218905f,  -0.139779f, 0.212141f,  0.056517f,  0.051114f,
  0.042860f,  -0.273258f, -0.340809f, 0.138983f,  -0.216996f, -0.241519f,
  -0.123244f, 0.078577f,  -0.472273f, -0.194201f, 0.125056f,  0.239761f,
  -0.332782f, 0.174782f,  -0.211400f, -0.129795f, 0.062195f,  0.113176f,
  -0.008869f, 0.140764f,  0.059833f,  0.163826f,  0.359293f,  -0.109797f,
  -0.022091f, -0.059536f, -0.188226f, 0.179709f,  0.031386f,  0.164790f,
  0.214364f,  0.198555f,  0.152262f,  -0.242980f, 0.319367f,  -0.136902f,
  0.046524f,  -0.043591f, 0.342178f,  -0.011757f, -0.014286f, 0.072871f,
  -0.278314f, -0.345303f, -0.252103f, -0.107154f, -0.235101f, -0.106739f,
  -0.120865f, -0.160042f, 0.240028f,  0.112902f,  -0.141587f, -0.703012f,
  -0.136591f, 0.318993f,  -0.154417f, -0.054668f, 0.192870f,  0.176166f,
  -0.029965f, 0.266942f,  -0.178384f, 0.038680f,  0.134403f,  -0.002426f,
  0.534825f,  -0.070923f, 0.413281f,  0.418148f,  0.093729f,  0.016454f,
  0.305358f,  -0.040512f, 0.069904f,  -0.227588f, -0.362220f, -0.031604f,
  -0.394901f, 0.071506f,  -0.342833f, -0.142550f, -0.164005f, 0.182600f,
  0.213062f,  0.076805f,  0.278758f,  0.125613f,  -0.035552f, 0.040971f,
  0.182785f,  -0.227961f, -0.105413f, -0.074949f, -0.084629f, -0.254767f,
  0.114657f,  0.047121f,  0.195902f,  0.264759f,  0.017799f,  0.210230f,
  0.150749f,  -0.142142f, 0.182494f,  -0.142415f, -0.259782f, -0.114830f,
  -0.198826f, 0.000061f,  -0.375668f, -0.276656f, -0.373202f, 0.210298f,
  0.422680f,  0.066960f,  0.351106f,  -0.209034f, 0.367195f,  -0.110274f,
  0.115573f,  -0.066642f, -0.389673f, -0.260447f, 0.056949f,  -0.180425f,
  0.069922f,  -0.153506f, -0.097053f, -0.111757f, 0.094069f,  0.144837f,
  -0.052984f, -0.506681f, -0.034474f, 0.279057f,  -0.105025f, 0.006656f,
  -0.125017f, -0.114096f, 0.103153f,  -0.117402f, -0.359472f, 0.072534f,
  0.110291f,  0.003088f,  -0.456897f, 0.038331f,  -0.322298f, 0.113942f,
  -0.119916f, -0.194392f, 0.093167f,  0.193459f,  0.074671f,  0.033602f,
  0.004440f,  -0.179578f, -0.036637f, -0.216172f, -0.296530f, -0.318992f,
  0.319160f,  -0.066218f, 0.291246f,  0.181292f,  0.089914f,  0.025273f,
  0.303128f,  0.019063f,  0.078545f,  -0.396919f, 0.014065f,  -0.122121f,
  0.037107f,  -0.151886f, -0.299392f, -0.172207f, -0.124571f, -0.232553f,
  0.102970f,  -0.225040f, 0.061059f,  -0.258188f, -0.469871f, -0.099607f,
  -0.061524f, -0.213700f, 0.070237f,  -0.289134f, -0.238225f, 0.256403f,
  -0.119344f, 0.067782f,  -0.398983f, -0.123975f, -0.200205f, -0.047038f,
  0.026569f,  0.031037f,  0.094302f,  -0.101239f, 0.433307f,  -0.303612f,
  0.088537f,  -0.164436f, 0.202471f,  -0.048592f, -0.251904f, 0.122577f,
  -0.309874f, -0.263405f, -0.292503f, 0.216589f,  0.035378f,  0.136599f,
  -0.145844f, -0.018211f, 0.174084f,  -0.449941f, -0.001428f, 0.064134f,
  0.039652f,  0.111083f,  -0.246076f, -0.204733f, 0.056559f,  -0.000123f,
  0.104049f,  0.138512f,  -0.128309f, 0.087855f,  0.232784f,  0.247138f,
  0.162766f,  0.154829f,  0.313605f,  -0.164115f, -0.050844f, 0.156549f,
  0.185279f,  -0.238962f, -0.308281f, -0.179592f, -0.193262f, 0.201670f,
  -0.203399f, -0.096831f, -0.127867f, 0.310674f,  -0.008181f, 0.004078f,
  -0.211038f, -0.193480f, -0.185639f, -0.150202f, -0.204858f, -0.240758f,
  0.114268f,  -0.032535f, -0.052403f, -0.234333f, -0.064072f, -0.208444f,
  -0.352853f, -0.224001f, -0.156330f, 0.215436f,  0.171846f,  0.291849f,
  0.108832f,  0.046991f,  -0.127801f, 0.032485f,  0.141493f,  0.123319f,
  -0.057250f, 0.315346f,  -0.061317f, -0.465086f, -0.130179f, -0.217841f,
  -0.239089f, -0.073251f, -0.327718f, 0.054905f,  -0.283169f, -0.028900f,
  0.071450f,  0.270072f,  0.248891f,  0.088052f,  0.253319f,  0.122808f,
  0.175490f,  -0.147805f, 0.089169f,  -0.045457f, -0.330788f, 0.099791f,
  -0.137376f, -0.195977f, -0.350942f, -0.284930f, -0.559037f, 0.030504f,
  0.162554f,  -0.199100f, -0.050453f, -0.131320f, -0.077863f, -0.066253f,
  -0.379723f, -0.424047f, -0.081182f, -0.252261f, -0.102815f, 0.058240f,
  -0.182036f, 0.176772f,  -0.070823f, 0.216054f,  -0.211533f, -0.232992f,
  0.279346f,  0.117984f,  0.236674f,  0.126625f,  -0.046220f, 0.044919f,
  0.278492f,  0.083944f,  0.180512f,  0.217994f,  0.401170f,  -0.064417f,
  0.011636f,  -0.139597f, -0.050020f, -0.268438f, -0.032803f, 0.024908f,
  -0.085713f, -0.012984f, -0.055192f, -0.338657f, 0.045826f,  -0.312849f,
  -0.023393f, -0.168800f, -0.030886f, -0.131816f, -0.253542f, -0.104812f,
  -0.354389f, 0.169464f,  0.094151f,  -0.217122f, -0.456397f, 0.211478f,
  0.219232f,  -0.155519f, -0.353700f, -0.264759f, -0.034709f, 0.034409f,
  -0.148639f, -0.132850f, -0.216791f, -0.118492f, 0.173721f,  -0.144181f,
  0.335028f,  0.176439f,  0.105980f,  0.169390f,  0.155615f,  -0.040618f,
  -0.176029f, 0.155569f,  -0.184833f, -0.171099f, -0.178663f, -0.032051f,
  -0.434334f, 0.092238f,  -0.263103f, 0.061804f,  -0.172957f, 0.005962f,
  -0.100176f, 0.125898f,  0.048092f,  -0.088141f, 0.247196f,  -0.221601f,
  -0.114474f, -0.124410f, -0.156393f, -0.181782f, -0.083562f, 0.034937f,
  0.403401f,  -0.046200f, 0.322259f,  0.219678f,  0.109850f,  0.051837f,
  0.196861f,  -0.019118f, 0.248818f,  -0.137567f, 0.127862f,  0.052293f,
  0.298726f,  0.275788f,  0.015344f,  0.058714f,  0.283691f,  -0.053794f,
  -0.123270f, -0.227761f, -0.141744f, -0.268515f, -0.007189f, -0.242117f,
  -0.252396f, -0.069017f, 0.034803f,  -0.003388f, -0.262577f, 0.062115f,
  -0.298393f, 0.215415f,  -0.153615f, 0.289902f,  0.085886f,  -0.504290f,
  0.077178f,  0.150861f,  -0.228848f, -0.261020f, 0.198204f,  0.162113f,
  0.346418f,  -0.286950f, 0.354756f,  -0.226419f, 0.024720f,  0.208037f,
  0.107286f,  -0.110849f, 0.104415f,  -0.207725f, 0.063932f,  -0.037748f,
  -0.167037f, -0.068282f, 0.320815f,  -0.051884f, 0.099989f,  -0.078388f,
  0.127071f,  0.046675f,  -0.336571f, -0.273080f, 0.264694f,  -0.007352f,
  -0.093828f, 0.094773f,  -0.144434f, 0.091795f,  -0.031615f, 0.056914f,
  0.064673f,  -0.136669f, 0.344734f,  0.225926f,  0.283451f,  -0.068354f,
  0.030572f,  0.180784f,  -0.378047f, -0.092962f, -0.083291f, 0.038970f,
  0.052094f,  -0.017932f, 0.216302f,  -0.184396f, 0.079888f,  0.210406f,
  -0.020627f, 0.244744f,  0.336972f,  -0.182914f, -0.220976f, -0.304225f,
  -0.330974f, -0.370868f, -0.084935f, -0.136489f, -0.210082f, -0.188088f,
  -0.408768f, 0.184693f,
};

static const float av1_tx_split_nn_bias_8x16_layer0[64] = {
  -0.274107f, 0.445751f,  0.234359f,  0.291593f,  0.163298f,  0.183707f,
  -0.548839f, -0.190779f, -0.163346f, -0.669028f, 0.399209f,  -0.354974f,
  0.000000f,  -0.254630f, 0.220149f,  0.371104f,  0.789759f,  0.270300f,
  0.195126f,  -0.206958f, 0.917708f,  -0.256232f, 1.131933f,  1.178944f,
  0.461270f,  0.246169f,  -0.818614f, -0.111986f, 0.759355f,  0.154889f,
  0.470299f,  -1.025250f, 0.678678f,  0.959346f,  -0.164105f, 0.544079f,
  -0.448733f, 0.649221f,  -0.536672f, 0.962758f,  -0.256427f, 0.808664f,
  -0.118694f, 0.684873f,  -0.015635f, -0.046469f, 0.075481f,  0.412647f,
  0.454456f,  -0.107169f, 0.775235f,  -0.261629f, -1.194849f, 0.010093f,
  -0.231289f, 0.658286f,  -0.769320f, 0.564545f,  0.482962f,  -0.131378f,
  -0.255844f, -0.078400f, 0.476752f,  0.643001f,
};

static const float av1_tx_split_nn_weights_8x16_layer1[64] = {
  -0.145065f, -0.145101f, 0.174786f,  0.196692f,  0.102025f,  -0.087735f,
  0.386353f,  -0.660539f, -0.183940f, 0.490045f,  -0.276404f, -0.145669f,
  0.209846f,  -0.085574f, -0.156821f, -0.377450f, -0.950010f, 0.450709f,
  -0.108545f, -0.261181f, 1.435606f,  -0.176621f, -1.158548f, 2.035680f,
  0.218069f,  -0.138629f, 0.305958f,  -0.277194f, -0.602468f, 0.203873f,
  0.120720f,  0.216095f,  -0.434502f, -0.579746f, -0.239450f, 0.755529f,
  0.545643f,  0.232091f,  0.330169f,  0.988136f,  -0.070465f, -0.345584f,
  -0.162455f, -0.617064f, 0.123881f,  -0.201098f, 0.222756f,  0.112932f,
  0.048647f,  -0.147890f, 0.394584f,  -0.262148f, 0.280564f,  -0.195432f,
  -0.047515f, 1.133410f,  0.255415f,  -0.299032f, -0.397807f, -0.153246f,
  -0.256734f, 0.177370f,  0.213522f,  -0.530158f,
};

static const float av1_tx_split_nn_bias_8x16_layer1[1] = {
  0.14910713f,
};

static const NN_CONFIG av1_tx_split_nnconfig_8x16 = {
  8,  // num_inputs
  1,  // num_outputs
  1,  // num_hidden_layers
  {
      64,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_8x16_layer0,
      av1_tx_split_nn_weights_8x16_layer1,
  },
  {
      av1_tx_split_nn_bias_8x16_layer0,
      av1_tx_split_nn_bias_8x16_layer1,
  },
};
/******************************************************************************/

// Tx split model for 16x16 block.
static const float av1_tx_split_nn_weights_16x16_layer0[12 * 24] = {
  -0.177215f, -0.297166f, 0.299924f,  0.207878f,  0.216871f,  0.173264f,
  0.295464f,  0.048395f,  0.154731f,  0.305880f,  0.056787f,  -0.166617f,
  0.115653f,  -0.529477f, -0.073995f, -0.211746f, -0.018169f, 0.000788f,
  -0.024940f, -0.007055f, 0.001392f,  0.021678f,  -1.594600f, -0.099593f,
  0.332930f,  0.103574f,  0.158249f,  0.182601f,  0.332665f,  0.226207f,
  -0.139566f, 0.185531f,  0.099074f,  -0.185654f, -0.203121f, -0.285678f,
  -0.313453f, -0.294452f, -0.143707f, -0.031265f, -0.453030f, -0.061874f,
  -0.066150f, -0.099058f, -0.458879f, 0.127544f,  0.338314f,  -0.161350f,
  0.030091f,  -0.075528f, 0.004320f,  0.353690f,  -0.013480f, -0.420402f,
  -0.004659f, -0.329401f, -0.001745f, 0.227384f,  -0.055183f, 0.121405f,
  0.160340f,  0.143603f,  -0.221813f, 0.079107f,  -0.657639f, -0.084348f,
  -0.303414f, 0.046774f,  -0.367679f, 0.060005f,  0.168645f,  0.084421f,
  -0.133625f, 0.301375f,  0.079412f,  -0.419303f, 0.017235f,  0.068637f,
  0.018384f,  -0.428325f, -0.019753f, 0.149444f,  -0.474836f, -0.287162f,
  0.198083f,  0.028292f,  -0.299092f, -0.005849f, -0.256245f, 0.233277f,
  -0.217561f, -0.264003f, 0.269411f,  0.207032f,  -0.339411f, -0.198431f,
  -0.028521f, 0.158076f,  0.177116f,  0.345702f,  -0.145132f, 0.064623f,
  -0.090867f, 0.288816f,  -0.263198f, -0.071028f, -0.044546f, 0.380017f,
  -0.014100f, -0.271192f, -0.318559f, 0.129015f,  -0.050314f, -0.093355f,
  -0.578498f, 0.099090f,  -0.133080f, -0.029975f, -0.059828f, -0.157765f,
  -0.321153f, -0.343671f, -0.242959f, 0.128304f,  0.017170f,  0.072787f,
  -0.475838f, -0.003806f, -0.068615f, 0.150556f,  -0.159903f, -0.416513f,
  0.218794f,  -0.290456f, -0.084569f, -0.170014f, -0.044414f, -0.153069f,
  -0.077329f, -0.089747f, -0.096526f, 0.537952f,  0.134725f,  -0.006469f,
  -0.323335f, -0.168183f, -0.107163f, -0.139954f, 0.011286f,  -0.021712f,
  -0.513992f, 0.259135f,  -0.319808f, 0.077811f,  0.104613f,  0.370571f,
  0.185244f,  0.065530f,  -0.091098f, -0.573741f, 0.111934f,  0.437417f,
  -0.123691f, 0.220641f,  -0.024783f, -0.149460f, -0.354185f, -0.134127f,
  0.038015f,  -0.380596f, 0.250980f,  0.142208f,  0.135170f,  -0.131129f,
  -0.357556f, -0.530945f, 0.159672f,  -0.147025f, -0.377829f, -0.504508f,
  -0.492870f, 0.020753f,  0.142818f,  0.025172f,  0.086140f,  0.091283f,
  0.087491f,  -0.186415f, 0.177785f,  -0.195121f, -1.191148f, -0.477102f,
  0.023371f,  0.227004f,  -0.023502f, -0.242913f, -0.074398f, -0.153480f,
  0.162900f,  0.415509f,  -0.162565f, -0.131709f, -0.258852f, -0.252027f,
  -0.080845f, -0.330274f, 0.021874f,  0.232398f,  0.069277f,  0.220567f,
  -0.024237f, -0.366771f, 0.081673f,  -0.429906f, -0.302170f, 0.061045f,
  0.352777f,  -0.230376f, 0.408153f,  0.064758f,  0.142051f,  0.007219f,
  0.622878f,  0.212577f,  0.036489f,  0.081150f,  -0.284767f, 0.107763f,
  -0.529786f, -0.072190f, -0.300421f, -0.287959f, -0.568900f, 0.011547f,
  -0.131696f, -0.356854f, -0.587962f, -0.026598f, 0.405829f,  0.057565f,
  0.414265f,  -0.159155f, 0.221456f,  0.146314f,  0.265776f,  -0.006516f,
  0.473978f,  -0.186431f, 0.288672f,  -0.060437f, 0.083380f,  -0.205641f,
  0.360016f,  0.222041f,  0.420011f,  0.024579f,  0.377546f,  0.250380f,
  -0.069900f, 0.296743f,  0.073532f,  -0.243225f, -0.374987f, -0.387288f,
  -0.237255f, -0.287013f, 0.417831f,  -0.252988f, -0.257652f, -0.066775f,
  -0.253926f, 0.057841f,  0.346133f,  -0.157797f, -0.406028f, -0.286893f,
  0.274507f,  -0.452561f, 0.143381f,  -0.097755f, 0.021242f,  0.034561f,
  0.044115f,  0.004065f,  0.066729f,  0.043558f,  0.102991f,  -0.477574f,
};

static const float av1_tx_split_nn_bias_16x16_layer0[24] = {
  -0.479033f, 1.467402f,  -0.366291f, 0.372511f,  0.715322f,  -0.605500f,
  0.176848f,  0.032318f,  0.237429f,  -0.046047f, 0.452082f,  0.451805f,
  -0.822845f, 0.636762f,  -0.057350f, 1.163978f,  0.728287f,  0.603654f,
  -0.245519f, -0.893569f, -1.428185f, 0.808870f,  -0.076159f, 1.231976f,
};

static const float av1_tx_split_nn_weights_16x16_layer1[24] = {
  -0.176161f, 1.670188f, -0.180755f, -0.321326f, 0.249728f,  -0.170504f,
  -0.538432f, 0.033893f, 0.149842f,  0.404140f,  -0.377812f, 0.338838f,
  -0.176091f, 0.249844f, -0.362533f, 1.412460f,  0.196862f,  0.278194f,
  -0.140444f, 0.297746f, 0.172533f,  0.116470f,  -0.151656f, -0.603250f,
};

static const float av1_tx_split_nn_bias_16x16_layer1[1] = {
  0.184803f,
};

static const NN_CONFIG av1_tx_split_nnconfig_16x16 = {
  12,  // num_inputs
  1,   // num_outputs
  1,   // num_hidden_layers
  {
      24,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_16x16_layer0,
      av1_tx_split_nn_weights_16x16_layer1,
  },
  {
      av1_tx_split_nn_bias_16x16_layer0,
      av1_tx_split_nn_bias_16x16_layer1,
  },
};
/******************************************************************************/

// Tx split model for 32x32 block.
static const float av1_tx_split_nn_weights_32x32_layer0[12 * 32] = {
  -0.439303f, 0.004813f,  -0.365052f, -0.116868f, -0.356716f, -0.196537f,
  -0.196770f, -0.076096f, 0.357004f,  -0.044909f, -0.112910f, -0.129081f,
  0.156725f,  -0.386346f, 0.038971f,  0.160696f,  0.204923f,  -0.384333f,
  -0.319546f, 0.028179f,  -0.250524f, -0.289669f, -0.284138f, -0.258963f,
  -0.180854f, -0.000807f, -0.029620f, -0.353134f, 0.212408f,  0.141414f,
  0.303016f,  0.098066f,  0.482455f,  0.036069f,  -0.166279f, 0.210119f,
  -0.086337f, -0.023550f, -0.250796f, -0.183945f, -0.393856f, 0.170608f,
  -0.306403f, 0.026318f,  -0.277296f, 0.092684f,  -0.033584f, -0.018371f,
  -0.025043f, -0.257659f, -0.139163f, -0.206949f, -0.190105f, 0.028053f,
  0.361851f,  -0.364726f, -0.096771f, -0.184166f, -0.433228f, -0.182191f,
  -0.097051f, 0.259172f,  0.016432f,  0.259358f,  0.145059f,  0.037196f,
  0.091581f,  -0.219644f, 0.140384f,  -0.446837f, -0.234531f, 0.149508f,
  -0.083429f, 0.186189f,  -0.099890f, -0.111277f, 0.495214f,  0.085053f,
  -0.266613f, -0.051366f, 0.148593f,  0.111875f,  0.077787f,  -0.371653f,
  -0.146157f, -0.229235f, 0.076203f,  0.488975f,  0.096771f,  -0.009483f,
  0.192985f,  0.246273f,  -0.192671f, -0.557890f, -0.292650f, -0.088907f,
  -0.106892f, -0.329659f, 0.012105f,  -0.359326f, 0.170723f,  -0.004357f,
  0.171593f,  -0.478768f, -0.236016f, -0.035077f, 0.133731f,  0.137962f,
  -0.397926f, -0.155164f, -0.276709f, -0.186602f, -0.258301f, 0.036965f,
  -0.649359f, 0.127605f,  0.097930f,  0.182775f,  -0.313324f, 0.053349f,
  0.204203f,  -0.222948f, -0.059008f, -0.049759f, -0.056848f, 0.087497f,
  -0.039987f, -0.055042f, -0.041623f, -0.078424f, -0.317291f, -0.191398f,
  0.632147f,  0.221825f,  0.268394f,  -0.096357f, 0.442545f,  -0.007117f,
  -0.036125f, 0.000525f,  0.088092f,  -0.203653f, 0.086925f,  0.439141f,
  0.329889f,  -0.370050f, -0.194306f, -0.207430f, 0.132779f,  -0.217614f,
  -0.039444f, -0.053019f, -0.260725f, -0.116563f, -0.271048f, 0.283737f,
  -0.007300f, 0.062257f,  -0.347865f, -0.296767f, -0.359123f, 0.230459f,
  -0.189117f, -0.087622f, -0.561091f, 0.184182f,  -0.044980f, 0.012643f,
  0.241672f,  0.050272f,  -0.204851f, -0.159285f, -0.064081f, -0.118666f,
  -0.269471f, 0.231668f,  0.135749f,  -0.131162f, 0.062760f,  0.100949f,
  0.074967f,  -0.056918f, 0.251707f,  0.034098f,  0.341290f,  -0.105027f,
  0.313246f,  -0.092679f, -0.014632f, -0.390967f, 0.136881f,  -0.241554f,
  0.097674f,  0.110832f,  -0.390245f, 0.017654f,  -0.506222f, 0.065252f,
  0.244834f,  -0.171352f, -0.331702f, 0.111043f,  0.125217f,  -0.058116f,
  -0.382595f, -0.052545f, 0.114261f,  -0.493617f, 0.243984f,  -0.171053f,
  0.165009f,  -0.063020f, 0.096502f,  0.341339f,  -0.013443f, 0.056372f,
  0.339284f,  0.398376f,  0.389409f,  0.257252f,  0.517368f,  0.078856f,
  0.087716f,  -0.171092f, 0.227461f,  0.125307f,  -0.054423f, -0.143161f,
  0.224041f,  -0.086477f, -0.092548f, 0.072392f,  -0.061608f, 0.258347f,
  0.147033f,  -0.478244f, -0.204869f, 0.038552f,  -0.144563f, 0.224087f,
  -0.296705f, 0.153889f,  -0.064624f, 0.085265f,  -0.103826f, 0.127971f,
  0.019965f,  0.111937f,  -0.074187f, -0.029518f, -0.127305f, -0.012210f,
  0.042714f,  0.070052f,  -0.202360f, 0.348144f,  -0.132097f, -0.209585f,
  -0.248286f, -0.065774f, -0.089482f, -0.133226f, 0.325430f,  -0.013468f,
  -0.406090f, -0.144936f, 0.208620f,  0.343445f,  -0.059639f, 0.114857f,
  -0.069431f, -0.218725f, 0.190575f,  -0.368101f, 0.030030f,  0.062815f,
  -0.239369f, -0.537852f, 0.022487f,  0.023038f,  0.190788f,  0.040123f,
  -0.004304f, 0.060749f,  -0.108929f, 0.136796f,  -0.542875f, -0.227074f,
  -0.182244f, 0.082559f,  0.019149f,  0.178854f,  0.120284f,  0.009070f,
  0.068268f,  -0.544822f, 0.120536f,  0.354028f,  -0.119890f, -0.122055f,
  -0.405335f, 0.122341f,  -0.304412f, 0.062405f,  -0.302568f, -0.276505f,
  -0.120915f, -0.221841f, 0.282007f,  -0.253971f, 0.059517f,  -0.144976f,
  0.149391f,  -0.047355f, -0.167742f, -0.392333f, -0.041132f, 0.342135f,
  0.017485f,  0.021038f,  -0.023728f, -0.192181f, -0.103996f, 0.092873f,
  -0.114365f, -0.397732f, -0.065421f, 0.053084f,  0.035201f,  0.053019f,
  -0.105377f, -0.039500f, 0.131904f,  -0.123911f, -0.390328f, -0.125198f,
  -0.000126f, 0.014864f,  -0.220187f, 0.084056f,  -0.492155f, -0.164979f,
  0.133592f,  0.121519f,  -0.240813f, 0.186680f,  0.118673f,  0.235006f,
  -0.239894f, -0.185759f, -0.336992f, 0.209620f,  -0.298845f, 0.127803f,
  -0.083992f, 0.194340f,  -0.245378f, 0.212308f,  0.142512f,  -0.163324f,
  0.383495f,  0.291065f,  0.286620f,  -0.239957f, 0.225127f,  -0.174424f,
  0.297231f,  -0.045434f, 0.156444f,  -0.184273f, -0.204567f, 0.202551f,
  0.370019f,  -0.073910f, 0.344897f,  0.063100f,  0.338547f,  -0.099145f,
  0.391863f,  -0.214244f, -0.241734f, -0.281851f, -0.035133f, -0.153157f,
};

static const float av1_tx_split_nn_bias_32x32_layer0[32] = {
  0.143343f,  -0.021982f, -0.314939f, 0.170867f,  -0.081248f, 0.125758f,
  -0.355762f, 0.279798f,  1.027712f,  -0.434660f, 1.072005f,  0.668893f,
  -0.031216f, -0.528650f, 0.328349f,  0.543645f,  -0.188810f, 0.221110f,
  -1.638637f, 0.058045f,  -1.731105f, -0.444284f, 0.513693f,  0.890025f,
  0.160288f,  0.393312f,  0.332856f,  -0.080767f, 0.299822f,  0.235876f,
  0.254942f,  -0.017796f,
};

static const float av1_tx_split_nn_weights_32x32_layer1[32] = {
  -0.090326f, -0.267553f, -0.026071f, 0.100912f,  0.279137f,  0.079064f,
  -0.074885f, 0.053804f,  0.736810f,  -0.031693f, -0.970514f, 0.174069f,
  0.095940f,  -0.065047f, 0.052911f,  0.176728f,  -0.058274f, 0.148364f,
  -0.162210f, 0.093875f,  -0.367663f, 0.020876f,  0.137280f,  -1.099116f,
  0.146854f,  0.075590f,  0.228534f,  0.141993f,  0.072143f,  0.101421f,
  -0.068547f, -0.154148f,
};

static const float av1_tx_split_nn_bias_32x32_layer1[1] = {
  0.316622f,
};

static const NN_CONFIG av1_tx_split_nnconfig_32x32 = {
  12,  // num_inputs
  1,   // num_outputs
  1,   // num_hidden_layers
  {
      32,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_32x32_layer0,
      av1_tx_split_nn_weights_32x32_layer1,
  },
  {
      av1_tx_split_nn_bias_32x32_layer0,
      av1_tx_split_nn_bias_32x32_layer1,
  },
};
/******************************************************************************/

// Tx split model for 64x64 block.
static const float av1_tx_split_nn_weights_64x64_layer0[12 * 32] = {
  -0.006828f, 0.149944f,  -0.017614f, -0.044599f, -0.024517f, 0.507698f,
  0.001039f,  0.037164f,  0.015091f,  -0.306620f, -0.162047f, -0.369440f,
  0.396310f,  0.087121f,  0.208609f,  -0.083068f, 0.493774f,  0.217682f,
  0.377393f,  0.172879f,  0.397422f,  0.078919f,  0.741350f,  0.064169f,
  -0.099989f, -0.192983f, -0.278230f, -0.310048f, -0.439965f, -0.226698f,
  -0.436596f, -0.007551f, -0.396721f, 0.153570f,  -0.190838f, -0.071869f,
  0.048799f,  -0.301301f, -0.005015f, 0.500480f,  -0.030622f, -0.559095f,
  -0.032634f, -0.054160f, -0.056979f, -0.456545f, 0.306536f,  -0.411323f,
  -0.005366f, -0.069496f, 0.019990f,  0.327931f,  -0.002516f, 0.393190f,
  0.001759f,  0.035093f,  -0.030302f, -0.528984f, 0.174781f,  0.241462f,
  -0.415427f, -0.164502f, 0.143065f,  -0.122595f, 0.082049f,  -0.143346f,
  0.055642f,  -0.124701f, 0.004050f,  -0.216235f, -2.681730f, 0.101658f,
  0.381239f,  0.465936f,  0.331154f,  0.301708f,  -0.360171f, 0.054886f,
  -0.118658f, 0.287921f,  0.277859f,  0.203784f,  0.247809f,  0.656924f,
  -0.354628f, 0.315081f,  0.105108f,  -0.510179f, 0.059267f,  0.061386f,
  0.076423f,  0.347119f,  0.100134f,  0.028402f,  -0.118621f, -0.238689f,
  0.080141f,  -0.138863f, 0.009009f,  -0.100526f, -0.138875f, 0.066992f,
  0.005949f,  0.564336f,  0.046994f,  0.004655f,  0.366047f,  0.014695f,
  -0.146928f, -0.024665f, -0.440357f, -0.109395f, 0.527231f,  -0.020925f,
  -0.227236f, -0.068141f, 0.282009f,  0.040192f,  -0.267100f, 0.229228f,
  0.133861f,  0.338706f,  -0.030178f, -0.040919f, -0.026343f, -0.330338f,
  -0.066931f, -0.110580f, -0.072056f, 0.599457f,  -0.020738f, 0.169200f,
  0.836240f,  -0.157548f, 0.386273f,  0.002404f,  0.329410f,  -0.007020f,
  0.351705f,  -0.041259f, 0.388861f,  0.003899f,  0.582627f,  0.023572f,
  0.409912f,  -0.158472f, 0.536383f,  0.525093f,  0.604247f,  0.439159f,
  0.692832f,  0.046272f,  0.590367f,  -0.082166f, 0.262357f,  0.478671f,
  0.031935f,  0.042675f,  0.120002f,  0.398616f,  -0.078967f, 0.227986f,
  -0.044679f, 0.151061f,  -0.085564f, 0.220205f,  -0.265606f, -0.203623f,
  0.204719f,  -0.125922f, 0.038544f,  -0.269379f, 0.025866f,  0.109967f,
  0.019064f,  -0.237297f, -0.309746f, -0.329118f, -0.278368f, -0.063859f,
  0.278496f,  0.018620f,  0.209971f,  0.296250f,  0.142850f,  0.288689f,
  0.137084f,  0.130517f,  0.128171f,  -0.155396f, -0.008449f, -0.099845f,
  0.173455f,  -0.059909f, -0.147318f, 0.102851f,  -0.251389f, -0.001448f,
  0.103907f,  0.297273f,  -0.027846f, 0.028260f,  -0.382601f, 0.346695f,
  -0.601641f, 0.162366f,  -0.477495f, -0.042731f, -0.387871f, -0.051791f,
  -0.401498f, -0.048446f, -0.456270f, -0.062287f, 0.493919f,  0.003008f,
  0.099917f,  -0.358525f, -0.094903f, -0.022811f, -0.062259f, 0.019455f,
  -0.050644f, 0.020041f,  -0.132912f, -0.061578f, -3.083691f, -0.014961f,
  -0.129115f, -0.710559f, 0.157213f,  -0.844037f, -0.121991f, -0.943386f,
  -0.231269f, -0.003462f, 0.331478f,  -0.132703f, -1.285993f, -0.120957f,
  -0.373755f, -0.322609f, 0.309059f,  -0.131523f, -0.118334f, -0.063805f,
  -0.104251f, 0.012166f,  -0.094699f, -0.283753f, 0.128168f,  -0.526929f,
  -0.050331f, 0.186153f,  0.005913f,  -0.221236f, 0.036363f,  0.160909f,
  -0.001342f, -0.382749f, 0.037820f,  0.281689f,  -0.024275f, 0.028854f,
  0.318291f,  0.318526f,  0.035778f,  0.034031f,  0.189663f,  -0.293367f,
  0.082022f,  0.127923f,  0.078866f,  -0.081361f, -0.268117f, 0.246675f,
  0.248605f,  -0.215479f, -0.073084f, 0.496140f,  -0.067327f, 0.396237f,
  -0.120739f, 0.033752f,  -0.044120f, -0.218941f, -0.028078f, 0.195132f,
  -0.040400f, 0.281604f,  -0.100471f, 0.415207f,  -0.258503f, -0.429749f,
  0.150569f,  -0.010859f, 0.136448f,  0.026589f,  0.148466f,  0.110764f,
  0.380967f,  0.009177f,  0.103075f,  0.116417f,  0.226273f,  -0.327746f,
  0.169346f,  0.284553f,  -0.094986f, 0.312745f,  -0.147840f, 0.025062f,
  -0.494482f, 0.112388f,  -0.213962f, 0.107050f,  -0.433371f, -0.096276f,
  -0.244835f, -0.003518f, -0.459148f, -0.145080f, 0.017150f,  0.042846f,
  -0.237479f, 0.104746f,  0.158677f,  0.358937f,  0.099921f,  0.277109f,
  0.012410f,  -0.062897f, 0.116130f,  0.255309f,  0.341628f,  0.145002f,
  -0.429344f, -0.016433f, -0.068985f, 0.285194f,  -0.286719f, -0.018298f,
  -0.179369f, -0.194655f, -0.165380f, 0.026071f,  -0.428268f, -0.379929f,
  -0.727543f, 0.179610f,  -0.963979f, -0.042026f, -0.616202f, 0.133401f,
  -0.784966f, 0.061205f,  -0.713357f, 0.129795f,  0.120512f,  -0.339545f,
  0.353557f,  0.114906f,  -0.329813f, -0.209987f, 0.085410f,  0.214313f,
  -0.122082f, 0.335770f,  -0.020937f, 0.202456f,  0.289023f,  -0.421186f,
  0.337905f,  0.407663f,  0.132771f,  0.071734f,  0.213914f,  0.128595f,
  0.302659f,  -0.209501f, 0.217756f,  0.253079f,  -0.089505f, -0.205614f,
};

static const float av1_tx_split_nn_bias_64x64_layer0[32] = {
  0.296914f,  -1.826816f, 0.346130f,  0.969520f,  -0.528154f, 1.175862f,
  -0.075985f, -0.097323f, -0.233059f, 0.004846f,  0.401279f,  -2.272435f,
  0.086257f,  0.414162f,  -0.194786f, -0.233887f, -0.113215f, -2.453546f,
  0.861214f,  0.298361f,  0.267397f,  -0.158557f, -0.119911f, -0.098134f,
  -0.339263f, 0.385871f,  -0.678123f, 0.263218f,  0.251611f,  -1.155773f,
  -0.365437f, 0.229255f,
};

static const float av1_tx_split_nn_weights_64x64_layer1[32] = {
  0.502104f,  -0.708023f, 0.419648f,  1.583418f,  0.419355f,  -1.462981f,
  -0.439623f, 0.405691f,  0.823257f,  0.061654f,  0.750875f,  0.775031f,
  -0.387909f, 0.447385f,  0.284690f,  0.353262f,  -0.224347f, 0.832864f,
  -1.708491f, -1.042447f, -0.272829f, 0.540640f,  0.310509f,  0.723745f,
  0.245592f,  -0.218417f, -0.597987f, -0.362301f, 0.702217f,  -0.692614f,
  0.207812f,  0.513560f,
};

static const float av1_tx_split_nn_bias_64x64_layer1[1] = { -0.2307045f };

static const NN_CONFIG av1_tx_split_nnconfig_64x64 = {
  12,  // num_inputs
  1,   // num_outputs
  1,   // num_hidden_layers
  {
      32,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_64x64_layer0,
      av1_tx_split_nn_weights_64x64_layer1,
  },
  {
      av1_tx_split_nn_bias_64x64_layer0,
      av1_tx_split_nn_bias_64x64_layer1,
  },
};
/******************************************************************************/

// Tx split model for 4x16 block.
static const float av1_tx_split_nn_weights_4x16_layer0[8 * 16] = {
  -1.344184f, -1.454625f, -0.703110f, -0.140570f, -0.841536f, -0.068131f,
  -2.128968f, -0.655518f, 0.432180f,  0.879752f,  -0.222211f, 0.061615f,
  -0.230969f, 0.569496f,  1.424188f,  0.598063f,  -0.436005f, -0.737606f,
  -0.137875f, -0.085730f, -0.076512f, -0.583101f, -0.937377f, -0.203556f,
  -0.215797f, -0.015361f, -0.124098f, -0.411917f, 0.340441f,  -0.331752f,
  -0.472607f, -0.097714f, -0.930572f, -1.354713f, -0.550724f, 0.176212f,
  -0.636060f, 0.183271f,  -0.610212f, 0.345895f,  -1.100906f, -1.605713f,
  0.111888f,  -0.140937f, 0.063013f,  -0.013315f, -0.273472f, -0.255870f,
  1.200328f,  0.274002f,  1.005776f,  0.322392f,  1.222373f,  0.158227f,
  0.408810f,  0.145022f,  0.139842f,  -1.249412f, 0.286672f,  -0.635699f,
  0.312562f,  -0.495606f, -1.117034f, -0.085107f, -0.097484f, -0.341521f,
  -0.132199f, -0.863055f, 0.217579f,  -1.161425f, -0.302087f, -1.357271f,
  -0.520724f, -1.211069f, -1.048729f, -0.333087f, -1.171527f, -0.280824f,
  -2.057684f, -0.228755f, 0.606278f,  0.101198f,  -0.314847f, -1.303255f,
  -0.294964f, 1.301923f,  0.041712f,  0.077593f,  -1.152746f, 0.495315f,
  -0.751566f, 0.230249f,  -0.840661f, 0.100731f,  1.346269f,  0.649898f,
  -1.432258f, -0.456710f, -1.018123f, -0.348559f, -1.225226f, -0.170717f,
  -0.354072f, 0.068292f,  -0.234168f, 0.277503f,  0.179134f,  0.907420f,
  0.354626f,  -0.627210f, 0.905779f,  0.512612f,  0.161190f,  -0.843177f,
  0.014953f,  -0.354983f, 0.011116f,  -0.429598f, -1.017138f, -0.211432f,
  0.941840f,  -0.281747f, 0.957776f,  -0.541914f, 1.041880f,  -0.433580f,
  -1.416451f, -0.166467f,
};

static const float av1_tx_split_nn_bias_4x16_layer0[16] = {
  3.086118f,  -3.235095f, 4.830956f,  -0.165706f, 0.955031f,  4.055783f,
  -0.311489f, 4.660205f,  -0.576277f, -0.248111f, -0.790519f, -1.686412f,
  -1.191704f, -3.800073f, 4.121552f,  -1.399397f,
};

static const float av1_tx_split_nn_weights_4x16_layer1[16] = {
  -0.758677f, 0.388776f,  0.439906f,  0.011390f, -0.084319f, -0.667969f,
  -0.467316f, -0.875491f, -0.160668f, 0.805292f, 0.114393f,  -0.549682f,
  0.462109f,  0.343315f,  1.092593f,  0.483152f,
};

static const float av1_tx_split_nn_bias_4x16_layer1[1] = {
  0.8205083f,
};

static const NN_CONFIG av1_tx_split_nnconfig_4x16 = {
  8,  // num_inputs
  1,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_4x16_layer0,
      av1_tx_split_nn_weights_4x16_layer1,
  },
  {
      av1_tx_split_nn_bias_4x16_layer0,
      av1_tx_split_nn_bias_4x16_layer1,
  },
};
/******************************************************************************/

// Tx split model for 16x32 block.
static const float av1_tx_split_nn_weights_16x32_layer0[8 * 32] = {
  0.180713f,  0.033211f,  0.607561f,  0.138642f,  0.637204f,  -0.000940f,
  0.012630f,  0.358109f,  0.022238f,  0.190418f,  0.079088f,  0.065925f,
  0.038242f,  0.162380f,  -0.122728f, 0.379382f,  -0.303283f, -0.327550f,
  0.029120f,  -0.284553f, 0.269588f,  -0.309805f, -0.241036f, -0.161103f,
  -0.304887f, 0.239843f,  -0.149146f, 0.311234f,  -0.073640f, -0.132718f,
  0.178901f,  0.474712f,  0.020280f,  0.063685f,  -0.609170f, -0.013658f,
  -0.338074f, 0.250429f,  0.082978f,  -0.186315f, -0.788959f, 0.039859f,
  -0.426461f, -0.001524f, -0.447211f, 0.378102f,  0.315617f,  0.017428f,
  0.745494f,  -0.219024f, 0.512836f,  0.200522f,  0.680449f,  0.313686f,
  -0.412569f, -0.132927f, 0.631120f,  0.042735f,  0.336153f,  0.044772f,
  0.432606f,  0.175681f,  -0.634411f, -0.073509f, -0.040643f, -0.559260f,
  -0.104034f, -0.570495f, -0.247365f, 0.063256f,  -0.582021f, -0.492585f,
  -0.194955f, -0.207934f, -0.506627f, 0.021743f,  -0.416518f, 0.320876f,
  0.115889f,  0.149399f,  -0.229376f, 0.095505f,  0.115191f,  -0.471921f,
  0.113068f,  0.343684f,  -0.036831f, 0.021240f,  0.295112f,  0.031166f,
  0.448201f,  -0.132241f, 0.164032f,  0.355572f,  0.072154f,  0.017335f,
  -0.046113f, 0.178719f,  -0.026881f, -0.242590f, 0.055073f,  -0.012958f,
  0.077904f,  0.351356f,  0.107655f,  0.260568f,  -0.080052f, -0.197553f,
  0.085763f,  0.263416f,  -0.327741f, 0.158855f,  0.056899f,  -0.162121f,
  0.339518f,  -0.571204f, 0.264966f,  -0.252214f, -0.202560f, -0.134213f,
  -0.330188f, 0.009470f,  -0.468376f, -0.065240f, -0.307957f, 0.116479f,
  -0.222238f, -0.458716f, 0.186493f,  -0.391415f, 0.118649f,  -0.104653f,
  -0.259958f, -0.332081f, -0.403785f, -0.050147f, -0.573511f, 0.177117f,
  -0.598358f, 0.164947f,  -0.119694f, -0.058520f, 0.203829f,  -0.267404f,
  -0.048202f, -0.600006f, 0.181594f,  -0.731805f, 0.146417f,  -0.687148f,
  -1.210525f, -0.450101f, -0.620635f, 0.208825f,  -0.611357f, 0.112202f,
  -0.309468f, -0.323545f, 0.357770f,  0.308061f,  0.553199f,  0.049012f,
  0.530093f,  -0.208597f, 0.607882f,  -0.058120f, -0.527634f, 0.018136f,
  0.060753f,  0.118894f,  0.175649f,  0.014731f,  0.428318f,  -0.106465f,
  -0.119077f, 0.080179f,  0.524997f,  0.368286f,  0.528286f,  0.213659f,
  0.639286f,  0.195079f,  -0.049815f, -0.092008f, -0.302958f, 0.298149f,
  -0.173870f, -0.145205f, -0.233589f, -0.303368f, 0.141275f,  0.325622f,
  -0.115293f, 0.155188f,  0.047225f,  0.231050f,  -0.167447f, 0.349754f,
  0.295544f,  -0.319466f, 0.095144f,  0.174612f,  -0.194652f, 0.305915f,
  -0.239008f, -0.037453f, 0.280696f,  0.125850f,  0.749196f,  -0.101919f,
  0.791808f,  -0.236811f, 0.064157f,  0.032865f,  -0.225911f, 0.350384f,
  0.723183f,  -0.103992f, 0.483085f,  -0.123992f, 0.602138f,  0.023895f,
  -0.692601f, -0.118387f, 0.162527f,  0.145178f,  -0.184702f, -0.017753f,
  -0.159436f, 0.124105f,  -0.131067f, 0.310275f,  0.151499f,  0.138924f,
  0.537459f,  0.263212f,  0.615896f,  0.281255f,  0.021293f,  -0.473459f,
  0.210145f,  -0.056682f, 0.063658f,  0.377254f,  -0.314410f, -0.183487f,
  0.300384f,  0.328471f,  0.164694f,  -0.159272f, -0.160942f, -0.502861f,
  -0.129147f, 0.045916f,  -0.606865f, -0.101378f,
};

static const float av1_tx_split_nn_bias_16x32_layer0[32] = {
  0.051664f,  -0.212487f, -0.077596f, -0.818467f, 0.638475f,  -0.759937f,
  0.157198f,  0.989640f,  1.586035f,  0.431144f,  0.041605f,  0.543085f,
  0.498379f,  0.320504f,  0.134233f,  0.670979f,  -0.105562f, -1.574879f,
  1.261812f,  -0.287530f, -1.610592f, 0.730899f,  -0.894240f, -0.657790f,
  0.270806f,  -0.181708f, 0.298578f,  0.817240f,  -0.221508f, -0.201771f,
  -0.294389f, 1.456413f,
};

static const float av1_tx_split_nn_weights_16x32_layer1[32] = {
  1.208914f,  0.324728f,  0.383352f,  -0.874321f, 0.172565f,  -0.580927f,
  -0.432927f, 0.433698f,  -0.801935f, 0.672028f,  0.563493f,  0.260077f,
  -0.200557f, -0.121638f, 0.530735f,  -0.525196f, 0.281799f,  0.624204f,
  -0.662775f, -0.230887f, 0.980989f,  0.223437f,  -0.790591f, 0.600724f,
  -0.273445f, 0.427635f,  -0.501641f, -0.878390f, 0.234731f,  -0.172550f,
  0.418904f,  1.792187f,
};

static const float av1_tx_split_nn_bias_16x32_layer1[1] = {
  -0.29233751f,
};

static const NN_CONFIG av1_tx_split_nnconfig_16x32 = {
  8,  // num_inputs
  1,  // num_outputs
  1,  // num_hidden_layers
  {
      32,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_16x32_layer0,
      av1_tx_split_nn_weights_16x32_layer1,
  },
  {
      av1_tx_split_nn_bias_16x32_layer0,
      av1_tx_split_nn_bias_16x32_layer1,
  },
};
/******************************************************************************/

// Tx split model for 32x64 block.
static const float av1_tx_split_nn_weights_32x64_layer0[8 * 32] = {
  0.031614f,  -0.110926f, 0.052418f,  -0.702506f, 0.045708f,  0.238329f,
  -0.021806f, -0.208128f, 0.509745f,  -0.293891f, 0.277788f,  0.113937f,
  0.741576f,  0.062848f,  0.351878f,  0.212532f,  0.385842f,  0.081517f,
  0.398502f,  -0.015156f, 0.242616f,  0.214619f,  -0.182678f, -0.170546f,
  0.110605f,  -0.236749f, -0.023831f, -0.285243f, 0.147156f,  -0.257639f,
  0.341355f,  -0.571641f, -0.721797f, 0.139588f,  -0.518494f, -0.206526f,
  -0.570560f, -0.184295f, 0.110271f,  0.210292f,  -0.109132f, -0.001080f,
  0.129251f,  -0.204230f, -0.396312f, -0.183024f, 0.421243f,  -0.013154f,
  0.222627f,  0.169826f,  0.226037f,  0.218153f,  -0.343528f, 0.274906f,
  -0.156632f, 0.250261f,  -0.484020f, 0.019909f,  -0.349575f, -0.286643f,
  -0.507396f, 0.202446f,  -0.154110f, -0.292644f, 0.122666f,  0.306963f,
  0.424895f,  0.005579f,  0.494094f,  -0.079551f, 0.473740f,  0.352414f,
  -0.356917f, 0.264331f,  -0.554487f, 0.119978f,  0.012291f,  -0.141641f,
  -0.254714f, -0.213723f, -0.116701f, -0.011267f, 0.190025f,  -0.118501f,
  0.305151f,  -0.316782f, -0.220801f, -0.308420f, -0.324285f, 0.421329f,
  -0.177066f, -0.055114f, 0.229698f,  -0.199523f, 0.054278f,  0.365020f,
  -0.060586f, -0.300618f, 0.157563f,  -0.064338f, -0.005711f, -0.176991f,
  -0.424502f, -0.111914f, 0.092608f,  0.126621f,  0.078547f,  0.148008f,
  0.024221f,  0.124599f,  0.001343f,  0.059402f,  0.453753f,  0.047102f,
  0.242544f,  0.055735f,  -0.067451f, -0.170061f, -0.170469f, -0.232173f,
  0.214908f,  0.248889f,  0.544348f,  -0.084566f, 0.402478f,  0.298031f,
  0.099038f,  -0.238019f, -0.475085f, -0.070042f, -0.754955f, -0.049095f,
  -0.783801f, -0.099857f, -0.582008f, -0.055194f, -0.103655f, 0.143689f,
  0.100219f,  0.293934f,  0.099271f,  -0.036320f, 0.356626f,  -0.261445f,
  0.879544f,  0.000878f,  0.532920f,  -0.093918f, 0.508867f,  -0.040215f,
  -0.789042f, -0.145380f, -0.090040f, -0.066636f, 0.015212f,  0.352989f,
  -0.058831f, -0.164588f, 0.039890f,  0.122861f,  0.222508f,  0.061217f,
  0.466487f,  0.022666f,  0.423777f,  -0.002200f, -0.656835f, -0.099760f,
  -0.520606f, 0.303204f,  -0.563620f, -0.160922f, -0.243203f, 0.313354f,
  -0.336516f, -0.206764f, -0.236040f, 0.325899f,  -0.418748f, 0.163205f,
  -0.476242f, -0.121928f, 0.139178f,  -0.157193f, -0.531766f, -0.180202f,
  -0.485254f, 0.187703f,  -0.440072f, 0.137854f,  0.029139f,  0.109530f,
  -0.078475f, -0.360618f, -0.334672f, -0.350890f, -0.403976f, 0.180336f,
  -0.304542f, 0.005123f,  0.413995f,  0.314639f,  0.342648f,  -0.293264f,
  0.358135f,  -0.180425f, -0.369530f, -0.048413f, 0.498366f,  0.121875f,
  0.270948f,  -0.187966f, 0.342503f,  0.174420f,  -0.352105f, 0.088080f,
  0.008277f,  0.020275f,  -0.002381f, 0.504389f,  -0.018832f, -0.366047f,
  -0.090947f, -0.168150f, 0.016184f,  -0.328914f, 0.089579f,  -0.017349f,
  0.005844f,  -0.005010f, -1.857514f, -0.282426f, 0.010177f,  -0.214727f,
  -0.182529f, 0.156943f,  -0.162032f, -0.472654f, 0.069432f,  0.016901f,
  -0.767905f, 0.137129f,  -0.411463f, 0.049056f,  -0.431657f, -0.037641f,
  0.785500f,  0.046225f,  0.195831f,  0.245204f,  0.368614f,  0.212261f,
  0.440626f,  -0.158048f, -0.461031f, -0.146280f,
};

static const float av1_tx_split_nn_bias_32x64_layer0[32] = {
  0.490777f,  -1.894238f, 0.621333f,  -0.076756f, 0.286298f, 0.286375f,
  -0.126431f, -0.350034f, -1.017572f, 0.620125f,  0.408128f, 0.238756f,
  -0.060728f, 0.210912f,  0.043124f,  0.445649f,  0.907025f, 0.360272f,
  1.083101f,  -0.068952f, 1.062348f,  0.396354f,  0.280075f, 0.501732f,
  0.328422f,  0.066241f,  0.474697f,  0.126313f,  0.741206f, 0.314796f,
  0.552712f,  0.299410f,
};

static const float av1_tx_split_nn_weights_32x64_layer1[32] = {
  1.033823f,  0.603439f,  0.304591f,  -0.279940f, -0.780909f, -0.132801f,
  0.154059f,  0.662014f,  -0.718368f, 0.198733f,  0.039766f,  -0.208516f,
  -0.104909f, -0.394209f, 0.081617f,  0.365041f,  -0.874960f, -0.063315f,
  -1.189897f, 0.337225f,  0.410893f,  0.307519f,  0.221323f,  0.233895f,
  0.469536f,  0.438557f,  0.280144f,  0.422423f,  -1.394513f, 0.781900f,
  0.352981f,  0.111265f,
};

static const float av1_tx_split_nn_bias_32x64_layer1[1] = {
  -0.18160765f,
};

static const NN_CONFIG av1_tx_split_nnconfig_32x64 = {
  8,  // num_inputs
  1,  // num_outputs
  1,  // num_hidden_layers
  {
      32,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_32x64_layer0,
      av1_tx_split_nn_weights_32x64_layer1,
  },
  {
      av1_tx_split_nn_bias_32x64_layer0,
      av1_tx_split_nn_bias_32x64_layer1,
  },
};
/******************************************************************************/

// Tx split model for 8x32 block.
static const float av1_tx_split_nn_weights_8x32_layer0[8 * 24] = {
  -0.687846f, 0.121404f,  -0.372905f, 0.126770f,  -0.103298f, -0.101650f,
  -0.148490f, -0.271740f, 0.682915f,  -0.079765f, 0.634347f,  -0.151503f,
  0.287692f,  -0.079072f, -0.236948f, 0.065064f,  0.713383f,  0.397123f,
  0.553621f,  0.368529f,  0.767663f,  -0.046601f, -0.392402f, -0.294822f,
  -0.292325f, -0.010573f, -0.837945f, 0.050113f,  -0.811360f, 0.199162f,
  0.150832f,  0.011602f,  0.369694f,  -0.225876f, 0.234113f,  -0.269808f,
  0.303805f,  -0.190281f, -0.451136f, 0.209755f,  -0.308894f, 0.326956f,
  0.313591f,  0.089923f,  -0.095754f, 0.390981f,  0.467366f,  0.169670f,
  0.853322f,  0.054055f,  0.830319f,  -0.121918f, 0.262019f,  -0.093526f,
  0.385558f,  0.419174f,  0.040198f,  -0.347030f, -0.450492f, -0.106764f,
  0.487502f,  -0.204188f, 0.430374f,  -0.116388f, 0.236407f,  -0.157376f,
  0.732294f,  -0.651387f, 0.347446f,  0.342575f,  0.048406f,  0.187657f,
  0.434899f,  -0.447782f, 0.032728f,  -0.071168f, -0.255327f, 0.104174f,
  0.095689f,  -0.431743f, 0.725694f,  0.031797f,  0.523171f,  0.061801f,
  0.469804f,  -0.071068f, -0.059024f, -0.211937f, 0.392134f,  -0.321490f,
  0.366060f,  -0.427798f, 0.166771f,  0.299652f,  0.044660f,  0.205142f,
  0.039133f,  -0.051835f, -0.465475f, 0.216976f,  -0.341156f, 0.095358f,
  0.230807f,  0.201674f,  0.279266f,  -0.713534f, -0.091690f, -0.569708f,
  -0.119001f, 0.252160f,  -1.544578f, -0.284477f, 0.555348f,  0.226471f,
  0.347690f,  0.034365f,  0.770835f,  -0.241859f, -0.130241f, 0.292936f,
  0.396622f,  -0.417916f, 0.492224f,  0.125517f,  0.344824f,  0.232172f,
  -0.432106f, -0.278745f, 0.035069f,  -0.307247f, -0.120760f, 0.170950f,
  0.433601f,  0.044286f,  0.141463f,  -0.041382f, 0.529346f,  0.010868f,
  -0.323674f, 0.185205f,  0.623459f,  0.232842f,  -0.406693f, -0.142944f,
  0.222988f,  0.343634f,  0.065401f,  0.002621f,  0.805335f,  -0.426926f,
  0.279181f,  0.131364f,  0.192339f,  -0.402391f, 0.544120f,  -0.060618f,
  0.467780f,  0.165224f,  -0.373131f, 0.002427f,  0.688064f,  0.322317f,
  0.259713f,  0.130583f,  0.185032f,  -0.189111f, -0.067821f, 0.010875f,
  0.644724f,  -0.179291f, 0.463222f,  0.155230f,  0.721384f,  -0.046019f,
  0.438501f,  0.440027f,  -0.462090f, -0.002039f, -0.468026f, -0.008890f,
  -0.328530f, 0.370102f,  0.482531f,  0.043471f,  -0.469732f, -0.532663f,
  0.122081f,  -0.379659f, 0.037219f,  -0.519913f, -0.128975f, -0.404365f,
};

static const float av1_tx_split_nn_bias_8x32_layer0[24] = {
  -1.198965f, 0.395204f,  -0.408627f, -0.021654f, -0.658355f, 0.154525f,
  -0.288354f, 1.207574f,  0.411608f,  0.964678f,  -1.176893f, 1.059006f,
  -0.472969f, 2.087975f,  1.065536f,  0.595569f,  0.197907f,  -0.349938f,
  1.013651f,  -0.931093f, -0.973595f, -0.459094f, -1.253062f, 1.624782f,
};

static const float av1_tx_split_nn_weights_8x32_layer1[24] = {
  0.815787f,  -0.393465f, -0.483427f, -0.565592f, 0.493494f,  0.430229f,
  -0.507073f, -0.251379f, -0.353418f, -0.495445f, 0.820029f,  0.649146f,
  -0.487383f, 1.844503f,  0.480324f,  -0.982705f, -0.501446f, -0.220584f,
  0.334299f,  0.802238f,  0.805838f,  -0.487848f, 0.300772f,  -1.232857f,
};

static const float av1_tx_split_nn_bias_8x32_layer1[1] = {
  0.13435879f,
};

static const NN_CONFIG av1_tx_split_nnconfig_8x32 = {
  8,  // num_inputs
  1,  // num_outputs
  1,  // num_hidden_layers
  {
      24,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_8x32_layer0,
      av1_tx_split_nn_weights_8x32_layer1,
  },
  {
      av1_tx_split_nn_bias_8x32_layer0,
      av1_tx_split_nn_bias_8x32_layer1,
  },
};
/******************************************************************************/

// Tx split model for 16x32 block.
static const float av1_tx_split_nn_weights_16x64_layer0[8 * 16] = {
  -0.378223f, -0.124216f, -0.514089f, -0.110117f, -0.585801f, -0.094838f,
  -0.455385f, -0.220254f, -0.504568f, -0.082351f, -0.476420f, -0.253993f,
  -0.454709f, -0.059461f, 0.210313f,  -0.155683f, 0.192968f,  -0.127804f,
  0.471996f,  0.253377f,  0.472625f,  0.485322f,  0.150560f,  0.164868f,
  -0.475587f, 0.447559f,  -0.455759f, -0.306665f, -0.194866f, -0.283716f,
  -0.243897f, 0.293020f,  -0.308298f, -0.191904f, -0.468568f, 0.014053f,
  -0.618848f, 0.096273f,  -0.444586f, 0.347750f,  -0.280643f, -0.062872f,
  0.118661f,  0.540099f,  0.104141f,  -0.279300f, -0.098721f, -0.173427f,
  -0.984558f, -0.424559f, -0.411928f, -0.120875f, -0.488999f, -0.050716f,
  -0.523103f, 0.093620f,  -0.930396f, -0.431997f, -1.163297f, 0.190384f,
  -0.422581f, -0.005354f, 0.450552f,  0.369210f,  0.562484f,  0.679922f,
  0.282099f,  -0.039075f, 0.404196f,  0.006371f,  0.069679f,  -0.196160f,
  -0.213675f, 0.275187f,  -0.104235f, -0.193090f, 0.003116f,  -0.252454f,
  -0.094591f, 0.210439f,  -0.137070f, 0.145043f,  0.024558f,  0.121718f,
  0.010138f,  0.301651f,  -0.377990f, 0.444414f,  0.001845f,  -0.095334f,
  0.550259f,  0.087603f,  0.792492f,  -0.044584f, 0.641706f,  -0.328458f,
  -0.447791f, 0.135376f,  0.356385f,  0.135748f,  0.310370f,  0.293757f,
  -0.062000f, -0.056368f, 0.343930f,  0.312039f,  0.370763f,  0.452381f,
  -0.023630f, -0.185909f, 0.422277f,  -0.006306f, 0.045166f,  0.423359f,
  -0.157735f, -0.084901f, 0.219527f,  -0.209510f, 0.575057f,  0.249276f,
  0.069267f,  0.233898f,  -0.229392f, 0.117197f,  -0.038551f, 0.293976f,
  0.101996f,  0.120878f,
};

static const float av1_tx_split_nn_bias_16x64_layer0[16] = {
  1.036995f,  0.160249f,  0.100264f,  0.694881f,  0.694677f,  0.128379f,
  -0.843405f, -0.405515f, 0.104139f,  0.182980f,  -0.025472f, 0.901067f,
  -0.299866f, -0.103079f, -0.190352f, -0.048121f,
};

static const float av1_tx_split_nn_weights_16x64_layer1[16] = {
  -1.778868f, 0.174690f,  0.211991f, 0.712138f,  0.589352f,  0.466652f,
  1.029146f,  -0.490044f, 0.483015f, 0.600215f,  -0.577776f, -0.755546f,
  0.348337f,  -0.205082f, 0.347129f, -0.322277f,
};

static const float av1_tx_split_nn_bias_16x64_layer1[1] = {
  0.04230947f,
};

static const NN_CONFIG av1_tx_split_nnconfig_16x64 = {
  8,  // num_inputs
  1,  // num_outputs
  1,  // num_hidden_layers
  {
      16,
  },  // num_hidden_nodes
  {
      av1_tx_split_nn_weights_16x64_layer0,
      av1_tx_split_nn_weights_16x64_layer1,
  },
  {
      av1_tx_split_nn_bias_16x64_layer0,
      av1_tx_split_nn_bias_16x64_layer1,
  },
};
/******************************************************************************/
#endif  // CONFIG_NEW_TX_PARTITION

// Map block size to its corresponding neural net model for tx split prediction.
static const NN_CONFIG *av1_tx_split_nnconfig_map[TX_SIZES_ALL] = {
  NULL,                          // TX_4X4,
  &av1_tx_split_nnconfig_8x8,    // TX_8X8,
  &av1_tx_split_nnconfig_16x16,  // TX_16X16,
  &av1_tx_split_nnconfig_32x32,  // TX_32X32,
  &av1_tx_split_nnconfig_64x64,  // TX_64X64,
  &av1_tx_split_nnconfig_4x8,    // TX_4X8,
  &av1_tx_split_nnconfig_4x8,    // TX_8X4,
  &av1_tx_split_nnconfig_8x16,   // TX_8X16,
  &av1_tx_split_nnconfig_8x16,   // TX_16X8,
  &av1_tx_split_nnconfig_16x32,  // TX_16X32,
  &av1_tx_split_nnconfig_16x32,  // TX_32X16,
  &av1_tx_split_nnconfig_32x64,  // TX_32X64,
  &av1_tx_split_nnconfig_32x64,  // TX_64X32,
  &av1_tx_split_nnconfig_4x16,   // TX_4X16,
  &av1_tx_split_nnconfig_4x16,   // TX_16X4,
  &av1_tx_split_nnconfig_8x32,   // TX_8X32,
  &av1_tx_split_nnconfig_8x32,   // TX_32X8,
  &av1_tx_split_nnconfig_16x64,  // TX_16X64,
  &av1_tx_split_nnconfig_16x64,  // TX_64X16,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_TX_PRUNE_MODEL_WEIGHTS_H_
