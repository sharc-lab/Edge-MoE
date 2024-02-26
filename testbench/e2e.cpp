#include "../include/kernel.hpp"
#include "../include/tbutil.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

constexpr double MSE_PASS_THRESHOLD = 0.1;
constexpr unsigned int DISPLAY_PATCH_LIMIT = 5;
constexpr unsigned int DISPLAY_DIM_LIMIT = 5;

using std::ifstream;
using std::ostream;
using std::ostringstream;
using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::fixed;
using std::left;
using std::setprecision;
using std::setw;

unsigned int num_images = 1;
bool reload_one_time_weights = true;
image_t images[1];
patch_blocks_t x[1];
patch_blocks_t tmp1;
patch_blocks_t tmp2;
patch_blocks_t tmp3;
patch_blocks_t tmp4;
fm_block_t tmp_hidden[NUM_PATCHES * ceildiv(max(VIT_HIDDEN_DIM, EXPERT_HIDDEN_DIM), FEATURE_BLOCK_SIZE)];
qxk_out_t attn;
softmax_info_t attn_softmax_info;
wt_patch_embed_t patch_embed_weights_in[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH];
wt_bias_t patch_embed_bias_in[FEATURE_DIM];
patch_blocks_t pos_embed;
wt_linear_t attn_weights[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM][FEATURE_DIM];
wt_attn_bias_t attn_bias[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM];
wt_linear_t moe_w_gate[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM];
wt_linear_t moe_weights_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM][FEATURE_DIM];
wt_bias_t moe_bias_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM];
wt_linear_t moe_weights_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM][EXPERT_HIDDEN_DIM];
wt_bias_t moe_bias_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM];
wt_linear_t vit_weights_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM][FEATURE_DIM];
wt_bias_t vit_bias_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM];
wt_linear_t vit_weights_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM][VIT_HIDDEN_DIM];
wt_bias_t vit_bias_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM];
wt_norm_t norm_weights[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM];
wt_bias_t norm_bias[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM];
unsigned int debug_id = 0;
patch_blocks_t reference_x;

int main(int argc, char* argv[])
{
    cout << "Loading inputs... " << flush;
    {
        string filename = "weights/image.float32.bin";
        ifstream ifs(filename, std::ios::binary);
        read(ifs, images[0]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    {
        string filename = "weights/patch_embed_weight.float32.bin";
        ifstream ifs(filename, std::ios::binary);
        read(ifs, patch_embed_weights_in);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    {
        string filename = "weights/patch_embed_bias.float32.bin";
        ifstream ifs(filename, std::ios::binary);
        read(ifs, patch_embed_bias_in);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    {
        string filename = "weights/pos_embed.float32.bin";
        ifstream ifs(filename, std::ios::binary);
        read(ifs, pos_embed);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_qkv_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, attn_weights[layer][ATTN_Q]);
        read(ifs, attn_weights[layer][ATTN_K]);
        read(ifs, attn_weights[layer][ATTN_V]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_attn_proj_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, attn_weights[layer][ATTN_PROJ]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_qkv_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, attn_bias[layer][ATTN_Q]);
        read(ifs, attn_bias[layer][ATTN_K]);
        read(ifs, attn_bias[layer][ATTN_V]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_attn_proj_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, attn_bias[layer][ATTN_PROJ]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(moe_layer, NUM_LAYERS / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (moe_layer * 2 + 1) << "_w_gate_T_task0.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, moe_w_gate[moe_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(moe_layer, NUM_LAYERS / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (moe_layer * 2 + 1) << "_htoh4_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, moe_weights_l1[moe_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(moe_layer, NUM_LAYERS / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (moe_layer * 2 + 1) << "_htoh4_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, moe_bias_l1[moe_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(moe_layer, NUM_LAYERS / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (moe_layer * 2 + 1) << "_h4toh_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, moe_weights_l2[moe_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(moe_layer, NUM_LAYERS / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (moe_layer * 2 + 1) << "_h4toh_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, moe_bias_l2[moe_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (vit_layer * 2) << "_fc1_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, vit_weights_l1[vit_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (vit_layer * 2) << "_fc1_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, vit_bias_l1[vit_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (vit_layer * 2) << "_fc2_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, vit_weights_l2[vit_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(vit_layer, (NUM_LAYERS + 1) / 2)
    {
        ostringstream oss;
        oss << "weights/l" << (vit_layer * 2) << "_fc2_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, vit_bias_l2[vit_layer]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_norm1_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, norm_weights[layer][NORM_1]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_norm2_weight.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, norm_weights[layer][NORM_2]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_norm1_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, norm_bias[layer][NORM_1]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    FOR_EACH(layer, NUM_LAYERS)
    {
        ostringstream oss;
        oss << "weights/l" << layer << "_norm2_bias.float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, norm_bias[layer][NORM_2]);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }

    string reference_var_name;
    {
        ostringstream oss;
        oss << "l" << (NUM_LAYERS - 1) << "_x_post_" << (((NUM_LAYERS - 1) % 2 == 0) ? "mlp" : "moe");
        reference_var_name = oss.str();
    }
    {
        ostringstream oss;
        oss << "weights/" << reference_var_name << ".float32.bin";
        string filename = oss.str();
        ifstream ifs(filename, std::ios::binary);
        read(ifs, reference_x);
        if (!ifs)
        {
            cerr << "Error reading " << filename << endl;
            return 1;
        }
    }
    cout << "done!" << endl;

    cout << "Running kernel... " << flush;
    ViT_compute(
        num_images,
        reload_one_time_weights,
        images,
        x,
        tmp1,
        tmp2,
        tmp3,
        tmp4,
        tmp_hidden,
        attn,
        attn_softmax_info,
        patch_embed_weights_in,
        patch_embed_bias_in,
        pos_embed,
        attn_weights,
        attn_bias,
        moe_w_gate,
        moe_weights_l1,
        moe_bias_l1,
        moe_weights_l2,
        moe_bias_l2,
        vit_weights_l1,
        vit_bias_l1,
        vit_weights_l2,
        vit_bias_l2,
        norm_weights,
        norm_bias,
        debug_id
    );
    cout << "done!" << endl << endl;

    cout << "Sample of values from x vs. " << reference_var_name << ":" << endl;
    {
        ostream formatted(cout.rdbuf());
        formatted << setprecision(8) << fixed;
        FOR_EACH(patch, DISPLAY_PATCH_LIMIT)
        {
            {
                cout << ((patch == 0) ? "[[" : " [");
                FOR_BLOCK(dim, DISPLAY_DIM_LIMIT, FEATURE_BLOCK_SIZE)
                {
                    FOR_OFFSET(dim)
                    {
                        double computed = x[0][patch][dim_block][dim_offset].to_double();
                        if (computed >= 0.0) cout << " ";
                        formatted << setw(9 + (computed < 0.0)) << left << computed;
                        if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
                    }
                }
                cout << ((patch == DISPLAY_PATCH_LIMIT - 1) ? "]]" : "],") << "    ";
            }
            {
                cout << ((patch == 0) ? "[[" : " [");
                FOR_BLOCK(dim, DISPLAY_DIM_LIMIT, FEATURE_BLOCK_SIZE)
                {
                    FOR_OFFSET(dim)
                    {
                        double actual = reference_x[patch][dim_block][dim_offset].to_double();
                        if (actual >= 0.0) cout << " ";
                        formatted << setw(9 + (actual < 0.0)) << left << actual;
                        if (dim != DISPLAY_DIM_LIMIT - 1) cout << ", ";
                    }
                }
                cout << ((patch == DISPLAY_PATCH_LIMIT - 1) ? "]]" : "],") << endl;
            }
        }
    }
    cout << endl;

    double mse = 0.0;
    double mae = 0.0;
    FOR_EACH(patch, NUM_PATCHES)
    {
        FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            FOR_OFFSET(dim)
            {
                double computed = x[0][patch][dim_block][dim_offset].to_double();
                double actual = reference_x[patch][dim_block][dim_offset].to_double();
                double error = actual - computed;
                double abs_error = (error < 0.0) ? -error : error;
                mse += error * error;
                mae += abs_error;
            }
        }
    }
    mse /= static_cast<double>(NUM_PATCHES * FEATURE_DIM);
    mae /= static_cast<double>(NUM_PATCHES * FEATURE_DIM);
    cout << "MSE: " << mse << endl;
    cout << "MAE: " << mae << endl;

    return (mse <= MSE_PASS_THRESHOLD) ? 0 : 1;
}
