#include "../topk_select.cuh"

namespace topk_select_bf16_normal {

using Config = TopkSelectConfig<maca_bfloat16, int32_t, false, false, false, 4096, 512, 1, 8192, 4096, 3, 512, 1>;

template
void run_topk_select_kernel<Config>(const TopkSelectArgs &args);

}   // topk_select_bf16_normal
