#include "../topk_select.cuh"

namespace topk_select_fp32 {

using Config = TopkSelectConfig<float, int32_t, true, false, true, 512, 512, 1, 8192, 2560, 3, 512, 1>;

template
void run_topk_select_kernel<Config>(const TopkSelectArgs &args);

}   // topk_select_fp32
