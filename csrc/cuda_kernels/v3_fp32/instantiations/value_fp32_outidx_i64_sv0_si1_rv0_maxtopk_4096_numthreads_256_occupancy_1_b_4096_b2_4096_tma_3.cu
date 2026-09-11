#include "../topk_select.cuh"

namespace topk_select_fp32 {

using Config = TopkSelectConfig<float, int64_t, false, true, false, 4096, 256, 1, 4096, 4096, 3, 512, 1>;

template
void run_topk_select_kernel<Config>(const TopkSelectArgs &args);

}   // topk_select_fp32
