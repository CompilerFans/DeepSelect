#include "../topk_select.cuh"

namespace topk_select_fp32 {

using Config = TopkSelectConfig<float, int64_t, false, false, false, 1024, 512, 1, 8192, 1536, 3, 512, 1>;

template
void run_topk_select_kernel<Config>(const TopkSelectArgs &args);

}   // topk_select_fp32
