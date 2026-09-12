#pragma once

#include "kerutils/common/common.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "cuda/common.h"
// [MACA] 此处原有 sm80/sm90/sm100 三组 intrinsics（cp.async / TMA gather /
// UMMA / cluster / st_async），已整组删除：它们全是内联 PTX 汇编，MACA 汇编器
// 只认 MACA ISA，且其中的 'l'（64 位）操作数约束在 mxcc 上直接报
// `invalid constraint`。DeepSelect 内核真正用到的那两个函数（st_shared、trap）
// 已在 cuda/common.h 里给出 MACA 实现；st_async 与 cluster 屏障只被 v3_cluster
// 使用，而 MACA 无 cluster 支持，该变体（含整个 v3_cluster 目录与 api.cu 里
// 对它的分发）已整体删除。
#endif
