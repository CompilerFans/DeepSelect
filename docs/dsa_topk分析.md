# dsa\_topk分析

# 优化背景

*   mcoplib 最初实现 sglang fast\_topk 算法在 small batch size, large sequence length 情形下 latency 过高
    

![lQLPKGX-5oT5jrPNA3XNA8Gw4_YsA91oaqoKIX_RjgYGAA_961_885.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/v9kqDeaZya1e8OVx/img/94e1b331-d1f2-49a2-a30a-ab784be80f83.png)

*   mcDeepGemm 库承接 cudnn-frontend DSA lightning indexer 前向推理接口实现时需要实现 topk selector 算子，功能类似但是要求任意 topk size
    
*   尝试融合 sglang 和 cudnn-frontend topk 功能，并结合 C500/C600U 硬件 spec 探索更高效的算法实现
    

# TopK 算法介绍

使用 GPU 实现的 TopK 算子大部分采用 radix selection 算法。

算法首先确认一种变换方法，可以把元素数据转化成若干 bit 的可排序 key

1.  **粗筛**：遍历元素确定 TopK 对应的 threshold bin
    
    1.  **Build histogram**：建立若干bit能排列出的所有情形的 histogram 数组。遍历所有元素，取变换后 key 的若干最高有效位，并在对应的 histogram bin 中累加计数
        
    2.  **反向 cumsum**：统计每个 bin 及所有更大 bin 的元素数（累加求和），找到第 K 大元素所在的 boundary bin，即 threshold
        
2.  **细筛**：挑选出 threshold 之上的元素， 只对 threshold 内的 candidate 重复剩余位数的比较
    
    1.  **compact**：再次完整扫描。>threshold 的元素直接写入结果，=threshold 的元素被压缩成 candidate
        
    2.  **refine**：对剩余的 candidates 进行多轮若干bit的重复 histogram-cumsum-compact 循环，直到选满 K 各元素或者 key 的最后一位，此时剩余元素完全相同，直接拿取剩余个数即可
        

# 算法分支总览

|  | **Singe** | **Chunks** | **Coarse12** |
| --- | --- | --- | --- |
| **Dispatch shape** | `n_rows`blocks \* 1024 threads | `init`：`n_rows`blocks \* 128 threads<br>`coarse_hist`：3~6`n_rows`blocks \* 1024 threads<br>`compact_refine`：`n_rows`blocks \* 1024 threads | `n_rows`blocks \* 640 threads |
| **Coarse key** | FP32 转 FP16 后生成 key | FP32 转 FP16 后生成 key | 直接使用 FP32 生成 key |
| **Coarse radix** | 高 10 bit，1024 bin | 高 10 bit，每个 chunk 各有 1024 bin | 高 12 bit，4096 bin |
| **Refine radix** | FP32 key：`8+8+8+8` bit | FP32 key：`8+8+8+8` bit | FP32 key 剩余：`8+8+4` bit |
| **Coarse cumsum** | 分层 warp suffix scan<br>*   16 个 64-lane warp 先各扫 64 bins<br>    <br>*   warp 0 再扫 warp totals | `coarse_hist`：每行最后到达 block 合并各 chunk 的 histogram，再做cumsum | *   4096 bin 每连续 16 bin 聚合成 256 bin histo 做 cumsum<br>    <br>*   在命中的 16 sub-bin 中寻找精确 threshold |
| **Fine cumsum** | 256-bin 单 warp suffix scan | `compact_refine`：每行最后到达 block 合并各 chunk 的histogram，再做cumsum | 256-bin 单 warp suffix scan |
| **Candidate capacity** | 4096 | 4096 | 2048 |
| **Candidate overflow** | 溢出后不保存后续 candidates | 溢出后不保存后续 candidates | 溢出后从全局数组中读取 |
| **Shared memory** | *   8 KB topk indices<br>    <br>*   4 KB histogram<br>    <br>*   32 KiB：2-stage 4096 candidate index<br>    <br>总计 ~44–45 KB | `init`：0 KB<br>`coarse_hist`：4KB histo<br>`compact_refine`：1KB histo +32 KB candidate（~33KB） | 16KB dyn shared mem 复用<br>*   Coarse：4096 bin histo<br>    <br>*   Refine：2-stage 2048 candidate index<br>    <br>1KB cumsum histogram<br>总计 ~17 KB |
| **Global workspace** | 无 | 每行30~50KB，跨 block 同步 | 无 |
| **Resident block/AP**<br>**（C600U）** | 2 （thread bound） | `init`：-<br>`coarse_hist`：2<br>`compact_refine`：2 | 3（thread bound） |
| **Resident block/AP**<br>**（C500）** | 1 （shared bound） | `init`：-<br>`coarse_hist`：2<br>`compact_refine`：1 | 3（thread bound） |
| **优势场景** | *   性能基线<br>    <br>*   小`n_rows`小 `length` | *   小 `n_rows` 大 `length`<br>    <br>*   提升 GPU 利用率 | *   大 `n_rows` 大 `topk`<br>    <br>*   coarse12 bin 筛选更细<br>    <br>*   高 resident 提升 并行度 |