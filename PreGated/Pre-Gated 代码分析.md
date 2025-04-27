项目源码：https://github.com/ranggihwang/Pregated_MoE
论文链接：https://arxiv.org/pdf/2308.12066
## 论文方案分析

Pre-gated 方案旨在优化 MoE 模型推理中的性能瓶颈，即动态加载专家（Expert）参数（通常存储在 CPU 内存或 SSD）到 GPU 显存所带来的延迟。其核心思想是：**在 GPU 计算当前 MoE 层时，利用预测机制提前（Pre）判断下一 MoE 层可能需要哪些专家，并异步地将这些预测的专家参数从低速存储介质预取（Prefetch）到 GPU 显存中，从而实现计算和数据加载的流水线重叠，隐藏数据传输延迟，提升推理吞吐量和降低延迟。**

## 代码分析
### **启用与配置 (eval_all.py & C++)**

*   **触发**: 在 `Pregated_MoE/scripts/eval_all.py` 脚本中，当 `method` 参数设置为 `"Pre-gated"` 时，会为 FasterTransformer 的配置项设置特定的值：
    *   `encoder_fetcher_mode = "1"` （可能代表按需获取或标准的 CPU 卸载模式）
    *   `decoder_fetcher_mode = "2"` （**关键：启用解码器端的 Pre-gated/Prefetch 模式**）
*   **传递**: 这些配置被写入 FasterTransformer C++ 库读取的配置文件（例如 `/workspace/FasterTransformer/cpp_config.ini`，根据实际路径调整）。
*   **读取**: 在 C++ 核心库中，`Pregated_MoE/src/fastertransformer/utils/config.h` 读取配置文件，并将 `decoder_fetcher_mode` 的值（即 2）解析为一个枚举类型 `FetchType`（值为 2 通常对应 `FetchType::PREFETCH`）。

### 初始化 Fetcher 上下文 (T5Decoder.cc -> FfnLayer.cc -> fetcher.h)

*   在模型组件（`Pregated_MoE/src/fastertransformer/models/t5/T5Decoder.cc`）初始化 FFN 层时，会将读取到的 `FetchType::PREFETCH` 模式传递给 FFN 层的 `initFetcherContext` 方法。
*   `Pregated_MoE/src/fastertransformer/layers/FfnLayer.cc` 中的 `initFetcherContext` 方法会根据传入的模式创建一个 `FetcherContext` 类的实例（定义在 `Pregated_MoE/src/fastertransformer/utils/fetcher.h`）。
*   `FetcherContext` 类是参数获取的核心，它内部维护：
    *   **模式 (`mode`)**: 存储传入的 `FetchType` (如 `PREFETCH` 或 `FETCH_ON_DEMAND`)。
    *   **双缓冲**: 工作区 (`_working_`) 和目标区 (`_dst_`) 两套 GPU 显存指针，用于异步加载和计算使用。
    *   **参数来源**: 当前层和下一层参数的位置信息。
    *   **状态标志**: 如 `first_time`（是否为第一个 MoE 层）、`last_time`（是否为最后一个 MoE 层）。
    *   **核心接口**:
        *   `fetch(experts, prefetch)`: 启动异步参数获取。`prefetch=true` 表示预取下一层。
        *   `sync()`: 同步等待当前正在进行的 fetch 操作完成，并交换 working 和 dst 缓冲区。
        *   `get_weights(...)`: 返回当前计算层所需的、已加载到 dst 缓冲区的专家参数指针。
        *   `set_source(...)`: 设置权重来源。
        *   `set_layer(...)`: 设置当前层信息和状态标志。

### "Pre-gated" 核心执行流程 (CutlassMoeFCRunner::run_moe_fc)

Pre-gated 的关键流水线逻辑在 MoE 计算的核心 Kernel Runner 中实现，即 `Pregated_MoE/src/fastertransformer/kernels/moe_kernels.cu` 中的 `CutlassMoeFCRunner::run_moe_fc` 函数：

*   **模式检查**: 函数首先检查 `fetcher_context_` 是否存在及其 `mode` 是否为 `PREFETCH`，并检查是否是第一个 MoE 层 (`!fetcher_context_->first_time`)。
*   **计算当前层门控**: 通过 `topk_gating_softmax_kernelLauncher` 计算当前输入对应的门控值，选出 TopK 个专家，得到 `expert_for_source_row`。
*   **排序**: 使用 `sorter_.run` 对选出的专家进行排序，得到 `permuted_experts_`（排序后、去重的、实际需要加载或计算的专家索引列表）和 `permuted_rows_`（数据排列信息）。
*   **Fetcher 交互 (关键部分)**:
    *   **If `mode == PREFETCH`:**
        *   **非首层 (`!first_time`)**:
            1.  `fetcher_context_->sync()`: **同步点1**。等待 *上一轮* 为 *当前层* 启动的预取操作完成。此时，当前层计算所需的专家参数已加载到 `FetcherContext` 的 `dst` 缓冲区。
            2.  `fetcher_context_->get_weights(...)`: 获取 `dst` 缓冲区中当前层所需的专家参数指针。
            3.  **【GPU 计算当前层】**: 使用 `get_weights` 返回的指针，调用 MoE GEMM Kernels（`moe_gemm_runner_.moe_gemm_bias_act` 等）执行当前层的 FFN 计算。
            4.  `fetcher_context_->fetch(permuted_experts_, true)`: **Pre-gated 核心**。使用 *当前层* 排序后的专家列表 `permuted_experts_`，**启动对 *下一层* 专家参数的异步预取**，加载到 `FetcherContext` 的 `working` 缓冲区。`prefetch=true` 表明这是一个预取操作。
        *   **首层 (`first_time`)**:
            1.  `fetcher_context_->fetch(permuted_experts_, false)`: **同步获取**当前层（第一层）所需的专家。`prefetch=false`。
            2.  `fetcher_context_->sync()`: **同步点2**。等待第一层的参数获取完成。
            3.  `fetcher_context_->get_weights(...)`: 获取第一层的专家参数指针。
            4.  **【GPU 计算当前层】**: 执行第一层的 FFN 计算。
            5.  **(无预取)**: 第一层执行完毕后，不立即启动下一层的预取（因为 `run_moe_fc` 的主要预取逻辑在非首层部分）。下一层的预取会在处理第二层时才启动。
    *   **If `mode == FETCH_ON_DEMAND`:**
        1.  `fetcher_context_->fetch(permuted_experts_, false)`: **同步获取**当前层所需的专家。
        2.  `fetcher_context_->sync()`: 等待获取完成。
        3.  `fetcher_context_->get_weights(...)`: 获取当前层的专家参数指针。
        4.  **【GPU 计算当前层】**: 执行当前层的 FFN 计算。
        5.  **(无预取)**.
![[Core_Code.png]]
## 预测 (Gating) 机制

也就是说**它使用当前 MoE 层通过门控网络（Gating Network）计算并排序后实际选择的专家列表 (`permuted_experts_`)，直接作为下一层需要预取的专家列表。** 这相当于假设下一层激活的专家与当前层相同。==***根本不存在所谓的预测！！！***==
可以使用以下的PipeLine进行展示
![[PipeLine-Page-1.png]]

## 错误处理机制
那么问题就来了：那么如果如果两层的专家列表不相同会发生什么？很显然，会**权重获取错误**，进而**计算使用错误权重**，导致计算结果大幅出现问题，即生成内容的回答也会出现乱码。
但是在当前的论文中缺少相应的验证，并且代码中的Input Token使用的为随机生成的词典组合，不具有任何意义。
经过添加T5-Tokenizer，使用自定义的Input Token，并修改Faster Transformer框架将Output进行输出，得到结果如下：
![[Output.png]]
可以看到，对于Pre-Gated方案，由于专家列表获取错误，无法正确生成内容，更加证明了该代码方案的局限性。