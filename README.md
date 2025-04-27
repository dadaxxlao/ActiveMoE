# ActiveMoE
The project aims to accelerate the inference speed of Mixture-of-Experts (MoE) large language models (LLM) on GPU memory-constrained devices, such as consumer-grade graphics cards and edge devices.
## Pre-Gated Challenges
1. **Architectural Modifications**: Requires integration of a novel "pre-gate" function into the model architecture, necessitating fine-tuning during adaptation. This predictive mechanism executes as part of forward propagation through GPU computation.
2. **Expert Cache Management Deficiency**: Lacks proactive data transfer initiation (prefetching) for subsequent layer experts to maximize overlap between transmission and computation. Core philosophy emphasizes "latency masking through anticipatory loading" rather than sophisticated cache management strategies (e.g., LRU/FIFO replacement policies, cache size optimization).
3. **Miss Handling Strategy Gap**: No explicit contingency for prediction inaccuracies or delayed prefetching causing expert unavailability. System defaults to blocking load operations when cache misses occur, with performance relying on temporal prediction accuracy through early initiation.
4. **Suboptimal Cache Utilization**: Single-layer prediction limits cached content to current and immediate next-layer experts. Multi-layer prefetching (N+2/N+3) could enhance cache foresight given sufficient temporal windows for anticipatory loading.

## Architectural Blueprint
![PipeLine-Page-1](/PreGated/Image/PipeLine-Page-2.png)
1. **Predictive Module**
    * **Function**: Probabilistic forecasting of expert selection for future layers (N+k)
    * **Input**: Hidden state from preceding MoE layer
    * **Output**: Expert probability distribution with confidence metrics
    * **Implementation**: Lightweight MLP trained via offline expert selection pattern analysis
    * **Execution**: Concurrent CPU operation synchronized with GPU computation pipeline
2. **Prefetch Orchestrator**
    * **Components**:
        - Priority Task Queue (Max-Heap implementation)
        - Asynchronous I/O Thread Pool
    * **Operation Modes**:
        - Speculative Prefetch (Background, interruptible)
        - Critical Path Prefetch (Foreground, prioritized)
3. **GPU Memory Management**
    * **Structure**:
        - Contiguous Memory Allocation with Slab Allocation
        - Metadata Tracking Matrix (Presence, Last Access, Loading State)
    * **Policy**: LRU with Hot-Cold Expert Segmentation
4. **Inference Engine Integration**
    * **Coordination Mechanisms**:
        - Compute-Ready Expert Prioritization
        - Partial Computation Enablement (Block-Level Execution)
        - Prefetch Task Preemption Interface
    * **Synchronization**:
        - Double-Buffered State Management
        - Non-Blocking Cache Status Checks
## **Operational Workflow**:
1. **Concurrent Prediction Phase**: CPU analyzes layer N inputs while GPU processes layer N-1 computations
2. **Speculative Prefetch Initiation**: Queue background loading of predicted experts for layer N+1/N+2
3. **Precision Verification**: Validate predictions against actual gate outputs upon layer N computation
4. **Cache Optimization**:
    - Purge unnecessary speculative prefetches
    - Elevate confirmed expert loading priority
5. **Adaptive Execution**:
    - Execute available experts immediately
    - Interleave remaining computations with progressive expert loading
6. **State Propagation**: Update cache metadata and prediction model feedback loop
