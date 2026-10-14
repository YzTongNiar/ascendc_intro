# 《从零构建FA算子》课程资料整理

> 本文档整理了课程参考资料（原始输入）及对应的要点总结，内容围绕昇腾NPU上FlashAttention（FA）算子的接口设计、Kernel实现、Tiling策略与调测方法。

---

## 一、课程总体大纲

### 原始资料

本课程《从零构建FA算子》系统讲解了如何基于昇腾NPU实现基础的FA（Flash Attention）算子，内容涵盖接口设计、kernel实现、tiling策略及调测方法。课程目标包括：掌握FA算子的基本实现思路、接口与tiling框架、kernel开发流程、分核策略及Cube和Vector侧的实现细节。

课程分为四个主要章节：

1. **FA算子接口与原型**：介绍昇腾NPU硬件及PyTorch接口定义，讲解接口设计与代码实现，包括理论回顾、代码设计、测验与思考题。
2. **FA算子kernel侧实现**：涵盖kernel作用、计算流程、分核切块（核间与核内）、核心参数与矩阵计算（MM1、MM2）、Vector计算（包括VF指令、Vector1和Vector2的实现）等内容，强调计算效率与数据搬运策略。
3. **FA算子Tiling侧实现**：讲解Tiling设计原理、实现流程、TilingKey与TilingData的生成、分核策略及代码实现，重点在于负载均衡与核心切分优化。
4. **FA算子调测**：介绍Tiling与kernel的调测方法，包括Debug Tools和mskpp工具的使用，以及Aclnn Fuzz测试框架的流程与调试演示。

课程还包括大考与满意度调查，帮助学员巩固知识并反馈学习体验。

---

## 二、第一章：FA算子接口与原型

### 原始资料

本节以FA算子为例，介绍昇腾算子的接口原型，包括PyTorch接口和C++接口，以及算子原型信息库。课程内容主要包括：

- **前置知识**：要求了解昇腾NPU基础、常见算子类型，以及PyTorch和C++的数据类型定义。
- **学习目标**：通过FA算子的接口示例，掌握PyTorch和C++接口的定义方法，并提供两个调用案例，指导如何使能接口。同时讲解算子原型和信息库的定义方式，并通过学习小册指导新增算子接口或配置信息库。
- **昇腾NPU架构回顾**：当前主要计算加速核心为Ascend 1920，包含24个Ascend Core。每个Core包含搬运单元（如MTE）、存储单元（如L1、UB、L0等）和计算单元（Cube、Vector、Scalar）。
- **算子分类**：根据计算单元使用情况，算子分为Cube算子（仅使用Cube核）、Vector算子（仅使用Vector核）和Mix融合算子（结合Cube和Vector核）。
- **实际应用示例**：以DeepStream V3网络为例，展示了不同类型的算子，如Matrix Multiply（Cube算子）、Relu（激活类算子）、Ma Pro Log（融合算子）等。其中，Count Batch Matrix Multiply因包含反量化处理，也被归类为融合算子。
- **课堂测验**：用于巩固学习内容。

### 要点总结

- **接口体系**：PyTorch接口、C++接口的定义方法，算子原型与信息库（op prototype/info库）的配置方式。
- **硬件背景**：Ascend 1920 含24个AI Core，每个Core由搬运单元（MTE）、存储单元（L1/L0/UB）、计算单元（Cube/Vector/Scalar）组成。
- **算子分类**：Cube算子（纯矩阵计算）、Vector算子（纯向量计算）、Mix融合算子（Cube+Vector，FA正属此类；含反量化的BatchMatMul也算融合算子）。

---

## 三、第二章：Kernel侧实现（总览）

### 原始资料

本节介绍了Kernel的作用及其在计算框架中的关键地位。Kernel是算子中技术含量最高的部分，其主要任务是根据tiling data和tiling key完成高性能计算。tiling data负责指导数据的切分，包括核间切分和核内切分，而Kernel则通过底层语言（如C语言）实现计算逻辑，确保计算流接口的泛化性与高性能。

Kernel的设计需要支持任意形状的输入，例如序列长度1~128k，头数1~512等，同时在不同场景下保持高性能。Kernel运行在NPU的多核架构上，采用SPMD（Single Program, Multiple Data）模式，即所有核运行相同的程序，但处理不同的数据，从而实现并行计算。

在计算过程中，Kernel涉及多条流水线（pipeline），包括数据搬运（红色）、计算（绿色）和数据搬出（黄色）。这些流水线之间存在依赖关系，例如计算必须等待数据搬运完成，数据搬出必须等待计算完成，以避免使用脏数据。然而，在计算过程中可以并行处理下一批数据的搬运，从而提高整体效率。

Kernel的外部接口通常为C语言的extern函数，可以直接从host侧调用，并在device上运行。其输入参数包括输入数据、输出地址、工作空间内存和tiling data，四部分缺一不可。输入数据通常为float或float16类型，而tiling data则指导Kernel如何进行计算切分和执行。

总结来说，Kernel的设计目标是实现高性能、泛化性强的计算，其难点在于如何高效利用多核架构和多条流水线并行，以达到最优性能。

> 注：原始资料中多处"telling"为"tiling"的语音识别转写错误，已按tiling理解。

### 要点总结

- **Kernel的定位**：算子中技术含量最高的部分，依据 **tiling data**（指导核间/核内切分）和 **tiling key** 完成高性能计算。
- **泛化性要求**：支持任意shape（如seq len 1~128k、head数1~512），且各场景下保持高性能。
- **SPMD模式**：多核运行同一程序、处理不同数据。
- **流水线并行**：搬入（MTE）→ 计算（Cube/Vector）→ 搬出三条流水线有依赖关系，但可与下一批数据的搬运重叠（double buffering思想），掩盖搬运开销。
- **接口形式**：host侧可调用的 extern "C" 函数，四个必备入参——输入数据、输出地址、workspace、tiling data。

---

## 四、第三章：FA Tiling侧实现（原理）

### 原始资料

本节介绍了FA Tiling的基本原理与代码实践。FA Tiling运行在Host侧（CPU上），主要功能包括算子模板选择、多核切分、核间负载均衡、高阶API基本块选择、workspace大小计算以及Tiling Key设置。其生成的参数（Tiling Key和Tiling Data）会传递到Kernel侧（Device上运行的MPU），用于数据搬运和计算，最终输出结果。

学习目标包括：

- 了解FA Tiling的设计原理；
- 掌握FA Tiling的实现流程；
- 理解Tiling Key和Tiling Data两个关键要素；
- 掌握FA Tiling的分核策略。

课程还包括代码Demo展示和代码组读环节，帮助加深理解。

### 要点总结

- **运行位置**：Tiling在Host侧（CPU）执行，产物（Tiling Key + Tiling Data）传给Device侧的Kernel使用。
- **六大功能**：算子模板选择、多核切分、核间负载均衡、高阶API基本块选择、workspace大小计算、Tiling Key设置。
- **两个关键要素**：
  - **Tiling Key**：用于kernel模板/分支选择（编译期确定代码路径）；
  - **Tiling Data**：运行时切分参数（核间切分+核内切块的具体数值）。
- **分核策略**：负载均衡是重点——把batch×head×seq等维度合理切到24个核上，避免部分核空闲。

---

## 五、补充资料：官方文档《AI CPU编程》

**来源**：昇腾CANN社区文档《Ascend C算子开发指南》→ 编程指南 → 编程模型 → AI CPU编程
**链接**：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/编程指南/编程模型/AI-CPU编程.md

### 要点总结

- **AI CPU定位**：Device侧ARM64架构处理器，与AI Core一样可直接访问Device内存，承担**非矩阵类、分支密集型**的复杂逻辑计算，作为AI Core的补充；运行环境是基础Linux，可用libc、C++标准库、STL。
- **核函数定义**：用 `__global__ __aicpu__` 声明，实现文件以 `.aicpu` 为后缀（或.cpp加 `-x aicpu` 编译选项）；限制：返回类型不能是void、入参只能是单个指针、不能是类成员函数。
- **调用方式**：Host侧 `.asc` 文件中用 `<<<numBlocks, nullptr, stream>>>` 异构调用语法；AI CPU暂不支持分核，numBlocks建议设为1；且AI Core与AI CPU算子**不能共用同一条stream**。
- **模板核函数**：需在 `.aicpu` 文件中显式实例化声明，Host侧对应extern声明。
- **支持产品**：Ascend 950PR/950DT、Atlas A3训练/推理系列、Atlas A2训练/推理系列。
- **与FA课程的关联**：FA是纯AI Core上的Mix算子（Cube+Vector），通常不涉及AI CPU；但完整昇腾算子体系中，AI CPU常用于shape推导、复杂控制逻辑或标量密集型算子，可作为第一章"算子类型"部分的补充背景。

---

## 六、未能获取的资料

- **知乎文章**：https://zhuanlan.zhihu.com/p/676682524 —— 直接访问被拦截，搜索未检索到具体内容，未能读取。
- **备选参考**：[基于Ascend C的FlashAttention算子性能优化最佳实践](https://zhuanlan.zhihu.com/p/702024606)（昇腾官方发布，讲FlashAttention/FlashAttention2的切分与片上内存优化，可供后续补充阅读）。

---

## 附：知识框架一览

```
《从零构建FA算子》
├── 第一章 接口与原型
│   ├── PyTorch / C++ 接口定义
│   ├── 算子原型与信息库配置
│   ├── Ascend 1920 架构（24核；MTE / L1·L0·UB / Cube·Vector·Scalar）
│   └── 算子分类：Cube算子 / Vector算子 / Mix融合算子（FA属此类）
├── 第二章 Kernel侧实现
│   ├── Kernel定位：依据 tiling data + tiling key 完成高性能计算
│   ├── 泛化性：seq 1~128k，head 1~512
│   ├── SPMD多核并行
│   ├── 流水线：搬入 → 计算 → 搬出（依赖+重叠）
│   └── 接口：extern "C"（输入、输出、workspace、tiling data）
├── 第三章 Tiling侧实现
│   ├── Host侧运行，六大功能（模板选择/多核切分/负载均衡/基本块选择/workspace计算/TilingKey设置）
│   ├── Tiling Key（编译期路径）与 Tiling Data（运行时切分参数）
│   └── 分核策略与负载均衡
├── 第四章 调测
│   ├── Debug Tools、mskpp工具
│   └── Aclnn Fuzz测试框架
└── 补充：AI CPU编程（官方文档）
```
