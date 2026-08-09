// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"


namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS VLMRawPerfMetrics {
    /** @brief Duration of preparation of embeddings */
    std::vector<MicroSeconds> prepare_embeddings_durations;
    /** @brief Number of image slices produced for each input image */
    std::vector<size_t> per_image_slice_counts;
    /** @brief Duration of LoRA adapter switching */
    std::vector<MicroSeconds> adapter_switch_durations;

    // --- adapter apply sub-phase profiling ---
    /** @brief Duration of preparing weight getters inside apply() */
    std::vector<MicroSeconds> adapter_prepare_weight_getters_durations;
    /** @brief Duration of query_state + building state index map */
    std::vector<MicroSeconds> adapter_query_state_durations;
    /** @brief Duration of set_lora_tensors loop (main bottleneck) */
    std::vector<MicroSeconds> adapter_set_lora_tensors_durations;
    /** @brief Sub-split of set_lora_tensors: GenAI/CPU concat evaluator (prepare_lora_tensors) */
    std::vector<MicroSeconds> adapter_prepare_tensors_durations;
    /** @brief Sub-split of set_lora_tensors: GPU plugin state upload (VariableState::set_state) */
    std::vector<MicroSeconds> adapter_set_state_durations;
    /** @brief Duration of setting constant tensors */
    std::vector<MicroSeconds> adapter_set_constants_durations;
};

struct OPENVINO_GENAI_EXPORTS VLMPerfMetrics : public PerfMetrics {
    /** @brief Mean and standard deviation of preparation of embeddings in milliseconds */
    MeanStdPair prepare_embeddings_duration;
    /** @brief Total number of image slices produced for the request */
    size_t total_image_slice_count = 0;
    /** @brief Mean and standard deviation of adapter switching duration in milliseconds */
    MeanStdPair adapter_switch_duration;

    // adapter apply sub-phases
    MeanStdPair adapter_prepare_weight_getters_duration;
    MeanStdPair adapter_query_state_duration;
    MeanStdPair adapter_set_lora_tensors_duration;
    MeanStdPair adapter_prepare_tensors_duration;
    MeanStdPair adapter_set_state_duration;
    MeanStdPair adapter_set_constants_duration;

    MeanStdPair get_prepare_embeddings_duration();
    size_t get_total_image_slice_count();
    MeanStdPair get_adapter_switch_duration();
    MeanStdPair get_adapter_prepare_weight_getters_duration();
    MeanStdPair get_adapter_query_state_duration();
    MeanStdPair get_adapter_set_lora_tensors_duration();
    MeanStdPair get_adapter_prepare_tensors_duration();
    MeanStdPair get_adapter_set_state_duration();
    MeanStdPair get_adapter_set_constants_duration();

    VLMPerfMetrics() = default;

    explicit VLMPerfMetrics(const PerfMetrics& perf_metrics) : PerfMetrics(perf_metrics), prepare_embeddings_duration(), total_image_slice_count(0) {};
    explicit VLMPerfMetrics(PerfMetrics&& perf_metrics) noexcept : PerfMetrics(std::move(perf_metrics)), prepare_embeddings_duration(), total_image_slice_count(0) {};

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    VLMPerfMetrics operator+(const VLMPerfMetrics& metrics) const;

    VLMRawPerfMetrics vlm_raw_metrics;
};

}
