package org.javallm.llama4j.model.params;

import jakarta.validation.constraints.AssertTrue;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import javax.annotation.Nullable;
import java.io.File;

@Builder
@ToString
public final class ModelParameters {
    @Getter
    @Builder.Default
    private boolean verbose = false;

    /**
     * Model context size
     */
    @Min(value = 8, message = "contextSize must larger than 512!")
    @Getter
    @Builder.Default
    private int contextSize = 512;

    /**
     * This parameter allows users to retain the original prompt when the model runs out of context, ensuring a connection to the initial instruction or conversation topic is maintained. It specifies the number of tokens from the initial prompt to retain when the model resets its internal context. By default, this value is set to 0 (meaning no tokens are kept). Use -1 to retain all tokens from the initial prompt.
     */
    @Getter
    @Builder.Default
    private int keepInitPromptTokensCnt = 0;

    /**
     * Batch size for prompt processing (must be >= 32 to use BLAS)
     */
    @Min(value = 0, message = "batchSize for prompt processing must larger than 0")
    @Getter
    @Builder.Default
    private int batchSize = 64;

    /**
     * The random seed
     */
    @Getter
    @Builder.Default
    private int seed = -1;

    /**
     * The path to the model
     */
    @NotBlank(message = "modelPath must be provided")
    @Getter
    private String modelPath;

    /**
     * Number of threads for evaluating inputs, set to -1 for auto-detection
     */
    @Getter
    @Builder.Default
    private int nThreads = Math.max(Math.round(Runtime.getRuntime().availableProcessors() / 2.0f), 1);

    /**
     * Base frequency for RoPE sampling.
     */
    @Getter
    @Builder.Default
    private float ropeFreqBase = 10000.0f;

    /**
     * Scaling factor for RoPE sampling.
     */
    @Getter
    @Builder.Default
    private float ropeFreqScale = 1.0f;

    /**
     * Path to the LoRA (Low-Rank Adaptation) adapter file to be applied to the model.
     */
    @Getter
    @Nullable
    private String loraPath;

    /**
     * Optional model to use as a base for the layers modified by the LoRA adapter.
     */
    @Getter
    @Nullable
    private String loraBase;

    /**
     * Embedding mode
     */
    @Getter
    @Builder.Default
    private boolean embeddingMode = false;

    @AssertTrue(message = "modelPath cannot be resolved, please check")
    public boolean isModelPathValid() {
        File f = new File(this.modelPath);
        return f.exists();
    }

    @AssertTrue(message = "contextSize must be larger than batchSize")
    public boolean isContextSizeLargerThanBatchSize() {
        return this.contextSize > this.batchSize;
    }
}
