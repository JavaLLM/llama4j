package org.javallm.llama4j.model.params;

import jakarta.validation.constraints.AssertTrue;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
import lombok.experimental.Accessors;

import javax.annotation.Nullable;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

@Accessors(chain = true)
@Data
public final class ModelParameters {
    private boolean verbose = false;

    /**
     * Model context size
     */
    @Min(value = 8, message = "contextSize must larger than 512!")
    private int contextSize = 512;

    /**
     * Batch size for prompt processing (must be >= 32 to use BLAS)
     */
    @Min(value = 0, message = "batchSize for prompt processing must larger than 0")
    private int batchSize = 64;

    /**
     * The random seed
     */
    private int seed = -1;

    /**
     * The path to the model
     */
    @NotBlank(message = "modelPath must be provided")
    private String modelPath;

    /**
     * Number of threads for evaluating inputs, set to -1 for auto-detection
     */
    private int nThreads = Math.max(Math.round(Runtime.getRuntime().availableProcessors() / 2.0f), 1);

    /**
     * Base frequency for RoPE sampling.
     */
    private float ropeFreqBase = 10000.0f;

    /**
     * Scaling factor for RoPE sampling.
     */
    private float ropeFreqScale = 1.0f;

    /**
     * Path to the LoRA (Low-Rank Adaptation) adapter file to be applied to the model.
     */
    @Nullable
    private String loraPath;

    /**
     * Optional model to use as a base for the layers modified by the LoRA adapter.
     */
    @Nullable
    private String loraBase;

    /**
     * Embedding mode
     */
    private boolean embeddingMode = false;

    private Map<String, String> extra = new HashMap<>();

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
