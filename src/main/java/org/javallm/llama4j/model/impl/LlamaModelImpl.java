package org.javallm.llama4j.model.impl;

import com.google.common.base.Preconditions;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.javallm.llama4j.model.LlamaModel;
import org.javallm.llama4j.model.params.ModelParameters;
import org.javallm.llama4j.model.params.PenalizeParameters;
import org.javallm.llama4j.model.params.SamplingParameters;
import org.javallm.llama4j.utils.ValidationUtils;
import org.javallm.llamacpp.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static org.javallm.llamacpp.global.llama.*;

/**
 * TODO: add slf4j LOGGING
 */
public class LlamaModelImpl implements LlamaModel {
    private final ModelParameters modelParams;

    /******************** State ***********************/
    private int nPastTokens = 0;
    private final int[] inputTokens;
    private final float[][] inputLogits;

    /******************** LLaMA.cpp internal ***********************/
    private final llama_context_params _params;
    private final llama_context _context;
    private final llama_model _model;
    private final int _n_ctx;
    private final int _n_vocab;
    private final int _n_embed;
    private final int _token_bos;
    private final int _token_eos;
    private final int _token_nl;

    /**
     * Constructor to initialize a Llama model
     *
     * @param modelParams model parameters
     * @throws IllegalArgumentException when the passed arguments are invalid
     */
    public LlamaModelImpl(ModelParameters modelParams) {
        Preconditions.checkNotNull(modelParams);
        ValidationUtils.validateOrThrow(modelParams);

        this.modelParams = modelParams;

        this._params = initLLaMAContextParams(modelParams);
        Preconditions.checkNotNull(this._params);

        this._model = llama_load_model_from_file(modelParams.getModelPath(), _params);
        Preconditions.checkNotNull(this._model);

        this._context = llama_new_context_with_model(_model, _params);
        Preconditions.checkNotNull(this._context);

        this._n_ctx = llama_n_ctx(this._context);
        Preconditions.checkState(this._n_ctx >= 8);

        this._n_vocab = llama_n_vocab(this._context);
        Preconditions.checkState(this._n_vocab >= 0);

        this._n_embed = llama_n_embd(this._context);
        Preconditions.checkState(this._n_embed >= 0);

        this._token_bos = llama_token_bos(this._context);
        this._token_eos = llama_token_eos(this._context);
        this._token_nl = llama_token_nl(this._context);

        applyLoRA();

        if (modelParams.isVerbose()) {
            BytePointer info = llama_print_system_info();
            System.out.println(new String(info.getStringBytes(), StandardCharsets.UTF_8));
        }

        // pre-allocate arrays for storing input tokens and the corresponding logits
        this.inputTokens = new int[contextSize()];
        this.inputLogits = new float[contextSize()][vocabSize()];
    }

    /**
     * Apply LoRA
     */
    private void applyLoRA() {
        if (StringUtils.isNoneBlank(modelParams.getLoraPath())) {
            String loraBase = StringUtils.isNotBlank(modelParams.getLoraBase()) ? modelParams.getLoraBase() : null;
            int resultCode = llama_model_apply_lora_from_file(this._model, modelParams.getLoraPath(), loraBase, modelParams.getNThreads());
            if (resultCode != 0) {
                // TODO: need special Exception class
                throw new RuntimeException(String.format("Failed to apply LoRA with loraBase=%s and loraPath=%s", modelParams.getLoraBase(), modelParams.getLoraPath()));
            }
        }
    }

    @Override
    public byte[] getState() {
        int stateSize = (int) llama_get_state_size(_context);
        if (stateSize <= 0) {
            return null;
        }

        byte[] state = new byte[stateSize];
        llama_copy_state_data(_context, state);
        return state;
    }

    @Override
    public void loadState(byte[] state) {
        Preconditions.checkNotNull(state);
        int stateSize = (int) llama_get_state_size(_context);
        if (state.length != stateSize) {
            throw new IllegalArgumentException(String.format("stateSize not match! expected = %d Bytes, actual = %d Bytes", stateSize, state.length));
        }
        llama_set_state_data(_context, state);
    }

    @Override
    public int[] tokenize(String text, boolean addBos) {
        int offset = addBos ? 1 : 0;

        // The String should be encoded into UTF-8 format before tokenization
        byte[] content = text.getBytes(StandardCharsets.UTF_8);
        int nBytes = content.length;
        int[] tokens = new int[nBytes + offset + 4];

        BytePointer input = new BytePointer(nBytes);
        input.put(content);

        int nTokens = llama_tokenize(
                this._context,
                input,
                tokens,
                contextSize(),
                addBos);
        if (nTokens < 0) {
            nTokens = Math.abs(nTokens);
            tokens = new int[nBytes + offset];
            nTokens = llama_tokenize(
                    this._context,
                    input,
                    tokens,
                    nTokens,
                    addBos
            );
            if (nTokens < 0) {
                throw new RuntimeException("Error happened during tokenization!");
            }
        }
        Preconditions.checkState(nTokens <= tokens.length);

        // Only take the first N tokens
        return ArrayUtils.subarray(tokens, 0, nTokens);
    }

    @Override
    public String detokenize(int[] tokens) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        // We must detokenize all bytes at once, since a word can be represented by more than one byte
        for (int token : tokens) {
            int bufferSize = 8;
            BytePointer buffer = new BytePointer(bufferSize);
            int n = llama_token_to_piece(_context, token, buffer, bufferSize);
            if (n < 0) {
                buffer.close();
                buffer = new BytePointer(n);
                int check = llama_token_to_piece(_context, token, buffer, n);
                Preconditions.checkState(check == -n);
            }

            byte[] data = Arrays.copyOf(buffer.getStringBytes(), Math.abs(n));
            try {
                stream.write(data);
            } catch (IOException ex) {
                throw new RuntimeException(String.format("Failed to detokenize: %s", Arrays.toString(tokens)), ex);
            }
        }
        byte[] bytes = stream.toByteArray();
        return convertToUtf8String(bytes);
    }

    /**
     * Convert bytes to UTF-8 String. If the byte array contains incomplete code point, discard it
     * @param bytes input
     * @return resulting string
     */
    private String convertToUtf8String(byte[] bytes) {
        String result = new String(bytes, StandardCharsets.UTF_8);
        if ( bytes.length != result.getBytes(StandardCharsets.UTF_8).length) {
            return null;
        }
        return result;
    }

    @Override
    public void reset(int nPastTokens) {
        Preconditions.checkState(nPastTokens >= 0, "nPastToken should >= 0");
        Preconditions.checkState(nPastTokens <= this.contextSize() && nPastTokens <= this.nPastTokens, "nPastToken is too large!");

        // shift tokens and logits
        int offset = -(this.nPastTokens - nPastTokens);
        ArrayUtils.shift(this.inputTokens, offset);
        ArrayUtils.shift(this.inputLogits, offset);

        // update nPastTokens
        this.nPastTokens = nPastTokens;
    }

    @Override
    public void evaluate(int[] tokens) {
        int nTokens = tokens.length;

        // batch evaluation
        for (int i = 0; i < nTokens; i += modelParams.getBatchSize()) {
            int actualBatchSize = Math.min(modelParams.getBatchSize(), nTokens - i);

            // Infinite text generation via context swapping
            // i.e., when the context window runs out, only retain (approximately) half of the tokens
            if (this.nPastTokens + actualBatchSize >= contextSize()) {
                int nPastTokens = contextSize() - Math.max(actualBatchSize, contextSize() / 2);
                reset(nPastTokens);
            }

            int[] batch = ArrayUtils.subarray(tokens, i, i + actualBatchSize);
            int returnCode = llama_eval(_context, batch, actualBatchSize, this.nPastTokens, modelParams.getNThreads());
            if (returnCode != 0) {
                throw new RuntimeException(String.format("Fail to eval tokens: %s", Arrays.toString(tokens)));
            }

            Preconditions.checkState(this.nPastTokens >= 0 && this.nPastTokens + actualBatchSize <= this.inputTokens.length - 1);

            // save tokens
            System.arraycopy(batch, 0, this.inputTokens, this.nPastTokens, actualBatchSize);

            // save logits
            float[] logits = new float[this.vocabSize()];
            llama_get_logits(this._context).get(logits);
            System.arraycopy(logits, 0, this.inputLogits[this.nPastTokens], 0, this.vocabSize());

            // update nPastTokens
            this.nPastTokens += actualBatchSize;
        }
    }

    @Override
    public int sample(SamplingParameters samplingParams, PenalizeParameters penalizeParameters) {
        float[] logits = new float[this.vocabSize()];
        llama_get_logits(this._context).get(logits);

        // TODO: Apply logit processors

        // Apply penalty
        llama_token_data_array candidates = penalize(penalizeParameters, logits);

        // Greedy sampling
        if (samplingParams.getTemperature() <= 0) {
            return llama_sample_token_greedy(_context, candidates);
        }

        // Miro State Sample Algorithm
        float mu = 2.0f * samplingParams.getMiroStatTau();
        FloatPointer miroStatMu = new FloatPointer(1);
        miroStatMu.put(mu);

        switch (samplingParams.getMiroStatStrategy()) {
            // micro state sampling algorithm v1
            case V1:
                int miroStatM = 100;
                llama_sample_temperature(_context, candidates, samplingParams.getTemperature());
                return llama_sample_token_mirostat(_context, candidates, samplingParams.getMiroStatTau(), samplingParams.getMiroStatEta(), miroStatM, miroStatMu);
            // micro state sampling algorithm v2
            case V2:
                llama_sample_temperature(_context, candidates, samplingParams.getTemperature());
                return llama_sample_token_mirostat_v2(_context, candidates, samplingParams.getMiroStatTau(), samplingParams.getMiroStatEta(), miroStatMu);
            case DISABLE:
            default:
                // Temperature sampling
                llama_sample_top_k(_context, candidates, samplingParams.getTopK(), 1);
                llama_sample_tail_free(_context, candidates, samplingParams.getTsfZ(), 1);
                llama_sample_typical(_context, candidates, samplingParams.getTypicalP(), 1);
                llama_sample_top_p(_context, candidates, samplingParams.getTopP(), 1);
                llama_sample_temperature(_context, candidates, samplingParams.getTemperature());

                return llama_sample_token(_context, candidates);
        }
    }

    @Override
    public float[] embed(String input) {
        if (!this.modelParams.isEmbeddingMode()) {
            throw new UnsupportedOperationException("Llama model must be called with parameter `embeddingMode=True` to call this method!");
        }

        debug(() -> llama_reset_timings(this._context));

        float[] embedding = new float[embeddingSize()];
        int[] tokens = tokenize(input, true);

        // reset model state
        reset();

        evaluate(tokens);
        llama_get_embeddings(this._context).get(embedding);

        debug(() -> llama_print_timings(this._context));

        return embedding;
    }

    /**
     * Apply penalty
     *
     * @param params penalize parameters
     * @return candidates tokens
     */
    private llama_token_data_array penalize(PenalizeParameters params, float[] logits) {
        Preconditions.checkNotNull(params);
        Preconditions.checkState(logits != null && logits.length == vocabSize());

        // Collect token candidates
        llama_token_data dataArray = new llama_token_data(vocabSize());
        for (int tokenId = 0; tokenId < vocabSize(); tokenId++) {
            llama_token_data tokenData = new llama_token_data();
            tokenData.id(tokenId);
            tokenData.logit(logits[tokenId]);
            tokenData.p(.0f);

            dataArray.getPointer(tokenId).put(tokenData.getPointer());
        }

        llama_token_data_array candidates = new llama_token_data_array();
        candidates.data(dataArray);
        candidates.size(vocabSize());
        candidates.sorted(false);

        // Save the logit for the new line token before applying penalty
        float newLineLogit = logits[newLineToken()];

        // Retain the last `lastNRepeat` tokensToBePenalized only
        int[] pastTokens = this.inputTokens();
        int N = pastTokens.length;
        int lastNRepeat = Math.min(Math.min(N, params.getRepeatLastTokensCount()), modelParams.getContextSize());
        int[] tokensToBePenalized = ArrayUtils.subarray(pastTokens, N - lastNRepeat, N);

        // Apply penalties
        llama_sample_repetition_penalty(
                _context,
                candidates,
                tokensToBePenalized,
                lastNRepeat,
                params.getRepeatPenalty()
        );

        llama_sample_frequency_and_presence_penalties(
                _context,
                candidates,
                tokensToBePenalized,
                lastNRepeat,
                params.getAlphaFrequency(),
                params.getAlphaPresence()
        );

        // If the new line token is not penalized, restore its logit value
        if (!params.isPenalizeNewLine()) {
            candidates.data().getPointer(this.newLineToken()).logit(newLineLogit);
        }

        return candidates;
    }

    @Override
    public void close() {
        if (this._context != null) {
            llama_free(this._context);
            this._context.close();
        }

        if (this._model != null) {
            llama_free_model(this._model);
            this._model.close();
        }

        if (this._params != null) {
            this._params.close();
        }
    }

    @Override
    public int contextSize() {
        return this._n_ctx;
    }

    @Override
    public int vocabSize() {
        return this._n_vocab;
    }

    @Override
    public int embeddingSize() {
        return this._n_embed;
    }

    @Override
    public int bosToken() {
        return this._token_bos;
    }

    @Override
    public int eosToken() {
        return this._token_eos;
    }

    @Override
    public int newLineToken() {
        return this._token_nl;
    }

    private llama_context_params initLLaMAContextParams(ModelParameters params) {
        llama_context_params llama_params = llama_context_default_params();

        int nGPULayers = Integer.parseInt(params.getExtra().getOrDefault("n_gpu_layers", "0"));
        llama_params.n_gpu_layers(nGPULayers);

        llama_params.n_ctx(params.getContextSize());
        llama_params.n_batch(params.getBatchSize());
        llama_params.seed(params.getSeed());

        llama_params.rope_freq_base(params.getRopeFreqBase());
        llama_params.rope_freq_scale(params.getRopeFreqScale());

        llama_params.embedding(params.isEmbeddingMode());

        return llama_params;
    }

    @Override
    public int nPastTokens() {
        return this.nPastTokens;
    }

    @Override
    public int[] inputTokens() {
        return ArrayUtils.subarray(this.inputTokens, 0, this.nPastTokens);
    }

    @Override
    public float[][] inputLogits() {
        return ArrayUtils.subarray(this.inputLogits, 0, this.nPastTokens);
    }

    private void debug(Runnable action) {
        if (this.modelParams.isVerbose()) {
            action.run();
        }
    }
}
