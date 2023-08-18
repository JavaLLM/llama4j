package org.javallm.llama4j.model;

import org.javallm.llama4j.model.params.PenalizeParameters;
import org.javallm.llama4j.model.params.SamplingParameters;

/**
 * The LLaMA model interface
 */
public interface LlamaModel extends AutoCloseable {
    /**
     * Convert a text to a sequence of tokens
     * @param text text to be tokenized
     * @param addBos boolean value to indicate whether to add a special <BOS> token (i.e., the beginning of a sequence) to the resulting tokens
     * @return the resulting tokens
     */
    int[] tokenize(String text, boolean addBos);

    default int[] tokenize(String text) {
        return tokenize(text, true);
    }

    /**
     * Recover text from tokens
     * @param tokens tokens to be de-tokenized
     * @return the resulting text
     */
    String detokenize(int[] tokens);

    default void reset() {
        reset(0);
    }

    void reset(int nPastTokens);

    /**
     * Evaluate tokens
     * @param tokens tokens to be evaluated
     */
    void evaluate(int[] tokens);

    /**
     * Perform sampling in an auto-regressive manner
     * @param samplingParams parameters related to sampling
     * @param penalizeParameters parameter related to penalization
     */
    int sample(SamplingParameters samplingParams, PenalizeParameters penalizeParameters);

    default int sample() {
        SamplingParameters samplingParams = SamplingParameters.builder().build();
        PenalizeParameters penalizeParams = PenalizeParameters.builder().build();

        return sample(samplingParams, penalizeParams);
    }

    /**
     * Embed an input string with the model
     * @param input the input string
     * @return the embedding
     * @throws UnsupportedOperationException if the underlying model does not support embedding
     */
    default float[] embed(String input) {
        throw new UnsupportedOperationException();
    }

    /**
     * Get the model state data, useful for persisting model state
     *
     * @return the model state data in bytes
     */
    byte[] getState();

    /**
     * Load the model state
     *
     * @param state model state data
     */
    void loadState(byte[] state);

    int bosToken();

    /**
     * @return the id of the <EOS> token
     */
    int eosToken();

    /**
     * @return the id of the new line (i.e., \n) token
     */
    int newLineToken();

    int embeddingSize();

    int contextSize();

    int vocabSize();

    int nPastTokens();

    int[] inputTokens();

    float[][] inputLogits();
}
