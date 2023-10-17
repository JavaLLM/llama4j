package org.javallm.llama4j.model.params;

import lombok.Data;
import lombok.experimental.Accessors;

@Accessors(chain = true)
@Data
public final class SamplingParameters {
    /**
     * Adjust the randomness of the generated text (default: 0.8).
     * <p>
     * Note: Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.
     */
    private float temperature = 0.8f;

    // -------------------- Top-K Sampling --------------------
    /**
     * Limit the next token selection to the K most probable tokens (default: 40).
     * <p>
     * Note: Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top-k (e.g., 100) will consider more tokens and lead to a more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text. The default value is 40.
     */
    private int topK = 40;

    // -------------------- Top-P Sampling (Nucleus Sampling) --------------------
    /**
     * Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P (default: 0.9).
     * <p>
     * Note: Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top-p (e.g., 0.95) will lead to a more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. The default value is 0.9.
     */
    private float topP = .95f;

    // -------------------- Tail Free Sampling (TFS) --------------------
    /**
     * Enable tail free sampling with parameter z (default: 1.0, 1.0 = disabled).
     * <p>
     * Note: Tail free sampling (TFS) is a text generation technique that aims to reduce the impact of less likely tokens, which may be less relevant, less coherent, or nonsensical, on the output. Similar to Top-P it tries to determine the bulk of the most likely tokens dynamically. But TFS filters out logits based on the second derivative of their probabilities. Adding tokens is stopped after the sum of the second derivatives reaches the parameter z. In short: TFS looks how quickly the probabilities of the tokens decrease and cuts off the tail of unlikely tokens using the parameter z. Typical values for z are in the range of 0.9 to 0.95. A value of 1.0 would include all tokens, and thus disables the effect of TFS.
     */
    private float tsfZ = 1.0f;

    // -------------------- Locally Typical Sampling --------------------
    /**
     * Enable locally typical sampling with parameter p (default: 1.0, 1.0 = disabled).
     * <p>
     * Note: Locally typical sampling promotes the generation of contextually coherent and diverse text by sampling tokens that are typical or expected based on the surrounding context. By setting the parameter p between 0 and 1, you can control the balance between producing text that is locally coherent and diverse. A value closer to 1 will promote more contextually coherent tokens, while a value closer to 0 will promote more diverse tokens. A value equal to 1 disables locally typical sampling.
     */
    private float typicalP = 1.0f;

    // -------------------- Mirostat Sampling --------------------
    /**
     * Enable Mirostat sampling, controlling perplexity during text generation, see also {@link MirostatStrategy#getByCode}
     * <p>
     * Note: Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. It aims to strike a balance between coherence and diversity, avoiding low-quality output caused by excessive repetition (boredom traps) or incoherence (confusion traps).
     */
    private String miroStatStrategy = MirostatStrategy.DISABLE.getCode();

    /**
     * Set the Mirostat learning rate, parameter eta (default: 0.1).
     * <p>
     * Note: The learning rate influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. The default value is 0.1.
     */
    private float miroStatEta = 0.1f;

    /**
     * Set the Mirostat target entropy, parameter tau (default: 5.0).
     * <p>
     * Note: The target entropy represents the desired perplexity value for the generated text. Adjusting the target entropy allows you to control the balance between coherence and diversity in the generated text. A lower value will result in more focused and coherent text, while a higher value will lead to more diverse and potentially less coherent text. The default value is 5.0.
     */
    private float miroStatTau = 5.0f;
}
