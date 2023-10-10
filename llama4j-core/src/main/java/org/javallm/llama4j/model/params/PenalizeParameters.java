package org.javallm.llama4j.model.params;

import lombok.Data;
import lombok.experimental.Accessors;

@Accessors(chain = true)
@Data
public final class PenalizeParameters {
    /**
     * Control the repetition of token sequences in the generated text (default: 1.1).
     */
    private float repeatPenalty = 1.1f;

    /**
     * Last n tokens to consider for penalizing repetition (default: 64, 0 = disabled, -1 = ctx-size).
     */
    private int repeatLastTokensCount = 64;

    /**
     * Whether the new line `\n` should be considered as a repeatable token
     */
    private boolean penalizeNewLine = true;

    private float alphaFrequency = .0f;

    private float alphaPresence = .0f;

}
