package org.javallm.llama4j.model.params;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

@Builder
@ToString
public final class PenalizeParameters {
    /**
     * Control the repetition of token sequences in the generated text (default: 1.1).
     */
    @Builder.Default
    @Getter
    private float repeatPenalty = 1.1f;

    /**
     * Last n tokens to consider for penalizing repetition (default: 64, 0 = disabled, -1 = ctx-size).
     */
    @Builder.Default
    @Getter
    private int repeatLastTokensCount = 64;

    /**
     * Whether the new line `\n` should be considered as a repeatable token
     */
    @Builder.Default
    @Getter
    private boolean penalizeNewLine = true;

    @Builder.Default
    @Getter
    private float alphaFrequency = .0f;

    @Builder.Default
    @Getter
    private float alphaPresence = .0f;

}
