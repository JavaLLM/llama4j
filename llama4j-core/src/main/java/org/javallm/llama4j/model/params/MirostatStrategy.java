package org.javallm.llama4j.model.params;

import lombok.Getter;
import org.apache.commons.lang3.StringUtils;

import javax.annotation.Nullable;

public enum MirostatStrategy {
    /**
     * Disabled
     */
    DISABLE("DISABLED", "Disable mirostat penalize strategy"),
    /**
     * Version 1
     */
    V1("MIRO_STAT_V1", "Use mirostat v1 strategy"),
    /**
     * Version 2
     */
    V2("MIRO_STAT_V2", "Use mirostat v2 strategy");

    @Getter
    private final String code;
    @Getter
    private final String description;

    MirostatStrategy(String code, String description) {
        this.code = code;
        this.description = description;
    }

    @Nullable
    public static MirostatStrategy getByCode(String code) {
        for (MirostatStrategy strategy : MirostatStrategy.values()) {
            if (StringUtils.equals(strategy.getCode(), code)) {
                return strategy;
            }
        }
        return null;
    }

    @Override
    public String toString() {
        return "MirostatStrategy{" +
                "code='" + code + '\'' +
                ", description='" + description + '\'' +
                '}';
    }
}
