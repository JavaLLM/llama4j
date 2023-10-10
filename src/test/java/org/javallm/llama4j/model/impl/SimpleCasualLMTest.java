/*
 * Ant Group
 * Copyright (c) 2004-2023 All Rights Reserved.
 */
package org.javallm.llama4j.model.impl;

import com.google.common.collect.ImmutableMap;
import org.javallm.llama4j.SimpleCasualLM;
import org.javallm.llama4j.TestUtils;
import org.javallm.llama4j.model.params.ModelParameters;
import org.junit.jupiter.api.Test;

/**
 * @author pengym
 * @version SimpleCasualLMTest.java, v 0.1 2023年09月02日 21:25 pengym
 */
public class SimpleCasualLMTest {
    private static final String MODEL_PATH = TestUtils.getResourceAbsolutePath("tinyllamas-stories-260k-f32.gguf");

    @Test
    public void test_infer() {
        SimpleCasualLM client = new SimpleCasualLM(MODEL_PATH);
        client.infer("Once upon a time, there was a little girl named Lily.", System.out::print);
    }

    @Test
    public void test_infer_gpu() {
        ModelParameters modelParameters = new ModelParameters()
                .setModelPath(MODEL_PATH)
                .setContextSize(2048)
                .setExtra(ImmutableMap.of(
                        "n_gpu_layers", "999"
                ));
        SimpleCasualLM client = new SimpleCasualLM(modelParameters);
        client.infer("Once upon a time, ", System.out::print);
    }
}