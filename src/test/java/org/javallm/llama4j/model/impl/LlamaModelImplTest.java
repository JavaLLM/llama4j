package org.javallm.llama4j.model.impl;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.javallm.llama4j.model.LlamaModel;
import org.javallm.llama4j.model.params.ModelParameters;
import org.javallm.llama4j.model.params.PenalizeParameters;
import org.javallm.llama4j.model.params.SamplingParameters;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;

public class LlamaModelImplTest {
    private static final String MODEL_PATH;

    // Use a tiny model for testing purpose
    static {
        try {
            File model = new File(LlamaModelImplTest.class.getClassLoader().getResource("TinyLLama-v0.ggmlv3.q4_0.bin").getFile());
            MODEL_PATH = model.getAbsolutePath();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void test_init_model() throws Exception {
        ModelParameters params = ModelParameters.builder()
                .modelPath(MODEL_PATH)
                .nThreads(8)
                .build();

        System.out.printf("Loading Llama model with parameters: %s", params);

        LlamaModel model = new LlamaModelImpl(params);
        assertThat(model).isNotNull();
        model.close();
    }

    @Test
    public void test_model_path_valid() {
        ModelParameters params = ModelParameters.builder()
                .modelPath(MODEL_PATH)
                .build();
        assertThat(params.isModelPathValid()).isTrue();
    }

    @Test
    public void test_init_model_path_not_exist() {
        ModelParameters params = ModelParameters.builder()
                .modelPath("NOT_EXISTS")
                .build();

        System.out.printf("Loading Llama model with parameters: %s", params);

        assertThatThrownBy(() -> new LlamaModelImpl(params))
                .isInstanceOf(Exception.class);
    }

    @Test
    public void test_tokenize() {
        ModelParameters params = ModelParameters.builder()
                .modelPath(MODEL_PATH)
                .nThreads(4)
                .build();
        LlamaModel model = new LlamaModelImpl(params);
        assertThat(model).isNotNull();

        String text = "你好！From LLaMA.cpp";
        int[] tokens = model.tokenize(text, false);
        System.out.println(Arrays.toString(tokens));
        String decoded = model.detokenize(tokens);
        System.out.printf("Decoded -> %s%n", decoded);
        assertThat(StringUtils.equals(text, decoded)).isTrue();
    }

    @Test
    public void test_embedding() {
        ModelParameters params = ModelParameters.builder()
                .verbose(true)
                .modelPath(MODEL_PATH)
                .embeddingMode(true)
                .nThreads(8)
                .build();

        LlamaModel model = new LlamaModelImpl(params);
        assertThat(model).isNotNull();

        String text = "Embedding 文本 from LLaMa.cpp";
        float[] embedding = model.embed(text);
        assertThat(embedding != null && embedding.length == model.embeddingSize()).isTrue();
        System.out.printf("Embedding -> %s%n", Arrays.toString(embedding));
    }

    @Test
    public void test_inference_simple() throws Exception {
        // Model initialization
        ModelParameters params = ModelParameters.builder()
                .modelPath(MODEL_PATH)
                .nThreads(4)
                .contextSize(2048)
                .build();
        LlamaModel model = new LlamaModelImpl(params);
        assertThat(model).isNotNull();

        // Prompt processing
        ArrayList<Integer> lastTokens = new ArrayList<>();

        String text = "Once upon a time, there was a little girl named Lily. She loved playing with her toys on top of her bed. One day, she decided to have a tea party with her stuffed animals. ";

        int[] tokens = model.tokenize(text, true);
        model.evaluate(tokens);

        for (int token : tokens) {
            lastTokens.add(token);
        }

        // Inference
        int maxTokens = 1000;
        PenalizeParameters penalizeParams = PenalizeParameters.builder().build();
        SamplingParameters samplingParameters = SamplingParameters.builder().build();

        for (int i = 0; i < maxTokens; i++) {
            int id = model.sample(samplingParameters, penalizeParams);
            lastTokens.add(id);

            if (id == model.eosToken()) {
                break;
            }

            model.evaluate(new int[]{id});
        }

        String response = model.detokenize(toArray(lastTokens));
        System.out.println(response);

        model.close();
    }

    @Test
    public void test_array_shift() {
        int[][] array = new int[2][3];
        array[0] = new int[]{1, 2, 3};
        array[1] = new int[]{4, 5, 6};

        ArrayUtils.shift(array, -1);
        System.out.printf("array[0] = %s; array[1] = %s", Arrays.toString(array[0]), Arrays.toString(array[1]));
    }

    @Test
    public void test_out_of_context() {
        // Model initialization
        ModelParameters params = ModelParameters.builder()
                .modelPath(MODEL_PATH)
                .nThreads(4)
                .contextSize(100)
                .batchSize(32)
                .build();
        LlamaModel model = new LlamaModelImpl(params);
        assertThat(model).isNotNull();

        String text = "LONG LONG AGO, IN A GALAXY, FAR FAR AWAY...";
        int[] tokens = model.tokenize(text, true);

        // repeat the tokens for 10 times
        int TIMES = 10;
        int[] manyTokens = new int[tokens.length * TIMES];
        for (int i = 0; i < TIMES; i++) {
            System.arraycopy(tokens, 0, manyTokens, i * tokens.length, tokens.length);
        }

        model.evaluate(manyTokens);

        int[] inputTokens = model.inputTokens();
        String decoded = model.detokenize(inputTokens);
        System.out.printf("Context Window = %s\n", decoded);
    }

    private int[] toArray(ArrayList<Integer> list) {
        return list.stream().mapToInt(i -> i).toArray();
    }
}