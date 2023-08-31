package org.javallm.llama4j.model.impl;

import com.google.common.collect.ImmutableMap;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.javallm.llama4j.TestUtils;
import org.javallm.llama4j.model.LlamaModel;
import org.javallm.llama4j.model.params.ModelParameters;
import org.javallm.llama4j.model.params.PenalizeParameters;
import org.javallm.llama4j.model.params.SamplingParameters;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;

public class LlamaModelImplTest {
    private static final String MODEL_PATH = TestUtils.getResourceAbsolutePath("tinyllamas-stories-260k-f32.gguf");
    private static final String Q8_0_MODEL_PATH = TestUtils.getResourceAbsolutePath("tinyllamas-stories-260k-q8_0.gguf");

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

        // remove leading BOS
        String decoded = model.detokenize(tokens).trim();

        System.out.printf("Decoded -> %s%n", decoded);
        assertThat(StringUtils.equals(text, decoded)).isTrue();
    }

    @Test
    public void test_detokenize_in_sequence() throws Exception {
        ModelParameters params = ModelParameters.builder()
                .modelPath(MODEL_PATH)
                .nThreads(4)
                .build();
        LlamaModel model = new LlamaModelImpl(params);

        String text = "你好！世界！";
        int[] tokens = model.tokenize(text, false);

        // BOS is not added
        for (int token : tokens) {
            assertThat(token).isNotEqualTo(model.bosToken());
        }

        StringBuilder builder = new StringBuilder();

        int i = 0, j = 0;
        int nPiece = 0;
        while (i < tokens.length) {
            while (true) {
                if (j > tokens.length) {
                    break;
                }
                String piece = model.detokenize(Arrays.copyOfRange(tokens, i, j));
                if (StringUtils.isBlank(piece)) {
                    j++;
                } else {
                    // left trim the first piece to remove the BOS token
                    if (nPiece == 0) {
                        piece = piece.replaceAll("^\\s+", "");
                    }
                    System.out.printf("tokens = [%d, %d] -> str = [%s]\n", i, j, piece);
                    builder.append(piece);
                    i = j;
                    nPiece++;
                    break;
                }
            }
        }

        String result = builder.toString();

        System.out.printf("Final output = [%s]\n", result);
        assertThat(StringUtils.equals(text, result)).isTrue();

        model.close();
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
    public void test_inference_stream_decode_with_GPU() throws Exception {
        // Model initialization
        ModelParameters params = ModelParameters.builder()
                .modelPath(Q8_0_MODEL_PATH) // Q8_0 is supported by GPU
                .nThreads(4)
                .contextSize(2048)
                .extra(ImmutableMap.of(
                        "n_gpu_layers", "4"
                ))
                .build();
        LlamaModel model = new LlamaModelImpl(params);
        assertThat(model).isNotNull();

        // Prompt processing
        String prompt = "Once upon a time, there was a little girl named Alice. She loved playing guitar and piano. One day, she";
        int[] tokens = model.tokenize(prompt, true);
        model.evaluate(tokens);

        System.out.println("\n==================== PROMPT ====================");
        System.out.print(prompt);

        // Inference
        int maxTokens = 2048;
        PenalizeParameters penalizeParams = PenalizeParameters.builder().build();
        SamplingParameters samplingParameters = SamplingParameters.builder().build();

        System.out.println("\n==================== RESPONSE ====================");

        ArrayList<Integer> cache = new ArrayList<>();
        for (int i = 0; i < maxTokens; i++) {
            int id = model.sample(samplingParameters, penalizeParams);
            if (id == model.eosToken()) {
                break;
            }

            // stream decode
            cache.add(id);
            String piece = model.detokenize(toArray(cache));
            if (StringUtils.isNotBlank(piece)) {
                cache.clear();
                System.out.print(piece);
            }

            model.evaluate(new int[]{id});

            Thread.sleep(1);
        }

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