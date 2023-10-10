package org.javallm.llama4j.natives;

import org.assertj.core.data.Percentage;
import org.bytedeco.javacpp.BytePointer;
import org.javallm.llama4j.model.impl.LlamaModelImplTest;
import org.javallm.llamacpp.llama_context;
import org.javallm.llamacpp.llama_context_params;
import org.javallm.llamacpp.llama_model;
import org.javallm.llamacpp.llama_token_data;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.javallm.llamacpp.global.llama.*;

public class LlamaTest {
    private static final String MODEL_PATH;

    // Use a tiny model for testing purpose
    static {
        try {
            File model = new File(LlamaModelImplTest.class.getClassLoader().getResource("tinyllamas-stories-260k-f32.gguf").getFile());
            MODEL_PATH = model.getAbsolutePath();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void test_simple() {
        llama_token_data arr = new llama_token_data(10);

        llama_token_data item1 = new llama_token_data();
        item1.id(1);
        item1.p(.3f);
        item1.logit(.25f);

        arr.getPointer(2).put(item1.getPointer());

        llama_token_data result = arr.getPointer(2);
        assertThat(result).isNotNull();
        assertThat(result.id()).isEqualTo(1);
        assertThat(result.p()).isCloseTo(0.3f, Percentage.withPercentage(0.00001));
    }

    @Test
    public void test_print_system_info() {
        BytePointer info = llama_print_system_info();
        System.out.println(new String(info.getStringBytes(), StandardCharsets.UTF_8));
    }

    @Test
    public void test_init_llama_params() {
        llama_context_params params = llama_context_default_params();
        assertThat(params).isNotNull();
    }

    @Test
    public void test_load_model() {
        llama_context_params params = llama_context_default_params();
        llama_model model = llama_load_model_from_file(MODEL_PATH, params);
        assertThat(model).isNotNull();

        llama_context context = llama_new_context_with_model(model, params);
        assertThat(context).isNotNull();
    }
}
