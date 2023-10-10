package org.javallm.llama4j.natives;

import org.assertj.core.data.Percentage;
import org.javallm.llamacpp.ggml_cgraph;
import org.javallm.llamacpp.ggml_context;
import org.javallm.llamacpp.ggml_init_params;
import org.javallm.llamacpp.ggml_tensor;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.javallm.llamacpp.global.llama.*;

public class GGMLTest {
    @Test
    public void test_compute_graph() {
        ggml_init_params init_params = new ggml_init_params();
        init_params.mem_size(128 * 1024 * 1024);
        init_params.no_alloc(false);
        init_params.mem_buffer(null);

        ggml_context context = ggml_init(init_params);
        ggml_tensor x = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);

        ggml_set_param(context, x);

        ggml_tensor a = ggml_new_tensor_1d(context, GGML_TYPE_F32, 1);
        ggml_tensor b = ggml_mul(context, x, x);
        ggml_tensor f = ggml_mul(context, b, a);

        ggml_print_objects(context);

        ggml_cgraph gf = ggml_build_forward(f);
        ggml_cgraph gb = ggml_build_backward(context, gf, false);

        ggml_set_f32(x, 2.0f);
        ggml_set_f32(a, 3.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(f.grad(), 1.0f);

        ggml_graph_compute_with_ctx(context, gb, 3);

        System.out.printf("f     = %f\n", ggml_get_f32_1d(f, 0));
        System.out.printf("df/dx = %f\n", ggml_get_f32_1d(x.grad(), 0));

        assertThat(ggml_get_f32_1d(f, 0)).isCloseTo(12.0f, Percentage.withPercentage(0.0000001));
        assertThat(ggml_get_f32_1d(x.grad(), 0)).isCloseTo(12.0f, Percentage.withPercentage(0.0000001));

        ggml_set_f32(x, 3.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(f.grad(), 1.0f);
        ggml_graph_compute_with_ctx(context, gb, 3);

        System.out.printf("f     = %f\n", ggml_get_f32_1d(f, 0));
        System.out.printf("df/dx = %f\n", ggml_get_f32_1d(x.grad(), 0));

        assertThat(ggml_get_f32_1d(f, 0)).isCloseTo(27.0f, Percentage.withPercentage(0.0000001));
        assertThat(ggml_get_f32_1d(x.grad(), 0)).isCloseTo(18.0f, Percentage.withPercentage(0.0000001));

        ggml_graph_dump_dot(gf, null, "test1-1-forward.dot");
        ggml_graph_dump_dot(gb, gf,  "test1-1-backward.dot");
        ggml_free(context);
    }

    @Test
    public void test_ggml_time_init() {
        ggml_time_init();
    }
}
