package org.javallm.llama4j;

import org.apache.commons.lang3.StringUtils;
import org.javallm.llama4j.model.LlamaModel;
import org.javallm.llama4j.model.impl.LlamaModelImpl;
import org.javallm.llama4j.model.params.ModelParameters;
import org.javallm.llama4j.model.params.PenalizeParameters;
import org.javallm.llama4j.model.params.SamplingParameters;
import org.javallm.llamacpp.global.llama;

import java.util.ArrayList;
import java.util.function.Consumer;

/**
 * @author pengym
 * @version SimpleCasualLM.java, v 0.1 2023年09月02日 21:11 pengym
 */
public class SimpleCasualLM {
    private final LlamaModel model;

    public SimpleCasualLM(String path) {
        ModelParameters params = ModelParameters.builder()
                .modelPath(path)
                .nThreads(8)
                .contextSize(2048)
                .batchSize(512)
                .build();
        model = new LlamaModelImpl(params);
    }

    public SimpleCasualLM(ModelParameters parameters) {
        model = new LlamaModelImpl(parameters);
    }

    public void infer(String prompt, Consumer<String> callback) {
        SamplingParameters samplingParams = SamplingParameters
                .builder()
                .build();
        PenalizeParameters penalizeParams = PenalizeParameters
                .builder()
                .build();
        infer(prompt, samplingParams, penalizeParams, callback);
    }

    public void infer(String prompt, SamplingParameters samplingParams, PenalizeParameters penalizeParams, Consumer<String> callback) {
        int[] tokens = model.tokenize(prompt, true);
        model.evaluate(tokens);

        ArrayList<Integer> cache = new ArrayList<>();
        int maxTokens = model.contextSize();
        for (int i = 0; i < maxTokens; i++) {
            int id = model.sample(samplingParams, penalizeParams);
            if (id == model.eosToken()) {
                break;
            }
            cache.add(id);
            String piece = model.detokenize(toArray(cache));
            if (StringUtils.isNotBlank(piece)) {
                cache.clear();
                callback.accept(piece);
            }

            model.evaluate(new int[]{id});
        }

        model.reset();
    }

    private int[] toArray(ArrayList<Integer> list) {
        return list.stream().mapToInt(i -> i).toArray();
    }
}
