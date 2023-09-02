# llama4j
> Note: This library is still under active development and is not yet ready for production use.

An easy-to-sse Java SDK for running the [LLaMA](https://ai.meta.com/llama/) ([v1](https://arxiv.org/abs/2302.13971) and [v2](https://arxiv.org/abs/2307.09288)) models and their variants on edge devices, powered by [LLaMA.cpp](https://github.com/ggerganov/llama.cpp).

## Get Started
Using llama4j with GGUF models is pretty easy. First add the related dependencies to your `pom.xml`:
```xml
<dependency>
    <groupId>org.javallm</groupId>
    <artifactId>llama4j</artifactId>
    <version>0.0.1</version>
</dependency>
```
Then, you should download GGML models from huggingface or somewhere else. Notice that only models with `.gguf` suffix since this is the latest format that the upstream `LLaMA.cpp` supports.

Now, you can happily play with your model! Here is a simple example:
```java
SimpleCasualLM client = new SimpleCasualLM(GGML_MODEL_PATH);
client.infer("Once upon a time, there was a little girl named Lily.", System.out::print);
```