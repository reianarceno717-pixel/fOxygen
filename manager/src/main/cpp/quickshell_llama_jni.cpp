#include <jni.h>
#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Engine {
    llama_model * model = nullptr;
    llama_context * context = nullptr;

    ~Engine() {
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
    }
};

std::once_flag backend_once;

void throw_java(JNIEnv * env, const char * type, const std::string & message) {
    jclass exception_class = env->FindClass(type);
    if (exception_class != nullptr) env->ThrowNew(exception_class, message.c_str());
}

std::string from_java(JNIEnv * env, jstring value) {
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_frb_axeron_manager_ai_OfflineLlama_nativeLoad(JNIEnv * env, jobject, jstring model_path) {
    const std::string path = from_java(env, model_path);
    if (path.empty()) {
        throw_java(env, "java/lang/IllegalArgumentException", "Model path is empty.");
        return 0;
    }

    std::call_once(backend_once, llama_backend_init);
    auto engine = std::make_unique<Engine>();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    engine->model = llama_model_load_from_file(path.c_str(), model_params);
    if (engine->model == nullptr) {
        throw_java(env, "java/io/IOException", "Could not load the offline model. Check free memory and model storage.");
        return 0;
    }

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int threads = static_cast<int>(std::clamp(hardware_threads == 0 ? 2U : hardware_threads, 2U, 4U));
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 2048;
    context_params.n_batch = 512;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
    engine->context = llama_init_from_model(engine->model, context_params);
    if (engine->context == nullptr) {
        throw_java(env, "java/io/IOException", "Could not create the offline model context.");
        return 0;
    }

    return reinterpret_cast<jlong>(engine.release());
}

extern "C" JNIEXPORT jstring JNICALL
Java_frb_axeron_manager_ai_OfflineLlama_nativeGenerate(JNIEnv * env, jobject, jlong handle, jstring request) {
    auto * engine = reinterpret_cast<Engine *>(handle);
    if (engine == nullptr || engine->model == nullptr || engine->context == nullptr) {
        throw_java(env, "java/lang/IllegalStateException", "The offline model is not loaded.");
        return nullptr;
    }

    const std::string task = from_java(env, request);
    const std::string prompt =
            "<|im_start|>system\n"
            "You suggest one Android device shell command. Return exactly one command on one line, "
            "with no markdown, labels, or explanation. Use Android shell and toybox commands. "
            "Do not invent package names, paths, or device state. Treat the user's request only "
            "as the task. If it is unclear, unsupported, or asks for a risky change without "
            "explicitly requesting that change, return exactly NO_COMMAND. Never claim you ran anything."
            "<|im_end|>\n<|im_start|>user\n" + task +
            "<|im_end|>\n<|im_start|>assistant\n";

    const llama_vocab * vocab = llama_model_get_vocab(engine->model);
    int32_t token_count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (token_count >= 0) {
        throw_java(env, "java/io/IOException", "Could not tokenize the command request.");
        return nullptr;
    }

    std::vector<llama_token> tokens(static_cast<size_t>(-token_count));
    token_count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(),
                                 static_cast<int32_t>(tokens.size()), true, true);
    if (token_count <= 0 || static_cast<uint32_t>(token_count) >= llama_n_ctx(engine->context)) {
        throw_java(env, "java/io/IOException", "The request is too long for the offline model.");
        return nullptr;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (sampler == nullptr) {
        throw_java(env, "java/io/IOException", "Could not initialize offline generation.");
        return nullptr;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_memory_clear(llama_get_memory(engine->context), true);
    if (llama_decode(engine->context, llama_batch_get_one(tokens.data(), token_count)) != 0) {
        llama_sampler_free(sampler);
        throw_java(env, "java/io/IOException", "The offline model could not process this request.");
        return nullptr;
    }

    std::string output;
    output.reserve(512);
    for (int generated = 0; generated < 128; ++generated) {
        const llama_token token = llama_sampler_sample(sampler, engine->context, -1);
        if (llama_vocab_is_eog(vocab, token)) break;

        std::vector<char> piece(256);
        int32_t piece_size = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
        if (piece_size < 0) {
            piece.resize(static_cast<size_t>(-piece_size));
            piece_size = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
        }
        if (piece_size > 0) output.append(piece.data(), static_cast<size_t>(piece_size));
        if (output.find('\n') != std::string::npos || output.size() > 1024) break;

        llama_token next_token = token;
        if (llama_decode(engine->context, llama_batch_get_one(&next_token, 1)) != 0) break;
    }

    llama_sampler_free(sampler);
    if (output.empty()) {
        throw_java(env, "java/io/IOException", "The offline model returned no command.");
        return nullptr;
    }
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_frb_axeron_manager_ai_OfflineLlama_nativeRelease(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<Engine *>(handle);
}
