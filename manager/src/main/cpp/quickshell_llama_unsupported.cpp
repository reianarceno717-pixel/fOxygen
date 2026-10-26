#include <jni.h>

extern "C" JNIEXPORT jlong JNICALL
Java_frb_axeron_manager_ai_OfflineLlama_nativeLoad(JNIEnv * env, jobject, jstring) {
    jclass exception_class = env->FindClass("java/lang/UnsupportedOperationException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, "Offline AI is available only on 64-bit ARM devices.");
    }
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_frb_axeron_manager_ai_OfflineLlama_nativeGenerate(JNIEnv * env, jobject, jlong, jstring) {
    jclass exception_class = env->FindClass("java/lang/UnsupportedOperationException");
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, "Offline AI is available only on 64-bit ARM devices.");
    }
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_frb_axeron_manager_ai_OfflineLlama_nativeRelease(JNIEnv *, jobject, jlong) {}
