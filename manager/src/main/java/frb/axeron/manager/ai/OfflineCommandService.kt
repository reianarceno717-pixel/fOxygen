package frb.axeron.manager.ai

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlin.coroutines.CoroutineContext
import kotlinx.coroutines.currentCoroutineContext

data class OfflineCommandSuggestion(
    val command: String,
    val description: String = "Generated on this device. Review it before running."
)

/** Downloads the pinned GGUF once, then runs all command suggestions locally. */
class OfflineCommandService(
    context: Context,
    client: OkHttpClient
) {
    private val modelFile = File(context.filesDir, "offline_ai/$MODEL_FILENAME")
    private val downloadFile = File(modelFile.parentFile, "$MODEL_FILENAME.download")
    private val client = client.newBuilder()
        .cache(null)
        .callTimeout(90, TimeUnit.MINUTES)
        .readTimeout(2, TimeUnit.MINUTES)
        .build()
    private val nativeLock = Any()
    private var nativeHandle = 0L

    val isSupported: Boolean
        get() = Build.VERSION.SDK_INT >= MIN_ANDROID_API &&
                Build.SUPPORTED_ABIS.firstOrNull() == SUPPORTED_ABI

    fun isModelInstalled(): Boolean = modelFile.isFile && modelFile.length() == MODEL_SIZE_BYTES

    suspend fun downloadModel(onProgress: suspend (downloaded: Long, total: Long) -> Unit) {
        if (!isSupported) throw IOException("Offline AI requires Android 9 or newer on a 64-bit ARM device.")
        if (isModelInstalled()) return

        val callerContext: CoroutineContext = currentCoroutineContext()
        withContext(Dispatchers.IO) {
            modelFile.parentFile?.mkdirs()
                ?: throw IOException("Could not create the offline model folder.")
            val request = Request.Builder()
                .url(MODEL_URL)
                .header("Accept-Encoding", "identity")
                .build()

            try {
                client.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        throw IOException("Model download failed (HTTP ${response.code}).")
                    }
                    val body = response.body ?: throw IOException("The model download was empty.")
                    val digest = MessageDigest.getInstance("SHA-256")
                    var downloaded = 0L
                    var lastReported = 0L
                    body.byteStream().use { input ->
                        FileOutputStream(downloadFile).use { output ->
                            val buffer = ByteArray(256 * 1024)
                            while (true) {
                                callerContext.ensureActive()
                                val count = input.read(buffer)
                                if (count < 0) break
                                output.write(buffer, 0, count)
                                digest.update(buffer, 0, count)
                                downloaded += count
                                if (downloaded - lastReported >= PROGRESS_STEP_BYTES) {
                                    lastReported = downloaded
                                    onProgress(downloaded, MODEL_SIZE_BYTES)
                                }
                            }
                            output.fd.sync()
                        }
                    }

                    if (downloaded != MODEL_SIZE_BYTES) {
                        throw IOException("The downloaded model has the wrong size. Please try again.")
                    }
                    val actualSha256 = digest.digest().joinToString("") {
                        (it.toInt() and 0xff).toString(16).padStart(2, '0')
                    }
                    if (!actualSha256.equals(MODEL_SHA256, ignoreCase = true)) {
                        throw IOException("The downloaded model failed its integrity check. Please try again.")
                    }
                    if (modelFile.exists() && !modelFile.delete()) {
                        throw IOException("Could not replace the existing model file.")
                    }
                    if (!downloadFile.renameTo(modelFile)) {
                        throw IOException("Could not install the downloaded model.")
                    }
                    onProgress(MODEL_SIZE_BYTES, MODEL_SIZE_BYTES)
                }
            } finally {
                downloadFile.delete()
            }
        }
    }

    suspend fun suggestCommand(request: String): OfflineCommandSuggestion? {
        val task = request.trim()
        require(task.isNotEmpty()) { "Describe the command you need." }
        require(task.length <= MAX_REQUEST_LENGTH) { "Keep the request under $MAX_REQUEST_LENGTH characters." }
        if (!isSupported) throw IOException("Offline AI requires Android 9 or newer on a 64-bit ARM device.")
        if (!isModelInstalled()) throw IOException("Download the offline model first. It is needed only once.")

        return withContext(Dispatchers.Default) {
            synchronized(nativeLock) {
                if (nativeHandle == 0L) {
                    nativeHandle = OfflineLlama.nativeLoad(modelFile.absolutePath)
                    if (nativeHandle == 0L) throw IOException("The offline model could not be loaded.")
                }
                val output = OfflineLlama.nativeGenerate(nativeHandle, task)
                    .trim()
                    .removeSurrounding("```")
                    .trim()
                if (output.isBlank() || output.equals("NO_COMMAND", ignoreCase = true)) return@synchronized null
                if (output.lineSequence().drop(1).any(String::isNotBlank)) {
                    throw IOException("The model returned more than one line. Try a shorter, clearer request.")
                }
                val command = output.lineSequence().first()
                    .trim()
                    .removePrefix("Command:")
                    .trim()
                if (command.isBlank() || command.equals("NO_COMMAND", ignoreCase = true)) null
                else OfflineCommandSuggestion(command)
            }
        }
    }

    fun close() {
        synchronized(nativeLock) {
            val handle = nativeHandle
            nativeHandle = 0L
            if (handle != 0L) OfflineLlama.nativeRelease(handle)
        }
    }

    companion object {
        const val MODEL_SIZE_LABEL = "491 MB"
        const val MODEL_DISPLAY_NAME = "Qwen2.5-Coder 0.5B · Q4_K_M"
        private const val MIN_ANDROID_API = 28
        private const val SUPPORTED_ABI = "arm64-v8a"
        private const val MAX_REQUEST_LENGTH = 800
        private const val PROGRESS_STEP_BYTES = 4L * 1024 * 1024
        private const val MODEL_SIZE_BYTES = 491_400_064L
        private const val MODEL_SHA256 = "1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32"
        private const val MODEL_FILENAME = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
        private const val MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/ebb2015119c907b064c512bf053e945850b5875f/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf?download=true"
    }
}

private object OfflineLlama {
    init {
        System.loadLibrary("quicksh_ai")
    }

    external fun nativeLoad(modelPath: String): Long
    external fun nativeGenerate(handle: Long, request: String): String
    external fun nativeRelease(handle: Long)
}
