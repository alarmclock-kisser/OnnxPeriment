using OnnxPeriment.Core;
using OnnxPeriment.Shared;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using LLama;
using LLama.Common;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace OnnxPeriment.Runtime
{
    public class LlamaService
    {
        private string? _loadedModelPath;
        private LLamaWeights? _weights;
        private LLamaContext? _context;
        private readonly Dictionary<string, LlamaChatContext> _contexts = new(StringComparer.OrdinalIgnoreCase);

        public string? LoadedContextFile { get; private set; }

        private const string RecentContextName = "recent";
        private const string DefaultContextName = "default";
        private const string BackendNotConfiguredMarker = "no inference backend is configured";

        public LlamaResponseStats? LastStats { get; private set; }

        private sealed class ManagedContext
        {
            public required LlamaChatContext Context { get; init; }
            public bool AutoSave { get; init; }
            public bool IsTransient { get; init; }
        }

        public string? LoadedModelPath => this._loadedModelPath;
        public bool ModelIsLoaded => !string.IsNullOrWhiteSpace(this._loadedModelPath);

        public List<string> ModelDirectories { get; private set; } =
        [
            "D:\\Models",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), ".lmstudio", "models"),
            "D:\\Programme\\LLM_MODELS"
        ];

        public List<string> ModelPaths { get; private set; } = [];

        public LlamaService(string[]? additionalDirectories = null)
        {
            this.ModelPaths = this.GetAvailableModelPaths(additionalDirectories);
        }

        public List<string> GetAvailableModelPaths(string[]? additionalDirectories = null, bool includeSubDirectories = true)
        {
            List<string> dirs = [.. this.ModelDirectories];
            if (additionalDirectories != null)
            {
                dirs.AddRange(additionalDirectories);
            }

            List<string> modelPaths = dirs
                .Where(Directory.Exists)
                .SelectMany(dir => Directory.GetFiles(dir, "*.gguf", includeSubDirectories ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly))
                .ToList();

            StaticLogger.Log($"Found {modelPaths.Count} GGUF model(s).");
            this.ModelPaths = modelPaths;
            return modelPaths;
        }

        public List<LlamaBackendStatus> VerifyBackends(bool testLoad = true)
        {
            var searchRoots = new List<string>
            {
                AppContext.BaseDirectory,
                Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native")
            };

            searchRoots = searchRoots
                .Where(path => Directory.Exists(path))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            var probes = new[]
            {
                new
                {
                    Name = "CPU",
                    Assemblies = new[] { "LLamaSharp.Backend.Cpu" },
                    NativeCandidates = new[] { "llama.dll" }
                },
                new
                {
                    Name = "CUDA12",
                    Assemblies = new[] { "LLamaSharp.Backend.Cuda12", "LLamaSharp.Backend.Cuda12.Windows" },
                    NativeCandidates = new[] { "ggml-cuda.dll", "llama.dll" }
                }
            };

            var results = new List<LlamaBackendStatus>();

            foreach (var probe in probes)
            {
                string? loadedAssemblyName = null;
                bool assemblyAvailable = false;
                string? error = null;

                foreach (var assemblyName in probe.Assemblies)
                {
                    try
                    {
                        Assembly.Load(assemblyName);
                        loadedAssemblyName = assemblyName;
                        assemblyAvailable = true;
                        break;
                    }
                    catch
                    {
                        // ignore and try next
                    }
                }

                string? nativePath = null;
                foreach (var candidate in probe.NativeCandidates)
                {
                    nativePath = FindNativeLibrary(searchRoots, candidate);
                    if (nativePath != null)
                    {
                        break;
                    }
                }

                bool nativeFound = nativePath != null;
                bool nativeLoadable = false;

                if (testLoad && nativePath != null)
                {
                    try
                    {
                        if (NativeLibrary.TryLoad(nativePath, out var handle))
                        {
                            nativeLoadable = true;
                            NativeLibrary.Free(handle);
                        }
                    }
                    catch (Exception ex)
                    {
                        error = ex.Message;
                    }
                }

                results.Add(new LlamaBackendStatus
                {
                    Name = probe.Name,
                    AssemblyName = loadedAssemblyName,
                    AssemblyAvailable = assemblyAvailable,
                    NativeLibraryPath = nativePath,
                    NativeLibraryFound = nativeFound,
                    NativeLibraryLoadable = nativeLoadable,
                    Error = error
                });
            }

            foreach (var result in results)
            {
                StaticLogger.Log($"Backend {result.Name}: assembly={(result.AssemblyAvailable ? result.AssemblyName : "missing")}, native={(result.NativeLibraryFound ? result.NativeLibraryPath : "missing")}, loadable={result.NativeLibraryLoadable}");
                if (!string.IsNullOrWhiteSpace(result.Error))
                {
                    StaticLogger.Log($"Backend {result.Name} error: {result.Error}");
                }
            }

            return results;
        }

        public async Task<IReadOnlyList<string>> GetContextsAsync()
        {
            return await Task.Run(() =>
            {
                var dir = GetContextDirectory();
                Directory.CreateDirectory(dir);

                return Directory.EnumerateFiles(dir, "*.json")
                    .Select(path => new FileInfo(path))
                    .OrderByDescending(info => info.LastWriteTimeUtc)
                    .Select(info => Path.GetFileNameWithoutExtension(info.Name))
                    .ToList();
            });
        }

        public async Task<LlamaChatContext?> LoadContextAsync(string nameOrRecent)
        {
            var normalized = NormalizeContextName(nameOrRecent);
            if (string.IsNullOrWhiteSpace(normalized))
            {
                return null;
            }

            return await Task.Run(() =>
            {
                var resolved = ResolveContextFilePath(normalized);
                if (string.IsNullOrWhiteSpace(resolved) || !File.Exists(resolved))
                {
                    return null;
                }

                var context = LoadContextFromFile(resolved);
                if (context == null)
                {
                    return null;
                }

                _contexts[context.Name] = context;
                this.LoadedContextFile = resolved;
                return context;
            });
        }

        public async Task<LlamaChatContext> CreateContextAsync(string name = "")
        {
            return await Task.Run(() =>
            {
                var context = new LlamaChatContext
                {
                    Name = string.IsNullOrWhiteSpace(name) ? GenerateTransientContextName() : name,
                    UpdatedAt = DateTime.UtcNow
                };

                if (!string.IsNullOrWhiteSpace(name))
                {
                    _contexts[context.Name] = context;
                }

                return context;
            });
        }

        public async Task<bool> SaveContextAsync(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                return false;
            }

            if (!_contexts.TryGetValue(name, out var context))
            {
                return false;
            }

            return await SaveContextAsync(context);
        }

        public async Task<bool> SaveContextAsync(LlamaChatContext context)
        {
            if (context == null || string.IsNullOrWhiteSpace(context.Name))
            {
                return false;
            }

            return await Task.Run(() =>
            {
                var dir = GetContextDirectory();
                Directory.CreateDirectory(dir);
                var safeName = SanitizeContextName(context.Name);
                var path = Path.Combine(dir, safeName + ".json");
                context.UpdatedAt = DateTime.UtcNow;

                var json = JsonSerializer.Serialize(context, new JsonSerializerOptions
                {
                    WriteIndented = true
                });
                File.WriteAllText(path, json, Encoding.UTF8);
                _contexts[context.Name] = context;
                this.LoadedContextFile = path;
                return true;
            });
        }

        public async Task<bool> DeleteContextAsync(string name)
        {
            var normalized = NormalizeContextName(name);
            if (string.IsNullOrWhiteSpace(normalized))
            {
                return false;
            }

            return await Task.Run(() =>
            {
                var dir = GetContextDirectory();
                var safeName = SanitizeContextName(normalized);
                var path = Path.Combine(dir, safeName + ".json");

                if (!File.Exists(path))
                {
                    _contexts.Remove(name);
                    return false;
                }

                File.Delete(path);
                _contexts.Remove(name);
                return true;
            });
        }

        private static string GetContextDirectory()
        {
            var baseDir = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            return Path.Combine(baseDir, "OnnxPeriment_Contexts");
        }

        private static string? ResolveContextFilePath(string nameOrRecent)
        {
            var dir = GetContextDirectory();
            Directory.CreateDirectory(dir);

            if (string.Equals(nameOrRecent, RecentContextName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(nameOrRecent, DefaultContextName, StringComparison.OrdinalIgnoreCase))
            {
                var recent = Directory.EnumerateFiles(dir, "*.json")
                    .Select(path => new FileInfo(path))
                    .OrderByDescending(info => info.LastWriteTimeUtc)
                    .FirstOrDefault();

                if (recent != null)
                {
                    return recent.FullName;
                }

                return Path.Combine(dir, SanitizeContextName(DefaultContextName) + ".json");
            }

            var safeName = SanitizeContextName(nameOrRecent);
            return Path.Combine(dir, safeName + ".json");
        }

        private static LlamaChatContext? LoadContextFromFile(string path)
        {
            try
            {
                var json = File.ReadAllText(path, Encoding.UTF8);
                var context = JsonSerializer.Deserialize<LlamaChatContext>(json);
                if (context != null)
                {
                    context.UpdatedAt = File.GetLastWriteTimeUtc(path);
                    return context;
                }
            }
            catch
            {
                // ignore
            }

            return null;
        }

        private static string SanitizeContextName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            var sb = new StringBuilder(name.Length);
            foreach (var ch in name)
            {
                sb.Append(invalid.Contains(ch) ? '_' : ch);
            }
            return sb.ToString();
        }

        private static string GenerateTransientContextName()
        {
            return "transient-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff");
        }

        private async Task<ManagedContext?> ResolveContextAsync(string? contextName)
        {
            if (contextName == null)
            {
                return null;
            }

            var normalized = NormalizeContextName(contextName);
            if (string.IsNullOrWhiteSpace(normalized))
            {
                var transient = new LlamaChatContext
                {
                    Name = GenerateTransientContextName(),
                    UpdatedAt = DateTime.UtcNow
                };
                return new ManagedContext
                {
                    Context = transient,
                    AutoSave = false,
                    IsTransient = true
                };
            }

            if (string.Equals(normalized, RecentContextName, StringComparison.OrdinalIgnoreCase)
                || string.Equals(normalized, DefaultContextName, StringComparison.OrdinalIgnoreCase))
            {
                var resolved = ResolveContextFilePath(normalized);
                if (resolved != null && File.Exists(resolved))
                {
                    var loaded = LoadContextFromFile(resolved);
                    if (loaded != null)
                    {
                        _contexts[loaded.Name] = loaded;
                        return new ManagedContext { Context = loaded, AutoSave = true, IsTransient = false };
                    }
                }

                var fallback = new LlamaChatContext
                {
                    Name = DefaultContextName,
                    UpdatedAt = DateTime.UtcNow
                };
                _contexts[fallback.Name] = fallback;
                return new ManagedContext { Context = fallback, AutoSave = true, IsTransient = false };
            }

            if (_contexts.TryGetValue(normalized, out var cached))
            {
                return new ManagedContext { Context = cached, AutoSave = false, IsTransient = false };
            }

            var path = ResolveContextFilePath(normalized);
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
            {
                var loaded = LoadContextFromFile(path);
                if (loaded != null)
                {
                    _contexts[loaded.Name] = loaded;
                    return new ManagedContext { Context = loaded, AutoSave = true, IsTransient = false };
                }
            }

            var created = new LlamaChatContext
            {
                Name = normalized,
                UpdatedAt = DateTime.UtcNow
            };
            _contexts[created.Name] = created;
            return new ManagedContext { Context = created, AutoSave = false, IsTransient = false };
        }

        private async Task FinalizeContextAsync(ManagedContext? managed, string prompt, string response)
        {
            if (managed == null)
            {
                return;
            }

            if (!string.IsNullOrWhiteSpace(prompt))
            {
                managed.Context.Messages.Add(new LlamaChatMessage
                {
                    Role = "user",
                    Content = prompt
                });
            }

            if (!string.IsNullOrWhiteSpace(response)
                && response.IndexOf(BackendNotConfiguredMarker, StringComparison.OrdinalIgnoreCase) < 0)
            {
                managed.Context.Messages.Add(new LlamaChatMessage
                {
                    Role = "assistant",
                    Content = response
                });
            }

            managed.Context.UpdatedAt = DateTime.UtcNow;

            if (managed.AutoSave)
            {
                await SaveContextAsync(managed.Context);
            }
            else if (managed.IsTransient)
            {
                // drop transient context
            }
        }

        private static string BuildPrompt(LlamaChatContext? context, string prompt)
        {
            if (context == null || context.Messages.Count == 0)
            {
                return prompt;
            }

            var sb = new StringBuilder();
            var history = context.Messages
                .Where(message => message.Content.IndexOf(BackendNotConfiguredMarker, StringComparison.OrdinalIgnoreCase) < 0)
                .TakeLast(10)
                .ToList();

            foreach (var message in history)
            {
                sb.Append(message.Role).Append(": ").AppendLine(message.Content);
            }

            sb.Append("user: ").AppendLine(prompt);
            sb.Append("assistant: ");
            return sb.ToString();
        }

        public async Task<string?> LoadModelAsync(string modelPathOrName, LlamaModelLoadOptions? options = null)
        {
            options ??= new LlamaModelLoadOptions();
            string? modelPath = modelPathOrName;
            if (!File.Exists(modelPathOrName))
            {
                modelPath = this.ModelPaths.FirstOrDefault(p => Path.GetFileNameWithoutExtension(p).Equals(modelPathOrName, StringComparison.OrdinalIgnoreCase));
                if (modelPath == null)
                {
                    StaticLogger.Log($"Model '{modelPathOrName}' not found.");
                    return null;
                }
            }

            StaticLogger.Log($"Loading model from '{modelPath}' with CUDA: {options.UseCuda}...");

            try
            {
                this.DisposeModel();

                var fullPath = Path.GetFullPath(modelPath);
                var gpuLayers = options.UseCuda
                    ? (options.GpuLayerCount > 0 ? options.GpuLayerCount : 999)
                    : 0;

                var modelParams = new ModelParams(fullPath)
                {
                    ContextSize = options.ContextSize,
                    GpuLayerCount = gpuLayers
                };

                await Task.Run(() =>
                {
                    this._weights = LLamaWeights.LoadFromFile(modelParams);
                    this._context = this._weights.CreateContext(modelParams);
                });

                this._loadedModelPath = fullPath;
                StaticLogger.Log($"Model loaded: {fullPath} (GPU layers: {gpuLayers}, context: {options.ContextSize})");
                return fullPath;
            }
            catch (Exception ex)
            {
                this.DisposeModel();
                StaticLogger.Log(ex);
                return null;
            }
        }

        public bool UnloadModel()
        {
            StaticLogger.Log("Unloading model...");
            this.DisposeModel();
            return true;
        }

        // Streaming (primary) + aggregation helpers
        public async IAsyncEnumerable<LlamaStreamChunk> GenerateTextStreamingAsync(
            LlamaGenerateRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var sw = Stopwatch.StartNew();
            this.LastStats = null;

            if (string.IsNullOrWhiteSpace(request.Prompt))
            {
                yield return new LlamaStreamChunk
                {
                    Content = "Prompt is empty.",
                    Index = 0,
                    IsFinal = true,
                    FinishReason = "error"
                };
                yield break;
            }

            if (!this.ModelIsLoaded || this._context == null)
            {
                yield return new LlamaStreamChunk
                {
                    Content = "Model not loaded.",
                    Index = 0,
                    IsFinal = true,
                    FinishReason = "error"
                };
                yield break;
            }

            var managedContext = await this.ResolveContextAsync(request.Context);

            var prompt = BuildPrompt(managedContext?.Context, request.Prompt);
            var inferenceParams = new InferenceParams
            {
                MaxTokens = request.Options.MaxTokens
            };

            var stopSequences = GetStopSequences(request.Options);
            var executor = new InteractiveExecutor(this._context);
            var responseBuilder = new StringBuilder();
            var chunkBuffer = new StringBuilder();
            int chunkSize = Math.Max(1, request.Options.StreamChunkSize);
            int index = 0;
            int emittedLength = 0;
            bool stopReached = false;

            await foreach (var piece in executor.InferAsync(prompt, inferenceParams, cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (string.IsNullOrEmpty(piece))
                {
                    continue;
                }

                responseBuilder.Append(piece);
                chunkBuffer.Append(piece);

                var stopIndex = FindStopIndex(responseBuilder.ToString(), stopSequences, emittedLength);
                if (stopIndex >= 0)
                {
                    responseBuilder.Length = stopIndex;
                    var remaining = responseBuilder.Length - emittedLength;
                    chunkBuffer.Clear();
                    if (remaining > 0)
                    {
                        chunkBuffer.Append(responseBuilder.ToString(emittedLength, remaining));
                    }
                    stopReached = true;
                }

                while (chunkBuffer.Length >= chunkSize)
                {
                    var content = chunkBuffer.ToString(0, chunkSize);
                    chunkBuffer.Remove(0, chunkSize);
                    emittedLength += content.Length;

                    yield return new LlamaStreamChunk
                    {
                        Content = content,
                        Index = index++,
                        IsFinal = false,
                        FinishReason = null
                    };
                }

                if (stopReached)
                {
                    break;
                }
            }

            if (chunkBuffer.Length > 0)
            {
                var content = chunkBuffer.ToString();
                emittedLength += content.Length;
                yield return new LlamaStreamChunk
                {
                    Content = content,
                    Index = index++,
                    IsFinal = !request.Options.AppendStatsToResponse,
                    FinishReason = !request.Options.AppendStatsToResponse ? "stop" : null
                };
            }

            sw.Stop();

            var response = responseBuilder.ToString();

            if (request.Options.AppendStatsToResponse)
            {
                var stats = BuildStats(response, sw.Elapsed);
                this.LastStats = stats;
                var statsText = $"\n\n[Stats] tokens={stats.TokenCount}, elapsed={stats.ElapsedSeconds:F2}s, tokens/s={stats.TokensPerSecond:F2}";
                yield return new LlamaStreamChunk
                {
                    Content = statsText,
                    Index = index,
                    IsFinal = true,
                    FinishReason = "stop",
                    Stats = stats
                };
            }

            await FinalizeContextAsync(managedContext, request.Prompt, response);
        }

        private static IReadOnlyList<string> GetStopSequences(LlamaGenerateOptions options)
        {
            var sequences = options.StopSequences?.Where(s => !string.IsNullOrWhiteSpace(s)).ToList() ?? new List<string>();
            if (sequences.Count == 0)
            {
                sequences.Add("user:");
                sequences.Add("assistant:");
                sequences.Add("<|user|>");
                sequences.Add("<|assistant|>");
            }

            return sequences;
        }

        private static int FindStopIndex(string text, IReadOnlyList<string> stopSequences, int startIndex)
        {
            int minIndex = -1;
            foreach (var sequence in stopSequences)
            {
                var index = text.IndexOf(sequence, startIndex, StringComparison.OrdinalIgnoreCase);
                if (index >= 0 && (minIndex < 0 || index < minIndex))
                {
                    minIndex = index;
                }
            }

            return minIndex;
        }

        public async Task<string?> GenerateTextToTextAsync(string prompt, LlamaGenerateOptions? options = null, string? context = "/recent")
        {
            options ??= new LlamaGenerateOptions();

            StaticLogger.Log($"Generating text from prompt: {prompt} with options: {options}...");

            try
            {
                var request = new LlamaGenerateRequest
                {
                    Prompt = prompt,
                    Options = options,
                    Context = context
                };

                var sb = new StringBuilder();
                await foreach (var chunk in this.GenerateTextStreamingAsync(request))
                {
                    if (chunk.Stats != null)
                    {
                        this.LastStats = chunk.Stats;
                        continue;
                    }

                    sb.Append(chunk.Content);
                }

                return sb.Length > 0 ? sb.ToString() : null;
            }
            catch (Exception ex)
            {
                StaticLogger.Log(ex);
                return null;
            }
        }

        public async Task<string?> GenerateTextWithImagesToTextAsync(string prompt, string[]? base64Images = null, LlamaGenerateOptions? options = null, string? context = null)
        {
            options ??= new LlamaGenerateOptions();
            if (base64Images == null || base64Images.Count() <= 0)
            {
                return await GenerateTextToTextAsync(prompt, options, context);
            }

            StaticLogger.Log($"Generating text from prompt with {base64Images?.Length ?? 0} images: {prompt} with options: {options}...");

            try
            {
                var request = new LlamaGenerateRequest
                {
                    Prompt = prompt,
                    Base64Images = base64Images,
                    Options = options,
                    Context = context
                };

                var sb = new StringBuilder();
                await foreach (var chunk in this.GenerateTextStreamingAsync(request))
                {
                    if (chunk.Stats != null)
                    {
                        this.LastStats = chunk.Stats;
                        continue;
                    }

                    sb.Append(chunk.Content);
                }

                return sb.Length > 0 ? sb.ToString() : null;
            }
            catch (Exception ex)
            {
                StaticLogger.Log(ex);
                return null;
            }
        }

        private void DisposeModel()
        {
            try
            {
                this._context?.Dispose();
                this._weights?.Dispose();
            }
            catch (Exception ex)
            {
                StaticLogger.Log(ex);
            }
            finally
            {
                this._context = null;
                this._weights = null;
                this._loadedModelPath = null;
            }
        }

        private static LlamaResponseStats BuildStats(string response, TimeSpan elapsed)
        {
            int tokenCount = CountTokens(response);
            double seconds = Math.Max(0.0001, elapsed.TotalSeconds);
            return new LlamaResponseStats
            {
                TokenCount = tokenCount,
                ElapsedSeconds = elapsed.TotalSeconds,
                TokensPerSecond = tokenCount / seconds
            };
        }

        private static int CountTokens(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return 0;
            }

            return text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
        }

        private static string? FindNativeLibrary(IEnumerable<string> searchRoots, string fileName)
        {
            foreach (var root in searchRoots)
            {
                var direct = Path.Combine(root, fileName);
                if (File.Exists(direct))
                {
                    return direct;
                }

                try
                {
                    var match = Directory.EnumerateFiles(root, fileName, SearchOption.AllDirectories)
                        .FirstOrDefault();
                    if (!string.IsNullOrEmpty(match))
                    {
                        return match;
                    }
                }
                catch
                {
                    // ignore path access issues
                }
            }

            return null;
        }

        public async Task<IReadOnlyList<string>> EnsureCudaDependenciesAsync(string? cudaRoot = null, string? outputDir = null)
        {
            return await Task.Run(() =>
            {
                var logs = new List<string>();

                string? resolvedCudaRoot = ResolveCudaRoot(cudaRoot);
                if (string.IsNullOrWhiteSpace(resolvedCudaRoot))
                {
                    logs.Add("CUDA root not found. Set CUDA_PATH or ensure nvcc is in PATH.");
                    StaticLogger.Log("CUDA root not found. Set CUDA_PATH or ensure nvcc is in PATH.");
                    return logs;
                }

                var cudaBin = Path.Combine(resolvedCudaRoot, "bin");
                if (!Directory.Exists(cudaBin))
                {
                    logs.Add($"CUDA bin folder not found: {cudaBin}");
                    StaticLogger.Log($"CUDA bin folder not found: {cudaBin}");
                    return logs;
                }

                var targetDir = outputDir ?? Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native");
                Directory.CreateDirectory(targetDir);

                var cudaDlls = new[]
                {
                    "cudart64_12.dll",
                    "cublas64_12.dll",
                    "cublasLt64_12.dll",
                    "cufft64_12.dll",
                    "curand64_10.dll",
                    "cusolver64_11.dll",
                    "cusparse64_12.dll"
                };

                var cudnnDlls = new[]
                {
                    "cudnn64_9.dll",
                    "cudnn_adv64_9.dll",
                    "cudnn_ops64_9.dll",
                    "cudnn_cnn64_9.dll"
                };

                CopyDlls(cudaBin, targetDir, cudaDlls, logs, "CUDA");
                CopyDlls(cudaBin, targetDir, cudnnDlls, logs, "cuDNN", warnWhenMissing: false);

                logs.Add($"CUDA root: {resolvedCudaRoot}");
                StaticLogger.Log($"CUDA root: {resolvedCudaRoot}");
                logs.Add($"Output dir: {targetDir}");
                StaticLogger.Log($"Output dir: {targetDir}");

                return logs;
            });
        }

        private static string? ResolveCudaRoot(string? cudaRoot)
        {
            if (!string.IsNullOrWhiteSpace(cudaRoot) && Directory.Exists(cudaRoot))
            {
                return Path.GetFullPath(cudaRoot);
            }

            var envCuda = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrWhiteSpace(envCuda) && Directory.Exists(envCuda))
            {
                return envCuda;
            }

            var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            foreach (var entry in path.Split(Path.PathSeparator))
            {
                var nvcc = Path.Combine(entry.Trim(), "nvcc.exe");
                if (File.Exists(nvcc))
                {
                    return Directory.GetParent(entry.Trim())?.FullName;
                }
            }

            var defaultPath = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8";
            if (Directory.Exists(defaultPath))
            {
                return defaultPath;
            }

            return null;
        }

        private static void CopyDlls(string sourceDir, string targetDir, string[] dlls, List<string> logs, string label, bool warnWhenMissing = true)
        {
            foreach (var dll in dlls)
            {
                var src = Path.Combine(sourceDir, dll);
                var dest = Path.Combine(targetDir, dll);
                if (File.Exists(src))
                {
                    File.Copy(src, dest, true);
                    logs.Add($"{label}: copied {dll}");
                }
                else if (warnWhenMissing)
                {
                    logs.Add($"{label}: missing {dll}");
                }
            }
        }

        private static string? NormalizeContextName(string? name)
        {
            if (name == null)
            {
                return null;
            }

            var trimmed = name.Trim();
            if (trimmed.StartsWith("/", StringComparison.Ordinal))
            {
                trimmed = trimmed.TrimStart('/');
            }

            return trimmed;
        }
    }
}
